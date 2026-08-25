from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .entity_facts import (
    fact_value,
    find_object_fact,
    normalise_entity_name,
    object_fact_key,
    object_name_from_args,
    object_visibility_evidence,
    people_visibility_evidence,
)
from .place_predicates import TRI_FALSE, TRI_TRUE, evaluate_place_predicate
from .visual_check import visual_query_goal_spec


CERTAINTY_UNKNOWN = 0
CERTAINTY_TRUE = 1
CERTAINTY_FALSE = 2


@dataclass(frozen=True)
class WorldQueryAnswer:
    success: bool
    message: str
    certainty: int
    answer_text: str
    evidence_json: str = "{}"


class QueryWorldEngine:
    def scopes_for_query(self, query_text: str) -> tuple[str, ...]:
        text = _normalise(query_text)
        if _looks_like_current_place_query(text):
            return ("navigation",)
        if _looks_like_current_task_query(text):
            return ("tasks",)
        if _looks_like_safety_query(text):
            return ("safety",)
        at_place = _extract_at_place_query(text)
        if at_place:
            predicate, _args = at_place
            if predicate == "person_at_place":
                return ("people", "places", "place_regions")
            return ("objects", "places", "place_regions")
        visual_goal = visual_query_goal_spec(query_text)
        predicate = str(visual_goal.get("predicate") or "")
        if predicate == "person_visible":
            return ("people",)
        if predicate == "object_visible":
            return ("objects",)
        return ("navigation", "tasks", "objects", "people")

    def answer_json(self, query_text: str, world_json: str) -> WorldQueryAnswer:
        try:
            snapshot = json.loads(world_json or "{}")
        except json.JSONDecodeError as exc:
            return WorldQueryAnswer(
                False,
                f"invalid world_json: {exc}",
                CERTAINTY_UNKNOWN,
                "I could not read the current world state.",
            )
        if not isinstance(snapshot, dict):
            return WorldQueryAnswer(
                False,
                "world_json root is not an object",
                CERTAINTY_UNKNOWN,
                "I could not read the current world state.",
            )

        text = _normalise(query_text)
        facts = snapshot.get("facts") if isinstance(snapshot.get("facts"), dict) else {}

        if _looks_like_current_place_query(text):
            return self._answer_current_place(facts)
        if _looks_like_current_task_query(text):
            return self._answer_current_task(facts)
        if _looks_like_safety_query(text):
            return self._answer_safety(facts)

        at_place = _extract_at_place_query(text)
        if at_place:
            predicate, args = at_place
            return self._answer_at_place_query(facts, predicate, args)

        visual_goal = visual_query_goal_spec(query_text)
        predicate = str(visual_goal.get("predicate") or "")
        args = visual_goal.get("args") if isinstance(visual_goal.get("args"), dict) else {}
        if predicate == "person_visible":
            return self._answer_people_query(facts)
        if predicate == "object_visible":
            return self._answer_object_query(facts, object_name_from_args(args))

        return WorldQueryAnswer(
            True,
            "unsupported_query",
            CERTAINTY_UNKNOWN,
            "I do not have a reliable world-state answer for that yet.",
            _evidence({"route": "unsupported_query"}),
        )

    def _answer_current_place(self, facts: dict[str, Any]) -> WorldQueryAnswer:
        navigation = facts.get("navigation") if isinstance(facts.get("navigation"), dict) else {}
        entry = navigation.get("current_place")
        place = str(fact_value(entry) or "").strip()
        if place:
            return WorldQueryAnswer(
                True,
                "current_place",
                CERTAINTY_TRUE,
                f"I am at {place}.",
                _evidence({"route": "current_place", "fact": entry}),
            )
        return WorldQueryAnswer(
            True,
            "current_place_unknown",
            CERTAINTY_UNKNOWN,
            "I do not have a fresh current-place fact right now.",
            _evidence({"route": "current_place", "fact": entry}),
        )

    def _answer_current_task(self, facts: dict[str, Any]) -> WorldQueryAnswer:
        tasks = facts.get("tasks") if isinstance(facts.get("tasks"), dict) else {}
        entry = tasks.get("active_mission")
        active = fact_value(entry)
        if active is None:
            return WorldQueryAnswer(
                True,
                "no_active_mission",
                CERTAINTY_TRUE,
                "I do not have an active AI-BT mission right now.",
                _evidence({"route": "current_task", "fact": entry}),
            )
        if isinstance(active, dict):
            title = str(active.get("title") or "the current task")
            state = str(active.get("state_name") or "UNKNOWN").lower()
            status = str(active.get("status_text") or "").strip()
            suffix = f" {status}" if status and status != state else ""
            return WorldQueryAnswer(
                True,
                "active_mission",
                CERTAINTY_TRUE,
                f"I am {state}: {title}.{suffix}",
                _evidence({"route": "current_task", "fact": entry}),
            )
        return WorldQueryAnswer(
            True,
            "active_mission_unknown_shape",
            CERTAINTY_UNKNOWN,
            "I have an active-task fact, but I cannot summarize it reliably.",
            _evidence({"route": "current_task", "fact": entry}),
        )

    def _answer_object_query(
        self,
        facts: dict[str, Any],
        object_name: str,
    ) -> WorldQueryAnswer:
        objects = facts.get("objects") if isinstance(facts.get("objects"), dict) else {}
        entry = find_object_fact(objects, object_name)
        evidence = object_visibility_evidence(entry, object_name)
        if evidence.get("matched") is None:
            return WorldQueryAnswer(
                True,
                "object_unknown",
                CERTAINTY_UNKNOWN,
                f"I do not have fresh evidence for {object_name}.",
                _evidence({"route": "object_query", "object": object_name, "evidence": evidence}),
            )
        if evidence.get("matched") is False:
            return WorldQueryAnswer(
                True,
                "object_not_visible",
                CERTAINTY_FALSE,
                f"I do not currently have reliable visible evidence for {object_name}.",
                _evidence(
                    {
                        "route": "object_query",
                        "object": object_name,
                        "fact": entry,
                        "evidence": evidence,
                    }
                ),
            )

        value = fact_value(entry)
        if not isinstance(value, dict):
            return WorldQueryAnswer(
                True,
                "object_fact_unknown_shape",
                CERTAINTY_UNKNOWN,
                f"I have an object fact for {object_name}, but I cannot summarize it reliably.",
                _evidence({"route": "object_query", "object": object_name, "fact": entry}),
            )

        seen_name = str(value.get("object_name") or object_name)
        score = value.get("score")
        position = value.get("position") if isinstance(value.get("position"), dict) else {}
        position_text = _position_text(position)
        score_text = _score_text(score)
        return WorldQueryAnswer(
            True,
            "object_visible",
            CERTAINTY_TRUE,
            f"I have fresh evidence for {seen_name}{position_text}{score_text}.",
            _evidence({"route": "object_query", "object": object_name, "fact": entry}),
        )

    def _answer_people_query(self, facts: dict[str, Any]) -> WorldQueryAnswer:
        people = facts.get("people") if isinstance(facts.get("people"), dict) else {}
        entry = people.get("visible_people")
        evidence = people_visibility_evidence(entry, min_count=1)
        observed = evidence.get("observed") if isinstance(evidence.get("observed"), dict) else {}
        count = observed.get("count") if isinstance(observed, dict) else None

        if evidence.get("matched") is True:
            noun = "person" if count == 1 else "people"
            count_text = str(count) if count is not None else "at least one"
            return WorldQueryAnswer(
                True,
                "people_visible",
                CERTAINTY_TRUE,
                f"I have fresh evidence for {count_text} visible {noun}.",
                _evidence({"route": "people_query", "fact": entry, "evidence": evidence}),
            )
        if evidence.get("matched") is False:
            return WorldQueryAnswer(
                True,
                "people_not_visible",
                CERTAINTY_FALSE,
                "I do not currently have visible-person evidence.",
                _evidence({"route": "people_query", "fact": entry, "evidence": evidence}),
            )
        return WorldQueryAnswer(
            True,
            "people_unknown",
            CERTAINTY_UNKNOWN,
            "I do not have fresh visible-people evidence right now.",
            _evidence({"route": "people_query", "fact": entry, "evidence": evidence}),
        )

    def _answer_at_place_query(
        self,
        facts: dict[str, Any],
        predicate: str,
        args: dict[str, Any],
    ) -> WorldQueryAnswer:
        result = evaluate_place_predicate(facts, predicate, args)
        place = str(args.get("place") or "").strip()
        entity = str(args.get("person_id") or args.get("object_id") or "").strip()
        label = entity or "that entity"
        if result.state == TRI_TRUE:
            return WorldQueryAnswer(
                True,
                predicate,
                CERTAINTY_TRUE,
                f"Yes. {label} is at {place}.",
                _evidence({"route": predicate, "args": args, "result": result.__dict__}),
            )
        if result.state == TRI_FALSE:
            return WorldQueryAnswer(
                True,
                predicate,
                CERTAINTY_FALSE,
                f"No. I have evidence that {label} is not at {place}.",
                _evidence({"route": predicate, "args": args, "result": result.__dict__}),
            )
        return WorldQueryAnswer(
            True,
            f"{predicate}_unknown",
            CERTAINTY_UNKNOWN,
            f"I do not have reliable place evidence for {label} at {place}.",
            _evidence({"route": predicate, "args": args, "result": result.__dict__}),
        )

    def _answer_safety(self, facts: dict[str, Any]) -> WorldQueryAnswer:
        safety = facts.get("safety") if isinstance(facts.get("safety"), dict) else {}
        active_entries = []
        inactive_entries = []
        for key, entry in safety.items():
            value = fact_value(entry)
            active = _safety_active(value)
            if active is True:
                active_entries.append((key, entry))
            elif active is False:
                inactive_entries.append((key, entry))
        if active_entries:
            keys = ", ".join(key for key, _entry in active_entries)
            return WorldQueryAnswer(
                True,
                "safety_stop_active",
                CERTAINTY_FALSE,
                f"I have an active safety-stop fact: {keys}.",
                _evidence({"route": "safety", "active": active_entries}),
            )
        if inactive_entries:
            return WorldQueryAnswer(
                True,
                "safety_clear",
                CERTAINTY_TRUE,
                "I do not have an active safety-stop fact right now.",
                _evidence({"route": "safety", "inactive": inactive_entries}),
            )
        return WorldQueryAnswer(
            True,
            "safety_unknown",
            CERTAINTY_UNKNOWN,
            "I do not have a fresh safety-stop fact right now.",
            _evidence({"route": "safety"}),
        )


def _normalise(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[?!.,;:]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _looks_like_current_place_query(text: str) -> bool:
    phrases = (
        "where are you",
        "where are we",
        "where am i",
        "current place",
        "current location",
        "your location",
        "你在哪",
        "你在哪里",
        "我们在哪",
        "当前位置",
    )
    return any(phrase in text for phrase in phrases)


def _looks_like_current_task_query(text: str) -> bool:
    phrases = (
        "what are you doing",
        "what is your task",
        "current task",
        "active task",
        "active mission",
        "what mission",
        "你在做什么",
        "当前任务",
        "现在任务",
    )
    return any(phrase in text for phrase in phrases)


def _looks_like_safety_query(text: str) -> bool:
    phrases = (
        "are you safe",
        "is it safe",
        "safety stop",
        "emergency stop",
        "e stop",
        "estop",
        "安全",
        "急停",
        "安全停止",
    )
    return any(phrase in text for phrase in phrases)


def _extract_at_place_query(text: str) -> tuple[str, dict[str, Any]] | None:
    person_patterns = (
        r"\bis person (?P<entity>[a-z0-9:_-]+) (?:at|in|inside) (?:the )?(?P<place>[a-z0-9 _-]+)$",
        r"\bis (?P<entity>person:[a-z0-9:_-]+) (?:at|in|inside) (?:the )?(?P<place>[a-z0-9 _-]+)$",
    )
    for pattern in person_patterns:
        match = re.search(pattern, text)
        if match:
            return (
                "person_at_place",
                {
                    "person_id": _normalise_person_id(match.group("entity")),
                    "place": _clean_place(match.group("place")),
                },
            )

    object_patterns = (
        r"\bis object (?P<entity>[a-z0-9:_-]+) (?:at|in|inside) (?:the )?(?P<place>[a-z0-9 _-]+)$",
        r"\bis (?:the )?(?P<entity>[a-z0-9:_-]+) (?:at|in|inside) (?:the )?(?P<place>[a-z0-9 _-]+)$",
    )
    for pattern in object_patterns:
        match = re.search(pattern, text)
        if match:
            entity = match.group("entity")
            if entity in {"you", "we", "i", "it", "safe", "safety"}:
                return None
            return (
                "object_at_place",
                {
                    "object_id": _normalise_object_id(entity),
                    "place": _clean_place(match.group("place")),
                },
            )
    return None


def _normalise_person_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("person:"):
        return value
    return "person:" + normalise_entity_name(value)


def _normalise_object_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("object:"):
        return value
    return object_fact_key(value)


def _clean_place(raw: str) -> str:
    return re.sub(r"\b(the|a|an|please)\b", " ", raw).strip(" _-")


def _safety_active(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("active", "engaged", "stopped", "stop"):
            if key in value:
                return bool(value.get(key))
        state = str(value.get("state") or value.get("status") or "").strip().lower()
        if not state:
            return None
        return state in {"active", "engaged", "stopped", "stop", "emergency", "e_stop", "estop"}
    if isinstance(value, str):
        state = value.strip().lower()
        if not state:
            return None
        return state in {"true", "active", "engaged", "stopped", "stop", "emergency", "e_stop", "estop"}
    return None


def _position_text(position: dict[str, Any]) -> str:
    if not position:
        return ""
    try:
        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])
    except (KeyError, TypeError, ValueError):
        return ""
    return f" near x={x:.2f}, y={y:.2f}, z={z:.2f}"


def _score_text(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    return f" with confidence {value:.2f}"


def _evidence(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
