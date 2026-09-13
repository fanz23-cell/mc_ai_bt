from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .entity_facts import (
    fact_value,
    find_object_fact,
    min_score_from_args,
    min_count_from_args,
    object_name_from_args,
    object_visibility_evidence,
    people_visibility_evidence,
    person_id_from_args,
)
from .executor import ExecutionResult
from .place_predicates import TRI_FALSE, TRI_TRUE, evaluate_place_predicate
from .visual_check import visual_check_goal_spec


class TriState(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    state: TriState
    message: str

    @property
    def is_true(self) -> bool:
        return self.state is TriState.TRUE


SnapshotProvider = Callable[[tuple[str, ...], float], str]


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    scopes: tuple[str, ...]
    snapshot_scope: str = "robot"
    # FOUND LIVE 2026-09-05 (E1 TURN2): this predicate can ONLY be answered from
    # the execution facts of the mission that just produced it -- _check_snapshot
    # below has no branch for it, so a world-state fallback always returns
    # UNKNOWN no matter what the scopes above say.
    #
    # Those scopes were the problem. Every predicate here declares scopes and a
    # snapshot_scope, which reads as "this is checkable against world state",
    # and for 18 of them that was simply untrue. A planner took the declaration
    # at face value and gated a TURN2 approach on Condition(person_named) --
    # asking whether alias 33 was bound, which IS durable truth, but through a
    # predicate that cannot see it. The mission stalled on "condition UNKNOWN:
    # no verification evidence for person_named" with the answer sitting in
    # entity_aliases the whole time. entity_alias_bound is the predicate that
    # can actually read it.
    #
    # Marking them makes the registry honest and lets planner.py render the
    # rule from data instead of a hand-written list. Runtime behaviour is
    # deliberately unchanged: the snapshot fetch still happens, it just could
    # never have helped.
    execution_only: bool = False


PREDICATE_REGISTRY: dict[str, PredicateSpec] = {
    "robot_at_place": PredicateSpec("robot_at_place", ("navigation",), "navigation"),
    "robot_near_interaction_owner": PredicateSpec(
        "robot_near_interaction_owner",
        ("people", "robot"),
        "robot",
    ),
    "relative_motion_completed": PredicateSpec("relative_motion_completed", ("robot",), "robot"),
    "animation_played": PredicateSpec("animation_played", ("robot",), "robot"),
    "object_visible": PredicateSpec("object_visible", ("objects",), "objects"),
    "person_visible": PredicateSpec("person_visible", ("people",), "people"),
    "person_at_place": PredicateSpec("person_at_place", ("people", "places", "place_regions"), "people"),
    "object_at_place": PredicateSpec("object_at_place", ("objects", "places", "place_regions"), "objects"),
    "entity_following": PredicateSpec("entity_following", ("entities", "people", "objects"), "entities"),
    "person_following": PredicateSpec("person_following", ("people",), "people"),
    "entity_at_place": PredicateSpec("entity_at_place", ("entities", "people", "places", "place_regions"), "entities", execution_only=True),
    "reach_completed": PredicateSpec("reach_completed", ("robot", "objects", "people", "interactions"), "robot", execution_only=True),
    "track_entity_arm_completed": PredicateSpec("track_entity_arm_completed", ("robot", "objects", "people", "entities"), "robot", execution_only=True),
    "contact_detected": PredicateSpec("contact_detected", ("robot", "objects", "people", "interactions"), "robot", execution_only=True),
    "target_state_changed": PredicateSpec("target_state_changed", ("robot", "objects", "interactions"), "robot", execution_only=True),
    "axis_aligned": PredicateSpec("axis_aligned", ("robot", "objects", "interactions"), "robot", execution_only=True),
    "axis_motion_completed": PredicateSpec("axis_motion_completed", ("robot", "objects", "interactions"), "robot", execution_only=True),
    "distance_maintained": PredicateSpec("distance_maintained", ("robot", "entities", "people", "objects"), "robot", execution_only=True),
    "pose_held": PredicateSpec("pose_held", ("robot", "objects", "people", "interactions"), "robot", execution_only=True),
    "oscillation_completed": PredicateSpec("oscillation_completed", ("robot", "objects", "people", "interactions"), "robot", execution_only=True),
    "retracted": PredicateSpec("retracted", ("robot",), "robot", execution_only=True),
    "touch_completed": PredicateSpec("touch_completed", ("robot", "objects", "people", "interactions"), "robot", execution_only=True),
    "human_confirmation": PredicateSpec("human_confirmation", ("tasks",), "tasks", execution_only=True),
    # locate_entity/get_pose/search_for_entity/check_relation (mc_embodied_skills,
    # RosActionSkillProvider) populate these via the generic PREDICATE_REGISTRY path
    # below -- their evidence already carries a "matched" bool, exactly the shape
    # _direct_predicate_result expects, so no bespoke handler is needed the way
    # object_visible/person_visible have one. mc_world_state itself never writes a
    # fact under these names, so the world-state snapshot fallback always reports
    # UNKNOWN for them; execution facts (the action that just ran) are the only real
    # source, same as animation_played/robot_at_place above.
    "entity_located": PredicateSpec("entity_located", ("objects", "people", "entities"), "objects", execution_only=True),
    "pose_available": PredicateSpec("pose_available", ("objects", "people", "entities"), "objects", execution_only=True),
    "relation_checked": PredicateSpec("relation_checked", ("objects", "people", "entities"), "objects", execution_only=True),
    "search_for_entity_completed": PredicateSpec(
        "search_for_entity_completed", ("objects", "people", "entities"), "objects", execution_only=True),
    # approach_entity (mc_embodied_skills) -- navigates to a located entity's live
    # position via nav2's raw NavigateToPose, not a preconfigured place (go_to_place).
    "entity_approached": PredicateSpec("entity_approached", ("objects", "people", "entities"), "objects", execution_only=True),
    # wait_for_participant (mc_embodied_skills) -- polls mc_world_state's "people"
    # scope directly until a matching SemanticPerson fact appears; evidence already
    # carries "matched", same generic path as entity_located et al. above.
    "participant_ready": PredicateSpec("participant_ready", ("people",), "people", execution_only=True),
    # face_entity (mc_embodied_skills) -- rotates in place via SimpleMove/Spin; v1
    # deliberately has no hard semantic re-verification (no "is the robot actually
    # facing X" world-state fact exists yet, same caveat reach_to's own evidence
    # lives with for some of its fields), so semantic_verified is always False in
    # its evidence -- but "matched" still reflects whether the Spin action itself
    # reported success, which is what this generic path reads.
    "entity_faced": PredicateSpec("entity_faced", ("robot", "objects", "people", "entities"), "robot", execution_only=True),
    # remember_person (mc_embodied_skills) -- only ever produces evidence after actually
    # locating the person (same _locate_with_scan real-evidence contract search_for_entity/
    # locate_entity use); "matched" reflects whether a person was genuinely found and the
    # name bound to it, same generic path as entity_located et al. above.
    "person_named": PredicateSpec("person_named", ("people", "people_names"), "people_names", execution_only=True),
    # scan_room (mc_embodied_skills) -- a bounded turn-and-count sweep; "matched"
    # reflects whether the sweep could query world state at all, same generic path
    # as entity_located et al. above. observed_track_count/frames_checked (renamed
    # 2026-09-03 from min_people_count -- see node.py's own comment: the count can
    # be an OVERcount from track-ID churn, not just an undercount, so it is not a
    # lower bound) get a dedicated branch in _check_execution_facts below (2026-09-08,
    # produced-data-closure round) that folds a hedged summary of them into this
    # predicate's own CheckResult.message -- CORRECTED comment: query_world.py does
    # NOT consume this data (confirmed by direct read, zero hits for
    # observed_track_count/room_scanned in that file); before this round nothing did.
    "room_scanned": PredicateSpec("room_scanned", ("people",), "people", execution_only=True),
    # remember_entity (mc_embodied_skills, C.2) -- bespoke handler below (not the
    # generic PREDICATE_REGISTRY path): the evidence/world-state shape is
    # {alias, entity_id, entity_class, created_by}, with no "matched"/"state"/
    # "value" field for the generic _direct_predicate_result to key off of, and
    # a real check needs to compare the SPECIFIC alias (and, if given, entity_id)
    # this goal_spec asked for against what was actually bound -- "something got
    # bound" is not the same claim as "the alias I asked about got bound".
    # FOUND 2026-09-02 (GPT review, confirmed by direct code read): this entry
    # did not exist until now, so a goal_spec using this predicate could never
    # resolve past UNKNOWN -- a remember_entity mission's own C.2 result could
    # never become the FINAL mission SUCCEEDED via its own natural predicate
    # (the live test that round had to fall back to the unrelated person_named
    # predicate to get a real SUCCEEDED at all).
    "entity_alias_bound": PredicateSpec("entity_alias_bound", ("entity_aliases",), "entity_aliases"),
}


class GoalChecker:
    def __init__(self, snapshot_provider: SnapshotProvider | None = None) -> None:
        self._snapshot_provider = snapshot_provider

    def check_json(
        self,
        goal_spec_json: str,
        execution: ExecutionResult,
    ) -> CheckResult:
        try:
            goal_spec = json.loads(goal_spec_json)
        except json.JSONDecodeError as exc:
            return CheckResult(TriState.FALSE, f"invalid goal_spec_json: {exc}")
        return self.check(goal_spec, execution)

    def check(self, goal_spec: dict[str, Any], execution: ExecutionResult) -> CheckResult:
        if not execution.success:
            return CheckResult(TriState.FALSE, f"execution failed: {execution.message}")

        goal_type = goal_spec.get("type")
        if goal_type == "human":
            return self._check_human(goal_spec, execution)
        if goal_type == "visual":
            parsed = visual_check_goal_spec(goal_spec)
            if parsed and parsed.get("type") != "visual":
                return self._check_structured(parsed, execution)
            return CheckResult(TriState.UNKNOWN, "visual goal check is not wired for this query")
        if goal_type == "hybrid":
            return self._check_structured(goal_spec, execution)
        if goal_type == "structured":
            return self._check_structured(goal_spec, execution)
        return CheckResult(TriState.FALSE, f"unknown goal type: {goal_type}")

    def _check_human(
        self,
        goal_spec: dict[str, Any],
        execution: ExecutionResult,
    ) -> CheckResult:
        verification = goal_spec.get("verification") if isinstance(goal_spec.get("verification"), dict) else {}
        mode = str(verification.get("mode") or "implicit").strip().lower()
        if mode.startswith("implicit"):
            return CheckResult(TriState.TRUE, "human/implicit goal accepted after execution")
        if mode in {"confirmation", "human_confirmation", "operator_confirmation", "explicit"}:
            direct = _human_confirmation_result(
                execution.facts.get("human_confirmation"),
                goal_spec,
                source="execution facts",
            )
            if direct.state is not TriState.UNKNOWN:
                return direct
            snapshot = self._snapshot("human_confirmation")
            if snapshot:
                tasks = snapshot.get("facts", {}).get("tasks", {})
                snapshot_result = _human_confirmation_result(
                    _snapshot_value(tasks.get("last_human_confirmation")),
                    goal_spec,
                    source="world state",
                )
                if snapshot_result.state is not TriState.UNKNOWN:
                    return snapshot_result
            return CheckResult(TriState.UNKNOWN, "human confirmation evidence missing")
        return CheckResult(TriState.UNKNOWN, f"unknown human verification mode: {mode}")

    def _check_structured(
        self,
        goal_spec: dict[str, Any],
        execution: ExecutionResult,
    ) -> CheckResult:
        predicate = str(goal_spec.get("predicate") or "")
        args = goal_spec.get("args") or {}
        facts = execution.facts

        fact_result = self._check_execution_facts(predicate, args, facts)
        if fact_result.state is not TriState.UNKNOWN:
            return fact_result

        snapshot = self._snapshot(predicate)
        if snapshot:
            snapshot_result = self._check_snapshot(predicate, args, snapshot)
            if snapshot_result.state is not TriState.UNKNOWN:
                return snapshot_result

        return CheckResult(TriState.UNKNOWN, f"no verification evidence for {predicate}")

    def _check_execution_facts(
        self,
        predicate: str,
        args: dict[str, Any],
        facts: dict[str, Any],
    ) -> CheckResult:
        if predicate == "robot_at_place":
            expected = str(args.get("name") or "")
            actual = str(facts.get("robot_at_place") or "")
            if actual and actual == expected:
                return CheckResult(TriState.TRUE, f"robot_at_place confirmed: {actual}")
            if actual:
                return CheckResult(TriState.FALSE, f"robot_at_place mismatch: {actual} != {expected}")
            return CheckResult(TriState.UNKNOWN, "robot_at_place fact missing")

        if predicate == "robot_near_interaction_owner":
            if facts.get("robot_near_interaction_owner") is True:
                return CheckResult(TriState.TRUE, "robot_near_interaction_owner confirmed")
            return CheckResult(TriState.UNKNOWN, "robot_near_interaction_owner fact missing")

        if predicate == "relative_motion_completed":
            actual = facts.get("relative_motion_completed")
            return _relative_motion_result(actual, args, source="execution facts")

        if predicate == "animation_played":
            expected = str(args.get("animation") or args.get("name") or "")
            actual = str(facts.get("animation_played") or "")
            if actual and actual == expected:
                return CheckResult(TriState.TRUE, f"animation_played confirmed: {actual}")
            if actual:
                return CheckResult(TriState.FALSE, f"animation_played mismatch: {actual} != {expected}")
            return CheckResult(TriState.UNKNOWN, "animation_played fact missing")

        if predicate == "place_remembered":
            expected = str(args.get("name") or "")
            actual = str(facts.get("place_remembered") or "")
            if actual and actual == expected:
                return CheckResult(TriState.TRUE, f"place_remembered confirmed: {actual}")
            if actual:
                return CheckResult(TriState.FALSE, f"place_remembered mismatch: {actual} != {expected}")
            return CheckResult(TriState.UNKNOWN, "place_remembered fact missing")

        if predicate == "object_visible":
            evidence = object_visibility_evidence(
                _execution_object_entry(args, facts),
                object_name_from_args(args),
                min_score=min_score_from_args(args),
            )
            return _object_visibility_result(evidence, source="execution facts")

        if predicate == "person_visible":
            evidence = people_visibility_evidence(
                _execution_people_entry(facts),
                person_id=person_id_from_args(args),
                min_count=min_count_from_args(args),
            )
            return _people_visibility_result(evidence, source="execution facts")

        if predicate in {"person_at_place", "object_at_place"}:
            direct = _direct_predicate_result(
                predicate,
                args,
                facts.get(predicate),
                source="execution facts",
            )
            if direct.state is not TriState.UNKNOWN:
                return direct
            place_result = evaluate_place_predicate(facts, predicate, args)
            return _place_predicate_check_result(place_result, predicate, source="execution facts")

        if predicate in {"entity_following", "person_following"}:
            return _person_following_result(
                _execution_following_entry(facts, predicate),
                args,
                source="execution facts",
            )

        if predicate == "entity_alias_bound":
            return _entity_alias_bound_result(args, facts.get("entity_alias_bound"), source="execution facts")

        if predicate == "room_scanned":
            # 2026-09-08, produced-data-closure round: observed_track_count is
            # confirmed (this session, direct hop-by-hop trace) to survive intact
            # all the way to this exact point (facts["room_scanned"]) and then be
            # discarded here -- _direct_predicate_result's own generic message
            # ("execution facts confirms room_scanned") never interpolates it, and
            # CheckResult's own frozen 2-field shape has nowhere else to carry it.
            # This wraps that SAME verdict (never recomputes TRUE/FALSE/UNKNOWN)
            # with a hedged summary when the count is present and the sweep
            # succeeded -- see _room_scanned_result's own docstring for why this
            # is a room_scanned-specific branch rather than a change to
            # _direct_predicate_result itself (which many other predicates share
            # unmodified).
            direct = _direct_predicate_result(
                predicate,
                args,
                facts.get(predicate),
                source="execution facts",
            )
            return _room_scanned_result(direct, facts.get(predicate))

        if predicate in PREDICATE_REGISTRY:
            return _direct_predicate_result(
                predicate,
                args,
                facts.get(predicate),
                source="execution facts",
            )

        return CheckResult(TriState.UNKNOWN, f"unknown structured predicate: {predicate}")

    def _snapshot(self, predicate: str) -> dict[str, Any]:
        if self._snapshot_provider is None:
            return {}
        scopes = _scopes_for_predicate(predicate)
        try:
            raw = self._snapshot_provider(scopes, 5.0)
        except Exception:
            return {}
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _check_snapshot(
        self,
        predicate: str,
        args: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> CheckResult:
        facts = snapshot.get("facts") or {}
        if predicate == "robot_at_place":
            nav = facts.get("navigation") or {}
            current = _snapshot_value(nav.get("current_place"))
            expected = str(args.get("name") or "")
            if current and current == expected:
                return CheckResult(TriState.TRUE, f"world state says robot_at_place: {current}")
            if current:
                return CheckResult(TriState.FALSE, f"world state place mismatch: {current} != {expected}")
        if predicate == "robot_near_interaction_owner":
            robot = facts.get("robot") if isinstance(facts.get("robot"), dict) else {}
            near = _snapshot_value(robot.get("near_interaction_owner"))
            if near is True:
                return CheckResult(TriState.TRUE, "world state confirms robot_near_interaction_owner")
            if near is False:
                return CheckResult(TriState.FALSE, "world state says robot is not near interaction owner")
        if predicate == "relative_motion_completed":
            robot = facts.get("robot") if isinstance(facts.get("robot"), dict) else {}
            return _relative_motion_result(
                _snapshot_value(robot.get("last_relative_motion")),
                args,
                source="world state",
            )
        if predicate == "animation_played":
            robot = facts.get("robot") if isinstance(facts.get("robot"), dict) else {}
            actual = str(_snapshot_value(robot.get("last_animation")) or "")
            expected = str(args.get("animation") or args.get("name") or "")
            if actual and actual == expected:
                return CheckResult(TriState.TRUE, f"world state confirms animation_played: {actual}")
            if actual:
                return CheckResult(TriState.FALSE, f"world state animation mismatch: {actual} != {expected}")
        if predicate == "object_visible":
            objects = facts.get("objects") if isinstance(facts.get("objects"), dict) else {}
            object_name = object_name_from_args(args)
            evidence = object_visibility_evidence(
                find_object_fact(objects, object_name),
                object_name,
                min_score=min_score_from_args(args),
            )
            return _object_visibility_result(evidence, source="world state")
        if predicate == "person_visible":
            people = facts.get("people") if isinstance(facts.get("people"), dict) else {}
            evidence = people_visibility_evidence(
                people.get("visible_people"),
                person_id=person_id_from_args(args),
                min_count=min_count_from_args(args),
            )
            return _people_visibility_result(evidence, source="world state")
        if predicate in {"person_at_place", "object_at_place"}:
            scope_name = "people" if predicate == "person_at_place" else "objects"
            scoped = facts.get(scope_name) if isinstance(facts.get(scope_name), dict) else {}
            direct = _direct_predicate_result(
                predicate,
                args,
                scoped.get(predicate),
                source="world state",
            )
            if direct.state is not TriState.UNKNOWN:
                return direct
            place_result = evaluate_place_predicate(facts, predicate, args)
            return _place_predicate_check_result(place_result, predicate, source="world state")
        if predicate in {"entity_following", "person_following"}:
            scope_name = "entities" if predicate == "entity_following" else "people"
            scope = facts.get(scope_name) if isinstance(facts.get(scope_name), dict) else {}
            return _person_following_result(
                scope.get("following"),
                args,
                source="world state",
            )
        if predicate == "entity_alias_bound":
            scoped = facts.get("entity_aliases") if isinstance(facts.get("entity_aliases"), dict) else {}
            alias_key = _normalize_alias(str(args.get("alias") or ""))
            entry = scoped.get(alias_key) if alias_key else None
            return _entity_alias_bound_result(args, entry, source="world state")
        if predicate in PREDICATE_REGISTRY:
            spec = PREDICATE_REGISTRY[predicate]
            scope = facts.get(spec.snapshot_scope) if isinstance(facts.get(spec.snapshot_scope), dict) else {}
            return _direct_predicate_result(
                predicate,
                args,
                scope.get(predicate),
                source="world state",
            )
        return CheckResult(TriState.UNKNOWN, f"world state has no evidence for {predicate}")


def _scopes_for_predicate(predicate: str) -> tuple[str, ...]:
    spec = PREDICATE_REGISTRY.get(predicate)
    return spec.scopes if spec is not None else ()


def _snapshot_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _direct_predicate_result(
    predicate: str,
    args: dict[str, Any],
    entry: Any,
    *,
    source: str,
) -> CheckResult:
    if entry is None:
        return CheckResult(TriState.UNKNOWN, f"{predicate} fact missing")
    value = fact_value(entry)
    if isinstance(value, bool):
        return CheckResult(
            TriState.TRUE if value else TriState.FALSE,
            f"{source} {'confirms' if value else 'contradicts'} {predicate}",
        )
    if not isinstance(value, dict):
        return CheckResult(TriState.UNKNOWN, f"{source} {predicate} fact has unknown shape")

    if value.get("semantic_verified") is False:
        return CheckResult(
            TriState.UNKNOWN,
            f"{source} {predicate} action completed but semantic verification is missing",
        )

    if predicate == "contact_detected" and not _contact_verified(value):
        return CheckResult(
            TriState.UNKNOWN,
            f"{source} contact_detected lacks verified contact evidence",
        )
    if predicate == "target_state_changed" and not _target_state_change_verified(value):
        return CheckResult(
            TriState.UNKNOWN,
            f"{source} target_state_changed lacks target-state verification",
        )

    expected_place = str(args.get("place") or args.get("place_name") or "").strip()
    actual_place = str(value.get("place") or value.get("place_name") or "").strip()
    if expected_place and actual_place and actual_place != expected_place:
        return CheckResult(
            TriState.FALSE,
            f"{source} {predicate} place mismatch: {actual_place} != {expected_place}",
        )

    state = _truthy_state(value, "at_place", "inside", "matched", "state", "value")
    if state is True:
        return CheckResult(TriState.TRUE, f"{source} confirms {predicate}")
    if state is False:
        return CheckResult(TriState.FALSE, f"{source} contradicts {predicate}")
    return CheckResult(TriState.UNKNOWN, f"{source} {predicate} fact is inconclusive")


def _room_scanned_result(direct: CheckResult, entry: Any) -> CheckResult:
    """room_scanned-specific message enrichment (2026-09-08, produced-data-closure
    round). _direct_predicate_result's own generic message ("execution facts
    confirms room_scanned") is correct but discards observed_track_count -- the
    one piece of this skill's own evidence a person actually asked for; CheckResult
    is a frozen 2-field (state, message) shape with nowhere else to carry it, and
    mission.py's mark_terminal/_build_mission_outcome_envelope never receive
    execution.facts at all, so this is the last point in the whole pipeline where
    the count still exists in memory (confirmed this session by direct hop-by-hop
    trace). This wraps the ALREADY-COMPUTED verdict from _direct_predicate_result
    with a hedged summary -- it never recomputes TRUE/FALSE/UNKNOWN, and it never
    invents a number where none exists. Deliberately scoped to room_scanned only:
    _direct_predicate_result itself is shared by many other predicates
    (entity_located, pose_available, contact_detected, ...) and is left untouched,
    so none of them are affected by this change.

    Hedging is deliberate, not incidental: observed_track_count is a live
    camera-tracking count of distinct identities seen during one bounded sweep,
    not a verified exact headcount of the room -- it can OVERcount (track-ID
    churn re-minting a new id for someone already counted) or UNDERcount (someone
    briefly out of every camera angle during the sweep). Never phrase this as a
    ground-truth number of people in the room.
    """
    if direct.state is not TriState.TRUE:
        return direct
    if not isinstance(entry, dict):
        return direct
    count = entry.get("observed_track_count")
    if not isinstance(count, int) or isinstance(count, bool):
        return direct
    identities = "distinct tracked identity" if count == 1 else "distinct tracked identities"
    if entry.get("count_exact") is True:
        hedge = f"counted {count} {identities} during the sweep"
    else:
        hedge = (
            f"observed {count} {identities} during the sweep -- a live "
            "camera-tracking count, not a verified exact headcount; it can "
            "overcount (track-ID churn) or undercount (someone briefly out of "
            "view)"
        )
    return CheckResult(direct.state, f"{direct.message} ({hedge})")


def _place_predicate_check_result(
    result,
    predicate: str,
    *,
    source: str,
) -> CheckResult:
    if result.state == TRI_TRUE:
        return CheckResult(TriState.TRUE, f"{source} confirms {predicate}: {result.reason}")
    if result.state == TRI_FALSE:
        return CheckResult(TriState.FALSE, f"{source} contradicts {predicate}: {result.reason}")
    return CheckResult(TriState.UNKNOWN, f"{source} has no conclusive {predicate} evidence: {result.reason}")


def _execution_following_entry(facts: dict[str, Any], predicate: str) -> Any:
    if predicate == "entity_following" and "entity_following" in facts:
        return facts.get("entity_following")
    if "person_following" in facts:
        return facts.get("person_following")
    entities = facts.get("entities")
    if isinstance(entities, dict):
        return entities.get("following")
    people = facts.get("people")
    if isinstance(people, dict):
        return people.get("following")
    return facts.get("following")


def _person_following_result(
    entry: Any,
    args: dict[str, Any],
    *,
    source: str,
) -> CheckResult:
    if entry is None:
        return CheckResult(TriState.UNKNOWN, "following fact missing")
    value = fact_value(entry)
    if isinstance(value, bool):
        return CheckResult(
            TriState.TRUE if value else TriState.FALSE,
            f"{source} {'confirms' if value else 'contradicts'} person_following",
        )
    if not isinstance(value, dict):
        return CheckResult(TriState.UNKNOWN, f"{source} following fact has unknown shape")

    if value.get("semantic_verified") is False:
        return CheckResult(
            TriState.UNKNOWN,
            f"{source} following action completed but semantic verification is missing",
        )

    expected_entity = str(
        args.get("entity")
        or args.get("entity_id")
        or args.get("person_id")
        or args.get("id")
        or args.get("person")
        or ""
    ).strip()
    actual_entity = str(
        value.get("entity")
        or value.get("entity_id")
        or value.get("person_id")
        or value.get("id")
        or value.get("person")
        or ""
    ).strip()
    if expected_entity and actual_entity and actual_entity != expected_entity:
        return CheckResult(
            TriState.FALSE,
            f"{source} following entity mismatch: {actual_entity} != {expected_entity}",
        )
    state = _truthy_state(value, "following", "active", "matched", "state", "value")
    if state is True:
        return CheckResult(TriState.TRUE, f"{source} confirms following")
    if state is False:
        return CheckResult(TriState.FALSE, f"{source} contradicts following")
    return CheckResult(TriState.UNKNOWN, f"{source} following fact is inconclusive")


def _normalize_alias(raw: str) -> str:
    """Same contract as seattle_lab/mc_embodied_skills/semantic_verifier.py's
    normalize_alias -- entity_aliases is keyed by THIS normalized form (see
    world_facts.py's entity_alias_bound handler, which writes the already-
    normalized alias the skill computed), so a reader comparing a goal_spec's
    raw args.alias against it must apply the identical algorithm or a
    case/whitespace-only difference would silently read as UNKNOWN. No
    shared import path between these two Docker images -- same duplication
    convention this integration already uses for entity_track_by_id."""
    import unicodedata
    return unicodedata.normalize("NFKC", str(raw or "")).strip().casefold()


def _entity_alias_bound_result(
    args: dict[str, Any],
    entry: Any,
    *,
    source: str,
) -> CheckResult:
    if entry is None:
        return CheckResult(TriState.UNKNOWN, "entity_alias_bound fact missing")
    value = fact_value(entry)
    if not isinstance(value, dict):
        return CheckResult(TriState.UNKNOWN, f"{source} entity_alias_bound fact has unknown shape")

    expected_alias = _normalize_alias(str(args.get("alias") or ""))
    actual_alias = _normalize_alias(str(value.get("alias") or ""))
    if expected_alias and actual_alias and actual_alias != expected_alias:
        return CheckResult(
            TriState.FALSE,
            f"{source} entity_alias_bound alias mismatch: {actual_alias!r} != {expected_alias!r}",
        )

    expected_entity_id = str(args.get("entity_id") or "").strip()
    actual_entity_id = str(value.get("entity_id") or "").strip()
    if expected_entity_id and actual_entity_id and actual_entity_id != expected_entity_id:
        return CheckResult(
            TriState.FALSE,
            f"{source} entity_alias_bound entity_id mismatch: {actual_entity_id} != {expected_entity_id}",
        )

    if actual_alias and actual_entity_id:
        return CheckResult(
            TriState.TRUE,
            f"{source} confirms entity_alias_bound: {actual_alias!r} -> {actual_entity_id}",
        )
    return CheckResult(TriState.UNKNOWN, f"{source} entity_alias_bound fact is inconclusive")


def _contact_verified(value: dict[str, Any]) -> bool:
    return (
        _truthy_state(value, "semantic_verified") is True
        and _truthy_state(value, "contact_verified", "contact", "contact_detected", "touched") is True
        and _number_field(value, "contact_duration_sec") >= 0.05
    )


def _target_state_change_verified(value: dict[str, Any]) -> bool:
    return (
        _truthy_state(value, "semantic_verified") is True
        and _truthy_state(value, "state_verified", "target_state_verified") is True
        and _truthy_state(value, "changed", "state_changed", "active", "value") is True
    )


def _number_field(value: dict[str, Any], key: str) -> float:
    try:
        return float(value.get(key))
    except (TypeError, ValueError):
        return 0.0


def _human_confirmation_result(
    entry: Any,
    goal_spec: dict[str, Any],
    *,
    source: str,
) -> CheckResult:
    if entry is None:
        return CheckResult(TriState.UNKNOWN, "human confirmation fact missing")
    value = fact_value(entry)
    if not isinstance(value, dict):
        return CheckResult(TriState.UNKNOWN, f"{source} human confirmation fact has unknown shape")

    expected = str(goal_spec.get("request_id") or "").strip()
    verification = goal_spec.get("verification") if isinstance(goal_spec.get("verification"), dict) else {}
    expected = expected or str(verification.get("request_id") or "").strip()
    actual = str(value.get("request_id") or "").strip()
    if expected and actual and actual != expected:
        return CheckResult(
            TriState.FALSE,
            f"{source} human confirmation request mismatch: {actual} != {expected}",
        )
    approved = value.get("approved")
    if isinstance(approved, bool):
        if approved:
            return CheckResult(TriState.TRUE, f"{source} confirms human approval")
        return CheckResult(TriState.FALSE, f"{source} says human confirmation was not approved")
    decision = str(value.get("decision") or "").strip().upper()
    if decision in {"1", "APPROVED"}:
        return CheckResult(TriState.TRUE, f"{source} confirms human approval")
    if decision in {"2", "DENIED", "3", "TIMEOUT", "4", "CANCELED"}:
        return CheckResult(TriState.FALSE, f"{source} says human confirmation was {decision}")
    return CheckResult(TriState.UNKNOWN, f"{source} human confirmation fact is inconclusive")


def _truthy_state(value: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key not in value:
            continue
        raw = value.get(key)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {
                "true",
                "yes",
                "active",
                "matched",
                "inside",
                "following",
                "complete",
                "completed",
            }:
                return True
            if lowered in {"false", "no", "inactive", "unmatched", "outside", "not_following", "failed"}:
                return False
    return None


def _relative_motion_result(
    actual: Any,
    args: dict[str, Any],
    *,
    source: str,
) -> CheckResult:
    if not isinstance(actual, dict):
        return CheckResult(TriState.UNKNOWN, "relative_motion_completed fact missing")
    expected_action = str(args.get("action") or "")
    if expected_action:
        observed_action = str(actual.get("action") or "")
        if observed_action != expected_action:
            return CheckResult(
                TriState.FALSE,
                f"{source} relative_motion action mismatch: {observed_action} != {expected_action}",
            )
    if "value" in args:
        try:
            expected_value = float(args.get("value"))
            observed_value = float(actual.get("value"))
        except (TypeError, ValueError):
            return CheckResult(
                TriState.FALSE,
                f"{source} relative_motion value is not numeric: {actual.get('value')}",
            )
        if abs(observed_value - expected_value) > 1e-6:
            return CheckResult(
                TriState.FALSE,
                f"{source} relative_motion value mismatch: {observed_value:g} != {expected_value:g}",
            )
    return CheckResult(TriState.TRUE, f"{source} confirms relative_motion_completed")


def _execution_object_entry(args: dict[str, Any], facts: dict[str, Any]) -> Any:
    object_name = object_name_from_args(args)
    direct = facts.get("object_visible")
    if isinstance(direct, dict):
        if _looks_like_single_object_fact(direct):
            return direct
        found = find_object_fact(direct, object_name)
        if found is not None:
            return found
    elif direct is not None:
        return direct

    objects = facts.get("objects")
    if isinstance(objects, dict):
        return find_object_fact(objects, object_name)
    return None


def _execution_people_entry(facts: dict[str, Any]) -> Any:
    if "person_visible" in facts:
        return facts.get("person_visible")
    if "visible_people" in facts:
        return facts.get("visible_people")
    people = facts.get("people")
    if isinstance(people, dict):
        return people.get("visible_people", people)
    return None


def _looks_like_single_object_fact(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "value",
            "object_name",
            "visible",
            "score",
            "position",
        )
    )


def _object_visibility_result(
    evidence: dict[str, Any],
    *,
    source: str,
) -> CheckResult:
    reason = str(evidence.get("reason") or "unknown")
    expected = str(evidence.get("expected") or "")
    observed = evidence.get("observed")
    if evidence.get("matched") is True:
        return CheckResult(TriState.TRUE, f"{source} confirms object_visible: {expected}")
    if evidence.get("matched") is False:
        return CheckResult(
            TriState.FALSE,
            f"{source} contradicts object_visible for {expected}: {reason} ({observed})",
        )
    return CheckResult(TriState.UNKNOWN, f"{source} has no conclusive object_visible evidence: {reason}")


def _people_visibility_result(
    evidence: dict[str, Any],
    *,
    source: str,
) -> CheckResult:
    reason = str(evidence.get("reason") or "unknown")
    observed = evidence.get("observed")
    if evidence.get("matched") is True:
        return CheckResult(TriState.TRUE, f"{source} confirms person_visible")
    if evidence.get("matched") is False:
        return CheckResult(
            TriState.FALSE,
            f"{source} contradicts person_visible: {reason} ({observed})",
        )
    return CheckResult(TriState.UNKNOWN, f"{source} has no conclusive person_visible evidence: {reason}")
