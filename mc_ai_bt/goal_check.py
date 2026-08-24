from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .entity_facts import (
    find_object_fact,
    min_score_from_args,
    min_count_from_args,
    object_name_from_args,
    object_visibility_evidence,
    people_visibility_evidence,
    person_id_from_args,
)
from .executor import ExecutionResult
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
            return CheckResult(TriState.TRUE, "human/implicit goal accepted after execution")
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
        return CheckResult(TriState.UNKNOWN, f"world state has no evidence for {predicate}")


def _scopes_for_predicate(predicate: str) -> tuple[str, ...]:
    if predicate == "robot_at_place":
        return ("navigation",)
    if predicate == "robot_near_interaction_owner":
        return ("people", "robot")
    if predicate == "relative_motion_completed":
        return ("robot",)
    if predicate == "animation_played":
        return ("robot",)
    if predicate == "object_visible":
        return ("objects",)
    if predicate == "person_visible":
        return ("people",)
    return ()


def _snapshot_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


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
