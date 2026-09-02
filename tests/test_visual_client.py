"""visual_client.py's evidence policy (A1): a "structured" goal_spec names a
structured physical predicate whose truth must come from its own designated
authoritative verifier -- never a generative VLM's opinion, even when the
goal_spec's own (planner-authored) verification.mode string claims to allow a
visual fallback. Only a "visual" goal_spec (free-form VQA, no predicate at
all) may legitimately settle via VisualCheck.

MissionCheckExecutor.check()/visual_check() touch no ROS internals directly
(unlike VisualCheckClient, which needs a real rclpy Node -- see
test_pause_resume_evidence_injection.py's own note on this), so fakes are
used for goal_checker/visual_client here, same idiom as the rest of this
suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mc_ai_bt.goal_check import CheckResult, TriState
from mc_ai_bt.visual_client import MissionCheckExecutor, _allows_visual_fallback


@dataclass
class _FakeExecution:
    success: bool = True
    facts: dict[str, Any] = field(default_factory=dict)


class _FakeGoalChecker:
    def __init__(self, result: CheckResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def check(self, goal_spec, execution):
        self.calls.append(goal_spec)
        return self._result


class _FakeVisualClient:
    def __init__(self, result: CheckResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def check(self, *, identity, check, facts, cancel_event=None):
        self.calls.append(check)
        return self._result


def _executor(goal_checker_result: CheckResult, visual_result: CheckResult) -> tuple[MissionCheckExecutor, _FakeGoalChecker, _FakeVisualClient]:
    goal_checker = _FakeGoalChecker(goal_checker_result)
    visual_client = _FakeVisualClient(visual_result)
    executor = MissionCheckExecutor(
        goal_checker=goal_checker, visual_client=visual_client, identity=object(),
    )
    return executor, goal_checker, visual_client


# --- _allows_visual_fallback: the pure policy function ------------------------

def test_structured_goal_spec_never_allows_visual_fallback_even_with_permissive_mode():
    for predicate in ("object_visible", "person_visible", "entity_approached", "robot_at_place"):
        goal_spec = {
            "type": "structured",
            "predicate": predicate,
            "verification": {"mode": "world_state_or_visual_check"},
        }
        assert _allows_visual_fallback(goal_spec) is False, predicate


def test_structured_goal_spec_denies_visual_fallback_even_with_type_visual_looking_mode_string():
    # A planner could set any string it wants into verification.mode; the
    # policy must key off goal_spec["type"], not trust that string at all.
    goal_spec = {"type": "structured", "predicate": "entity_approached", "verification": {"mode": "visual_check_only"}}
    assert _allows_visual_fallback(goal_spec) is False


def test_visual_type_goal_spec_allows_fallback_unconditionally():
    goal_spec = {"type": "visual", "query": "do you see a red cup?"}
    assert _allows_visual_fallback(goal_spec) is True


def test_unrecognized_type_falls_back_to_legacy_mode_string_check():
    # Preserves prior behavior for goal_spec shapes that are neither
    # "structured" nor "visual" -- a no-op change for anything besides the
    # two explicitly-covered types.
    assert _allows_visual_fallback({"verification": {"mode": "world_state_or_visual_check"}}) is True
    assert _allows_visual_fallback({"verification": {"mode": "world_state"}}) is False
    assert _allows_visual_fallback({}) is False


# --- MissionCheckExecutor.check(): the actual call-site behavior --------------

def test_structured_predicate_unknown_never_calls_visual_client():
    executor, goal_checker, visual_client = _executor(
        goal_checker_result=CheckResult(TriState.UNKNOWN, "no fresh evidence"),
        visual_result=CheckResult(TriState.TRUE, "VLM says yes"),
    )
    goal_spec = {
        "type": "structured", "predicate": "entity_approached", "args": {"target": "alice"},
        "verification": {"mode": "world_state_or_visual_check"},
    }
    result = executor.check(goal_spec, _FakeExecution())
    assert result.state is TriState.UNKNOWN
    assert visual_client.calls == []  # never invoked -- the whole point of A1
    assert "no fresh evidence" in result.message


def test_structured_predicate_true_from_goal_checker_short_circuits_before_any_fallback_question():
    executor, goal_checker, visual_client = _executor(
        goal_checker_result=CheckResult(TriState.TRUE, "world state confirms"),
        visual_result=CheckResult(TriState.FALSE, "VLM disagrees"),
    )
    goal_spec = {"type": "structured", "predicate": "object_visible", "verification": {"mode": "world_state_or_visual_check"}}
    result = executor.check(goal_spec, _FakeExecution())
    assert result.state is TriState.TRUE
    assert visual_client.calls == []


def test_visual_type_goal_spec_unknown_from_goal_checker_does_call_visual_client():
    executor, goal_checker, visual_client = _executor(
        goal_checker_result=CheckResult(TriState.UNKNOWN, "no direct fact"),
        visual_result=CheckResult(TriState.TRUE, "VLM confirms"),
    )
    goal_spec = {"type": "visual", "query": "do you see a red cup?"}
    result = executor.check(goal_spec, _FakeExecution())
    assert result.state is TriState.TRUE
    assert len(visual_client.calls) == 1
