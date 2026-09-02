"""D v1 end-to-end: proves the actual point of the whole design -- when a physical
predicate's DecisionBroker strategy resolves within the fast window, the mission
NEVER needs mission.py's pause()/resume() at all. Contrast with
test_pause_resume_evidence_injection.py's experiment A/B, which proved the OLD
path (needs_decision -> PAUSED -> bare resume) replays any Sequence prefix -- here
the same Action-then-Condition shape runs its Action exactly once, because the
whole thing never leaves one BtExecutor.execute() call.

Wires BtExecutor + GoalChecker + DecisionBroker directly (skips MissionManager --
that layer was already exercised in test_pause_resume_evidence_injection.py; this
file isolates the executor-level claim: does inline resolution avoid ever setting
needs_decision at all, for a Sequence-direct Condition, and does it correctly NOT
apply inside Parallel).

Also proves the 4th-party correctness review's specific regressions stay fixed:
a service outage never reads as a confirmed FALSE, and two different targets in
the same mission run never contaminate each other -- both only checkable at this
end-to-end level, since the bug lived in what got written to (and read back from)
shared state between two resolve() calls.
"""
import json

import pytest

pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

from mc_ai_bt.decision_broker import DecisionBroker, LocateResult  # noqa: E402
from mc_ai_bt.executor import BtExecutor, DecisionOutcome, ExecutionResult  # noqa: E402
from mc_ai_bt.goal_check import CheckResult, GoalChecker, TriState  # noqa: E402
from mc_ai_bt.node import AiBtNode  # noqa: E402


class _CountingSkills:
    def __init__(self):
        self.said = []

    def execute_skill(self, name, args, cancel_event=None, *, timeout_sec=None):
        if name != "say":
            return ExecutionResult(False, f"unsupported skill: {name}")
        self.said.append(args.get("text"))
        return ExecutionResult(True, "said", {})


class _FixedLocator:
    def __init__(self, result: LocateResult):
        self._result = result
        self.calls = []

    def locate(self, target, *, timeout_sec=2.0):
        self.calls.append(target)
        return self._result


class _PerTargetLocator:
    """Returns a different, fixed LocateResult per target name -- proves two
    different entity_approached checks in the same run never share state."""

    def __init__(self, by_target: dict):
        self._by_target = by_target
        self.calls = []

    def locate(self, target, *, timeout_sec=2.0):
        self.calls.append(target)
        return self._by_target.get(target, LocateResult("NOT_FOUND", reason="unregistered target"))


def _broker(*, object_locator, fast_window_sec=1.0) -> DecisionBroker:
    broker = DecisionBroker.__new__(DecisionBroker)
    broker._node = None
    broker._object_locator = object_locator
    broker._person_visible_checker = None
    broker._fast_window_sec = fast_window_sec
    broker._poll_interval_sec = 0.02
    broker._confirmation_timeout_sec = 1.0
    broker._confirmation_client = None  # unused: this file only exercises PHYSICAL predicates
    return broker


def _tree(target: str = "alice"):
    return {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": f"checking on {target}"}},
            {"type": "Condition", "predicate": "entity_approached", "args": {"target": target}},
        ],
    }


def test_inline_resolution_succeeds_mission_never_needs_pause_action_runs_once():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.9}))
    broker = _broker(object_locator=locator)
    skills = _CountingSkills()
    checker = GoalChecker(lambda _s, _a: "")  # never consulted -- entity_approached is now check-local

    result = BtExecutor().execute(_tree(), skills, checks=checker, decision_resolver=broker)

    assert result.success
    assert result.blocked is False
    assert result.needs_decision is False  # the whole point: no pause/resume was ever needed
    assert skills.said == ["checking on alice"]  # Action ran exactly once -- no replay


def test_inline_resolution_gives_up_falls_back_to_todays_exact_pause_signal():
    locator = _FixedLocator(LocateResult("NOT_FOUND", reason="not found"))
    broker = _broker(object_locator=locator, fast_window_sec=0.1)
    skills = _CountingSkills()
    checker = GoalChecker(lambda _s, _a: "")

    result = BtExecutor().execute(_tree(), skills, checks=checker, decision_resolver=broker)

    assert not result.success
    assert result.blocked is True
    assert result.needs_decision is True  # exactly today's signal -- _run_mission still pauses
    assert skills.said == ["checking on alice"]  # Action still only ran once (no replay yet either)


def test_inline_resolution_stays_unknown_on_service_outage_not_a_false_success():
    # FOUND LIVE 2026-09-01 (4th-party review): a perception outage must fall back to
    # today's pause path (an evidentiary gap), never resolve as a confirmed FALSE.
    locator = _FixedLocator(LocateResult("INCONCLUSIVE", reason="object localization service unavailable"))
    broker = _broker(object_locator=locator, fast_window_sec=0.1)
    skills = _CountingSkills()
    checker = GoalChecker(lambda _s, _a: "")

    result = BtExecutor().execute(_tree(), skills, checks=checker, decision_resolver=broker)

    assert result.needs_decision is True  # NOT a hard FAILED from a misread outage


def test_inline_resolution_does_not_confuse_two_different_targets_across_the_same_broker():
    # FOUND LIVE 2026-09-01 (4th-party review): the first cut wrote a single global
    # WorldState key, so confirming Alice was approached could satisfy an unrelated
    # later check for Bob. Two full BtExecutor runs against the SAME broker instance
    # (as a real mission runner would reuse across missions) must not cross-talk.
    locator = _PerTargetLocator({
        "alice": LocateResult("FOUND", fact={"x": 0.1, "y": 0.0, "z": 0.0, "score": 0.9}),
        "bob": LocateResult("FOUND", fact={"x": 9.0, "y": 0.0, "z": 0.0, "score": 0.9}),
    })
    broker = _broker(object_locator=locator)
    checker = GoalChecker(lambda _s, _a: "")

    alice_result = BtExecutor().execute(_tree("alice"), _CountingSkills(), checks=checker, decision_resolver=broker)
    bob_result = BtExecutor().execute(_tree("bob"), _CountingSkills(), checks=checker, decision_resolver=broker)

    assert alice_result.success
    assert not bob_result.success  # far away -- must not read alice's TRUE
    assert bob_result.needs_decision is False  # resolved FALSE, not an evidentiary gap


def test_inline_resolution_does_not_apply_inside_parallel_even_though_it_would_have_succeeded():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.9}))
    broker = _broker(object_locator=locator)
    skills = _CountingSkills()
    checker = GoalChecker(lambda _s, _a: "")
    tree = {
        "type": "Parallel",
        "children": [
            {"type": "Condition", "predicate": "entity_approached", "args": {"target": "alice"}},
            {"type": "Action", "skill": "say", "args": {"text": "meanwhile"}},
        ],
    }

    result = BtExecutor().execute(tree, skills, checks=checker, decision_resolver=broker)

    assert not result.success
    assert result.needs_decision is True  # fell straight to today's path, resolver never consulted
    assert locator.calls == []  # confirms it, not just the symptom


def test_nested_sequence_inside_fallback_still_resolves_inline():
    # Scope is "serial contexts" (root/Sequence/Fallback, any nesting depth), not
    # literally "direct child of the root" -- both propagate _allow_inline_resolution
    # unchanged since neither introduces the concurrency/orphan-thread hazards
    # Parallel/Retry/Timeout do. This pins that as an intentional, tested shape.
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 0.1, "y": 0.0, "z": 0.0, "score": 0.9}))
    broker = _broker(object_locator=locator)
    checker = GoalChecker(lambda _s, _a: "")
    tree = {
        "type": "Fallback",
        "children": [
            {
                "type": "Sequence",
                "children": [
                    {"type": "Condition", "predicate": "entity_approached", "args": {"target": "alice"}},
                ],
            },
        ],
    }

    result = BtExecutor().execute(tree, _CountingSkills(), checks=checker, decision_resolver=broker)

    assert result.success
    assert result.needs_decision is False


# --- the final mission.goal_spec_json check must ALSO go through DecisionBroker ----
# FOUND LIVE 2026-09-01 (4th-party review), the single biggest coverage gap in the
# first D v1 cut: only BT-internal Condition/GoalCheck nodes were ever offered to the
# resolver -- a mission's own final success criterion (goal_spec_json, checked
# separately in node.py's _run_mission after the whole BT succeeds) bypassed it
# completely. _resolve_final_goal_check touches no `self` state at all (pure
# json.loads + a call on the resolver it's given), so it's callable unbound with
# self=None, same as every other node.py-adjacent test in this suite that avoids
# needing a live rclpy node.

class _FakeMissionResolver:
    def __init__(self, outcome: DecisionOutcome):
        self._outcome = outcome
        self.calls = []

    def resolve(self, *, predicate, args, reason, facts, cancel_event):
        self.calls.append({"predicate": predicate, "args": args})
        return self._outcome


def test_final_goal_check_unknown_is_offered_to_the_resolver_and_can_resolve_true():
    resolver = _FakeMissionResolver(DecisionOutcome("TRUE", "resolved by broker"))
    goal_spec_json = json.dumps({"type": "structured", "predicate": "entity_approached", "args": {"target": "alice"}})
    execution = ExecutionResult(True, "bt succeeded", {})
    check = CheckResult(TriState.UNKNOWN, "no verification evidence for entity_approached")

    outcome = AiBtNode._resolve_final_goal_check(None, resolver, goal_spec_json, execution, check, None)

    assert outcome.state == "TRUE"
    assert resolver.calls == [{"predicate": "entity_approached", "args": {"target": "alice"}}]


def test_final_goal_check_still_unknown_falls_back_to_todays_pause_signal():
    resolver = _FakeMissionResolver(DecisionOutcome("UNKNOWN", "no strategy"))
    goal_spec_json = json.dumps({"type": "structured", "predicate": "robot_at_place", "args": {"name": "kitchen"}})
    execution = ExecutionResult(True, "bt succeeded", {})
    check = CheckResult(TriState.UNKNOWN, "no verification evidence for robot_at_place")

    outcome = AiBtNode._resolve_final_goal_check(None, resolver, goal_spec_json, execution, check, None)

    assert outcome.state == "UNKNOWN"


def test_final_goal_check_handles_a_non_structured_goal_spec_without_crashing():
    # e.g. goal_spec type == "human" -- there is no predicate/args to resolve at all.
    # _resolve_final_goal_check's own job is just to parse and delegate without
    # crashing; a real DecisionBroker.resolve() (tested directly in
    # test_decision_broker.py's test_resolve_returns_unknown_immediately_for_a_
    # predicate_with_no_strategy) is what actually turns predicate="" into UNKNOWN --
    # this fake mirrors that dispatch rule to prove the empty predicate reaches it
    # correctly, not to re-test the rule itself.
    resolver = _FakeMissionResolver(DecisionOutcome("UNKNOWN", "no resolution strategy defined for ''"))
    goal_spec_json = json.dumps({"type": "human", "verification": {"mode": "implicit_conversation"}})
    execution = ExecutionResult(True, "bt succeeded", {})
    check = CheckResult(TriState.UNKNOWN, "human goal check is not conclusive")

    outcome = AiBtNode._resolve_final_goal_check(None, resolver, goal_spec_json, execution, check, None)

    assert outcome.state == "UNKNOWN"
    assert resolver.calls == [{"predicate": "", "args": {}}]  # reached the resolver, didn't crash


def test_final_goal_check_handles_invalid_json_without_crashing():
    resolver = _FakeMissionResolver(DecisionOutcome("TRUE", "should never be reached"))
    execution = ExecutionResult(True, "bt succeeded", {})
    check = CheckResult(TriState.UNKNOWN, "invalid")

    outcome = AiBtNode._resolve_final_goal_check(None, resolver, "not json", execution, check, None)

    assert outcome.state == "UNKNOWN"
    assert resolver.calls == []
