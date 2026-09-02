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
"""
import json

import pytest

pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

from mc_ai_bt.decision_broker import DecisionBroker  # noqa: E402
from mc_ai_bt.executor import BtExecutor, ExecutionResult  # noqa: E402
from mc_ai_bt.goal_check import GoalChecker  # noqa: E402


class _CountingSkills:
    def __init__(self):
        self.said = []

    def execute_skill(self, name, args, cancel_event=None, *, timeout_sec=None):
        if name != "say":
            return ExecutionResult(False, f"unsupported skill: {name}")
        self.said.append(args.get("text"))
        return ExecutionResult(True, "said", {})


class _MutableWorldState:
    def __init__(self):
        self._facts: dict = {}

    def write(self, scope: str, key: str, value) -> None:
        self._facts.setdefault(scope, {})[key] = {"value": value}

    def __call__(self, _scopes, _max_age):
        return json.dumps({"facts": self._facts})


class _WorldWriterIntoFake:
    """Stands in for WorldStateWriter -- routes DecisionBroker's writes straight into
    the same _MutableWorldState the GoalChecker's snapshot_provider reads, exactly
    like the real /mc_world_state/update_facts + /mc_world_state/get_snapshot pair
    would (a write landing before the next snapshot fetch)."""

    def __init__(self, world_state: _MutableWorldState):
        self._world_state = world_state

    def update_fact(self, *, source, scope, key, value, merge=False, timeout_sec=0.5):
        self._world_state.write(scope, key, value)
        return True, "ok"


class _FakeObjectLocator:
    def __init__(self, result):
        self._result = result

    def locate(self, object_name, *, timeout_sec=2.0):
        return self._result


def _broker(world_state, *, object_locator, checks, fast_window_sec=1.0) -> DecisionBroker:
    broker = DecisionBroker.__new__(DecisionBroker)
    broker._node = None
    broker._checks = checks
    broker._object_locator = object_locator
    broker._world_writer = _WorldWriterIntoFake(world_state)
    broker._fast_window_sec = fast_window_sec
    broker._poll_interval_sec = 0.02
    broker._confirmation_timeout_sec = 1.0
    broker._confirmation_client = None  # unused: this file only exercises PHYSICAL predicates
    return broker


def _tree():
    return {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "checking on alice"}},
            {"type": "Condition", "predicate": "entity_approached", "args": {"target": "alice"}},
        ],
    }


def test_inline_resolution_succeeds_mission_never_needs_pause_action_runs_once():
    world_state = _MutableWorldState()  # starts empty -> entity_approached is UNKNOWN
    checker = GoalChecker(world_state)
    locator = _FakeObjectLocator(({"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.9}, "found"))
    broker = _broker(world_state, object_locator=locator, checks=checker)
    skills = _CountingSkills()

    result = BtExecutor().execute(_tree(), skills, checks=checker, decision_resolver=broker)

    assert result.success
    assert result.blocked is False
    assert result.needs_decision is False  # the whole point: no pause/resume was ever needed
    assert skills.said == ["checking on alice"]  # Action ran exactly once -- no replay


def test_inline_resolution_gives_up_falls_back_to_todays_exact_pause_signal():
    world_state = _MutableWorldState()  # never gets a fact -- target genuinely never found
    checker = GoalChecker(world_state)
    locator = _FakeObjectLocator((None, "not found"))
    broker = _broker(world_state, object_locator=locator, checks=checker, fast_window_sec=0.1)
    skills = _CountingSkills()

    result = BtExecutor().execute(_tree(), skills, checks=checker, decision_resolver=broker)

    assert not result.success
    assert result.blocked is True
    assert result.needs_decision is True  # exactly today's signal -- _run_mission still pauses
    assert skills.said == ["checking on alice"]  # Action still only ran once (no replay yet either)


def test_inline_resolution_does_not_apply_inside_parallel_even_though_it_would_have_succeeded():
    world_state = _MutableWorldState()
    checker = GoalChecker(world_state)
    # The SAME locator that succeeds in the first test -- proving the difference here
    # is purely the tree position, not the resolver's own capability.
    locator = _FakeObjectLocator(({"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.9}, "found"))
    broker = _broker(world_state, object_locator=locator, checks=checker)
    skills = _CountingSkills()
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
