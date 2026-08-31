"""Pins AiBtNode._plan_with_hard_timeout: a mission must not be able to sit in
"planning" forever just because the injected LLM client's own timeout failed
to fire -- see that method's docstring for the live incident this fixes
(a real planner call that neither returned nor raised for 4+ minutes, holding
the single active-mission slot the whole time).

Calls the method unbound against a minimal fake `self` rather than a real
AiBtNode: the method only touches self.get_parameter("planner_timeout") and
self._planning, and this package has no rclpy-node test harness (matching
every other *_node.py file in this repo -- see mc_world_state's equivalent
note), so this is the narrowest real thing to test without one.
"""

import time

from mc_ai_bt.context_builder import ContextBuilder
from mc_ai_bt.mission import MissionManager
from mc_ai_bt.node import AiBtNode
from mc_ai_bt.planner import BootstrapPlanner
from mc_ai_bt.planning_pipeline import PlanningPipeline
from mc_ai_bt.policy_guard import PolicyGuard
from mc_ai_bt.validator import PlanValidator


class _HungPlanner:
    """Stands in for the live LLM backend hang this fix was written for: a
    call that never returns and never raises within any timeout the
    underlying client claims to honor."""

    def plan(self, intent_text: str, context_json: str = "") -> str:
        time.sleep(3600)
        return "{}"  # pragma: no cover -- the test never waits this long


class _FakeParameter:
    def __init__(self, value):
        self.value = value


class _FakeNodeSelf:
    def __init__(self, planning: PlanningPipeline, timeout_sec: float):
        self._planning = planning
        self._timeout_sec = timeout_sec

    def get_parameter(self, name: str) -> _FakeParameter:
        assert name == "planner_timeout"
        return _FakeParameter(self._timeout_sec)


def _mission(intent: str = "go to test_place"):
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text=intent,
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json='{"language":"en-US"}',
    )
    return mission, manager.all()


def _pipeline(planner) -> PlanningPipeline:
    return PlanningPipeline(
        planner=planner,
        context_builder=ContextBuilder(),
        validator=PlanValidator(),
        policy_guard=PolicyGuard(),
    )


def test_hung_planner_returns_within_configured_timeout_not_forever():
    mission, missions = _mission()
    fake_self = _FakeNodeSelf(_pipeline(_HungPlanner()), timeout_sec=0.2)
    started = time.monotonic()
    result = AiBtNode._plan_with_hard_timeout(fake_self, mission, missions)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"hard timeout did not bound the call: took {elapsed:.2f}s"
    assert result.ok is False
    assert result.stage == "planner"
    assert "timed out" in result.message


def test_normal_planning_result_passes_through_unaffected():
    mission, missions = _mission()
    fake_self = _FakeNodeSelf(_pipeline(BootstrapPlanner()), timeout_sec=5.0)
    result = AiBtNode._plan_with_hard_timeout(fake_self, mission, missions)
    assert result.ok is True
    assert result.bt_json
