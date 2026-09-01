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

import threading
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


# --- _plan_and_start_async must not block the calling ROS service handler (FOUND LIVE
# 2026-08-31, see _plan_and_start_async's own docstring) --------------------------------
#
# SubmitTaskIntent.srv documents the mission as running asynchronously, but
# _handle_submit/_handle_resume/_handle_reprioritize used to call _plan_and_start directly
# -- the service callback itself blocked on the full LLM planning latency, which is exactly
# how the live incident _plan_with_hard_timeout was written for actually happened (a hung
# planner call held the caller's own request hostage too, not just the mission).
# _plan_with_hard_timeout alone does not fix this: it bounds ONE call's latency, but nothing
# stopped the CALLING thread (the service callback) from blocking on it.


class _BlockingPlanAndStart:
    """Stands in for the real _plan_and_start: proves the caller does not wait for it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.received_mission = None

    def __call__(self, mission) -> None:
        self.started.set()
        self.release.wait(timeout=5.0)
        self.received_mission = mission


class _FakePlanAndStartSelf:
    def __init__(self, plan_and_start) -> None:
        self._plan_and_start = plan_and_start


def test_plan_and_start_async_returns_before_planning_completes():
    fake_plan_and_start = _BlockingPlanAndStart()
    fake_self = _FakePlanAndStartSelf(fake_plan_and_start)
    mission, _missions = _mission()

    started = time.monotonic()
    AiBtNode._plan_and_start_async(fake_self, mission)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"_plan_and_start_async blocked the caller for {elapsed:.2f}s"
    assert fake_plan_and_start.started.wait(timeout=1.0), "background thread never ran _plan_and_start"
    fake_plan_and_start.release.set()
    time.sleep(0.05)
    assert fake_plan_and_start.received_mission is mission
