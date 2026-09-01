"""Diagnostic experiment A (3rd-party review, 2026-09-01, following the same review thread
that produced experiment B in test_executor.py::test_reexecuting_after_a_pause_replays_the_
earlier_sequence_prefix and experiment C in test_goal_check.py::test_world_snapshot_can_
confirm_entity_located_when_execution_fact_missing).

Question: for a Condition-as-first/only-node BT (the one shape experiment B already proved
is replay-safe -- nothing with a side effect runs before the Condition, so re-executing the
whole tree from the root on resume costs nothing extra), does a plain mc_world_state fact
write ("UpdateWorldFacts") followed by a bare ResumeMission call actually flip the mission
end-to-end from PAUSED to SUCCEEDED, using only services this repo already fully wires up
today -- with NO new mission-control service at all?

This wires MissionManager + BtExecutor + GoalChecker together, mirroring node.py's real
_run_mission (node.py:384-459) and _handle_resume (node.py:523-542) branching exactly --
quoted inline below at each step -- without importing node.py itself, which is a real
rclpy.Node subclass and cannot be imported outside the ROS container (the same reason no
test_node.py exists anywhere in this suite already; visual_client.py's MissionCheckExecutor
is skipped for the same reason -- GoalChecker alone already implements the same check()/
check_json() interface it wraps, since this goal_spec is never a "visual" type).
"""
import json

from mc_ai_bt.executor import BtExecutor
from mc_ai_bt.goal_check import GoalChecker, TriState
from mc_ai_bt.mission import (
    MissionManager,
    STATE_BLOCKED,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)


class _NoopSkills:
    """The Condition-only tree below never reaches an Action node."""

    def execute_skill(self, name, args, cancel_event=None, *, timeout_sec=None):
        raise AssertionError(f"no Action node should run in this tree, got skill={name!r}")


class _MutableWorldState:
    """Stands in for mc_world_state's GetWorldSnapshot: starts with nothing, `write` mimics
    a real UpdateWorldFacts call landing before the next snapshot fetch."""

    def __init__(self):
        self._facts: dict = {}

    def write(self, scope: str, key: str, value) -> None:
        self._facts.setdefault(scope, {})[key] = {"value": value}

    def __call__(self, _scopes, _max_age):
        return json.dumps({"facts": self._facts})


def _run_mission_once(manager: MissionManager, mission_id: str, bt_json: str, goal_spec_json: str, checks):
    """Mirrors node.py:384-459's _run_mission branching exactly (escalate vs. mark_terminal),
    against the real MissionManager/BtExecutor/GoalChecker classes instead of a live ROS node."""
    execution = BtExecutor().execute_json(bt_json, _NoopSkills(), checks=checks)
    if execution.blocked:
        if execution.needs_decision:
            return manager.pause(mission_id, f"awaiting Omega decision: {execution.message}")
        return manager.mark_terminal(mission_id, state=STATE_BLOCKED, message=execution.message)
    if execution.success:
        check = checks.check_json(goal_spec_json, execution)
        if check.state is TriState.TRUE:
            return manager.mark_terminal(mission_id, state=STATE_SUCCEEDED, message=f"goal check TRUE: {check.message}")
        if check.state is TriState.UNKNOWN:
            return manager.pause(mission_id, f"awaiting Omega decision: goal check UNKNOWN: {check.message}")
        return manager.mark_terminal(mission_id, state=STATE_FAILED, message=f"goal check FALSE: {check.message}")
    return manager.mark_terminal(mission_id, state=STATE_FAILED, message=execution.message)


def test_world_state_write_plus_bare_resume_flips_a_condition_first_mission_to_succeeded():
    world_state = _MutableWorldState()
    checks = GoalChecker(world_state)
    manager = MissionManager()

    bt_json = json.dumps({"type": "Condition", "predicate": "robot_at_place", "args": {"name": "kitchen"}})
    goal_spec_json = json.dumps(
        {"type": "structured", "predicate": "robot_at_place", "args": {"name": "kitchen"},
         "verification": {"mode": "world_state_or_nav_result"}}
    )

    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the kitchen", source="voice", operator_id="user",
        parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
    )
    manager.set_plan(mission.identity.mission_id, bt_json, goal_spec_json)
    mission_id = mission.identity.mission_id

    # First run: mc_world_state has nothing for navigation.current_place yet -- the
    # Condition comes back UNKNOWN (evidentiary gap, not a confirmed FALSE), and the mission
    # pauses awaiting Omega's decision, exactly like a real robot's mission would.
    first = _run_mission_once(manager, mission_id, bt_json, goal_spec_json, checks)
    assert first.mission.state == STATE_PAUSED
    assert "awaiting Omega decision" in first.message

    # The evidence injection this experiment is actually testing: a plain fact write, the
    # same shape mc_world_state/UpdateWorldFacts.srv already supports today, with NO new
    # mission-control service, no decision_id, nothing mission-specific at all.
    world_state.write("navigation", "current_place", "kitchen")

    # Bare resume -- ResumeMission.srv's real, current, evidence-free shape (mission_id +
    # reason only). No new service. mission.py's own resume() only ever writes status_text.
    resumed = manager.resume(mission_id, "camera confirmed the robot is in the kitchen")
    assert resumed.mission.state == STATE_RUNNING

    # Second run: the exact same Condition-only tree, re-executed from its root (there is no
    # checkpoint -- see experiment B) -- but this time the fact is there, so it resolves TRUE
    # and the mission goes all the way to SUCCEEDED with zero new services.
    second = _run_mission_once(manager, mission_id, bt_json, goal_spec_json, checks)
    assert second.mission.state == STATE_SUCCEEDED
    assert "goal check TRUE" in second.message


def test_bare_resume_without_new_evidence_escalates_to_blocked_on_the_very_next_pause():
    # The other half of the same experiment, proving the anti-thrash hazard experiment A's
    # own success path depends on: if the evidence write in the test above is skipped (Omega
    # resumes without anything actually changing), the SAME Condition re-evaluates to the
    # SAME UNKNOWN and produces the SAME pause reason -- and mission.py's own
    # _resumed_from_reason safeguard (mission.py:210-219; see also test_mission_manager.py::
    # test_resume_then_identical_repause_escalates_to_blocked_not_silent_limbo) turns that
    # into a permanent STATE_BLOCKED on the very NEXT pause -- not a second harmless PAUSED,
    # and not requiring a third resume attempt to discover.
    world_state = _MutableWorldState()
    checks = GoalChecker(world_state)
    manager = MissionManager()

    bt_json = json.dumps({"type": "Condition", "predicate": "robot_at_place", "args": {"name": "kitchen"}})
    goal_spec_json = json.dumps(
        {"type": "structured", "predicate": "robot_at_place", "args": {"name": "kitchen"},
         "verification": {"mode": "world_state_or_nav_result"}}
    )
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the kitchen", source="voice", operator_id="user",
        parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
    )
    manager.set_plan(mission.identity.mission_id, bt_json, goal_spec_json)
    mission_id = mission.identity.mission_id

    first = _run_mission_once(manager, mission_id, bt_json, goal_spec_json, checks)
    assert first.mission.state == STATE_PAUSED

    # No UpdateWorldFacts write here -- Omega resumes on nothing but its own say-so.
    manager.resume(mission_id, "Omega said it looked fine, no new evidence actually written")
    second = _run_mission_once(manager, mission_id, bt_json, goal_spec_json, checks)

    assert second.mission.state == STATE_BLOCKED
    assert "resuming again will not resolve this" in second.mission.status_text
