from mc_ai_bt.trigger_manager import (
    ACTION_BLOCK_ACTIVE,
    ACTION_NO_ACTION,
    ACTION_REPLAN_ACTIVE,
    TriggerDecision,
    TriggerManager,
    trigger_decision_matches_mission,
)


def test_trigger_manager_ignores_regular_fact_updates():
    decision = TriggerManager().handle_world_event(
        event_type="FACT_UPDATED",
        snapshot_id="snap",
        payload_json='{"scope":"navigation"}',
    )

    assert decision.action == ACTION_NO_ACTION
    assert decision.details["payload"]["scope"] == "navigation"


def test_trigger_manager_blocks_active_on_safety_event():
    decision = TriggerManager().handle_world_event(
        event_type="SAFETY_STOP_OCCURRED",
        snapshot_id="snap",
    )

    assert decision.action == ACTION_BLOCK_ACTIVE
    assert decision.reason == "SAFETY_STOP_OCCURRED"


def test_trigger_manager_marks_mission_relevant_events_for_replan():
    decision = TriggerManager().handle_world_event(
        event_type="GOAL_EVIDENCE_CHANGED",
        payload_json='{"mission_id":"m1"}',
    )

    assert decision.action == ACTION_REPLAN_ACTIVE
    assert decision.details["payload"]["mission_id"] == "m1"


def test_trigger_manager_ignores_goal_evidence_without_replan_recommendation():
    decision = TriggerManager().handle_world_event(
        event_type="GOAL_EVIDENCE_CHANGED",
        payload_json='{"mission_id":"m1","replan_recommended":false}',
    )

    assert decision.action == ACTION_NO_ACTION
    assert decision.reason == "GOAL_EVIDENCE_CHANGED"


def test_trigger_manager_preserves_invalid_payload_for_debugging():
    decision = TriggerManager().handle_world_event(
        event_type="FACT_UPDATED",
        payload_json="{not json",
    )

    assert decision.action == ACTION_NO_ACTION
    assert decision.details["payload"]["invalid_payload_json"] == "{not json"


def test_trigger_decision_matches_current_mission_identity():
    decision = TriggerDecision(
        ACTION_REPLAN_ACTIVE,
        "replan",
        details={"payload": {"mission_id": "m1", "plan_version": 3}},
    )

    assert trigger_decision_matches_mission(
        decision,
        mission_id="m1",
        plan_version=3,
    )
    assert not trigger_decision_matches_mission(
        decision,
        mission_id="m2",
        plan_version=3,
    )
    assert not trigger_decision_matches_mission(
        decision,
        mission_id="m1",
        plan_version=4,
    )
