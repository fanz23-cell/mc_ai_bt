from mc_ai_bt.trigger_manager import (
    ACTION_BLOCKED,
    ACTION_ASK_CLARIFICATION,
    ACTION_LOCAL_HANDLED,
    ACTION_NO_ACTION,
    ACTION_PLAN_NEW_MISSION,
    ACTION_REPLAN,
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

    assert decision.action == ACTION_BLOCKED
    assert decision.reason == "SAFETY_STOP_OCCURRED"


def test_trigger_manager_marks_mission_relevant_events_for_replan():
    decision = TriggerManager().handle_world_event(
        event_type="GOAL_EVIDENCE_CHANGED",
        payload_json='{"mission_id":"m1"}',
    )

    assert decision.action == ACTION_REPLAN
    assert decision.details["payload"]["mission_id"] == "m1"


def test_trigger_manager_ignores_goal_evidence_without_replan_recommendation():
    decision = TriggerManager().handle_world_event(
        event_type="GOAL_EVIDENCE_CHANGED",
        payload_json='{"mission_id":"m1","replan_recommended":false}',
    )

    assert decision.action == ACTION_LOCAL_HANDLED
    assert decision.reason == "GOAL_EVIDENCE_CHANGED"


def test_trigger_manager_can_request_new_mission_for_idle_social_event():
    decision = TriggerManager().handle_world_event(
        event_type="PERSON_APPROACHED",
        payload_json='{"person_id":"person:1"}',
        has_active_mission=False,
    )

    assert decision.action == ACTION_PLAN_NEW_MISSION
    assert decision.details["payload"]["person_id"] == "person:1"


def test_trigger_manager_handles_social_event_locally_when_mission_is_active():
    decision = TriggerManager().handle_world_event(
        event_type="PERSON_APPROACHED",
        has_active_mission=True,
    )

    assert decision.action == ACTION_LOCAL_HANDLED


def test_trigger_manager_can_request_clarification_for_missing_target():
    decision = TriggerManager().handle_world_event(
        event_type="MISSING_REQUIRED_TARGET",
        payload_json='{"target_role":"object"}',
        has_active_mission=True,
    )

    assert decision.action == ACTION_ASK_CLARIFICATION
    assert decision.details["payload"]["target_role"] == "object"


def test_trigger_manager_preserves_invalid_payload_for_debugging():
    decision = TriggerManager().handle_world_event(
        event_type="FACT_UPDATED",
        payload_json="{not json",
    )

    assert decision.action == ACTION_NO_ACTION
    assert decision.details["payload"]["invalid_payload_json"] == "{not json"


def test_trigger_decision_matches_current_mission_identity():
    decision = TriggerDecision(
        ACTION_REPLAN,
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
