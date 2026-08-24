from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


ACTION_NO_ACTION = "NO_ACTION"
ACTION_BLOCK_ACTIVE = "BLOCK_ACTIVE"
ACTION_REPLAN_ACTIVE = "REPLAN_ACTIVE"

SAFETY_EVENT_TYPES = {
    "SAFETY_STOP_OCCURRED",
    "SAFETY_STOP_ACTIVE",
    "E_STOP",
    "EMERGENCY_STOP",
}

REPLAN_EVENT_TYPES = {
    "MISSION_RELEVANT_FACT_CHANGED",
    "GOAL_EVIDENCE_CHANGED",
}


@dataclass(frozen=True)
class TriggerDecision:
    action: str
    message: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_no_action(self) -> bool:
        return self.action == ACTION_NO_ACTION


class TriggerManager:
    """First-pass world-event policy for AI-BT.

    World State owns event detection. TriggerManager owns the decision about
    whether a stable event is worth changing mission execution.
    """

    def handle_world_event(
        self,
        *,
        event_type: str,
        snapshot_id: str = "",
        payload_json: str = "",
    ) -> TriggerDecision:
        event_type = event_type.strip()
        payload = _loads_object(payload_json)

        if event_type in SAFETY_EVENT_TYPES:
            return TriggerDecision(
                ACTION_BLOCK_ACTIVE,
                "safety stop event blocked active mission",
                reason=event_type,
                details={"snapshot_id": snapshot_id, "payload": payload},
            )

        if (
            event_type == "GOAL_EVIDENCE_CHANGED"
            and payload.get("replan_recommended") is False
        ):
            return TriggerDecision(
                ACTION_NO_ACTION,
                "goal evidence changed without replan recommendation",
                reason=event_type,
                details={"snapshot_id": snapshot_id, "payload": payload},
            )

        if event_type in REPLAN_EVENT_TYPES:
            return TriggerDecision(
                ACTION_REPLAN_ACTIVE,
                "mission-relevant world event requests replan",
                reason=event_type,
                details={"snapshot_id": snapshot_id, "payload": payload},
            )

        return TriggerDecision(
            ACTION_NO_ACTION,
            "world event recorded; no mission change",
            reason=event_type or "unknown_event",
            details={"snapshot_id": snapshot_id, "payload": payload},
        )


def trigger_decision_matches_mission(
    decision: TriggerDecision,
    *,
    mission_id: str,
    plan_version: int,
) -> bool:
    payload = decision.details.get("payload")
    if not isinstance(payload, dict):
        return True
    expected_mission_id = str(payload.get("mission_id") or "").strip()
    if expected_mission_id and expected_mission_id != mission_id:
        return False
    if "plan_version" not in payload:
        return True
    try:
        expected_plan_version = int(payload.get("plan_version"))
    except (TypeError, ValueError):
        return False
    return expected_plan_version == int(plan_version)


def _loads_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"invalid_payload_json": raw}
    if isinstance(value, dict):
        return value
    return {"payload": value}
