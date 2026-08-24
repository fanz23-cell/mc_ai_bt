from __future__ import annotations

import json
from typing import Any

from .mission import (
    Mission,
    STATE_BLOCKED,
    STATE_CANCELED,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_PLANNING,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_SUCCEEDED,
    STATE_UNKNOWN,
    STATE_WAITING_FOR_PERMISSION,
)
from .visual_check import visual_check_goal_spec
from .world_facts import WorldFactUpdate


STATE_NAMES = {
    STATE_UNKNOWN: "UNKNOWN",
    STATE_QUEUED: "QUEUED",
    STATE_PLANNING: "PLANNING",
    STATE_WAITING_FOR_PERMISSION: "WAITING_FOR_PERMISSION",
    STATE_RUNNING: "RUNNING",
    STATE_PAUSED: "PAUSED",
    STATE_SUCCEEDED: "SUCCEEDED",
    STATE_FAILED: "FAILED",
    STATE_CANCELED: "CANCELED",
    STATE_BLOCKED: "BLOCKED",
}

ACTIVE_STATES = {
    STATE_PLANNING,
    STATE_WAITING_FOR_PERMISSION,
    STATE_RUNNING,
    STATE_PAUSED,
}


def task_projection_updates(missions: tuple[Mission, ...]) -> tuple[WorldFactUpdate, ...]:
    ordered = sorted(missions, key=lambda mission: mission.identity.mission_id)
    active = _select_active(ordered)
    index = {
        "schema": "mc_ai_bt.task_projection.v1",
        "active_mission_id": active.identity.mission_id if active else "",
        "counts": _state_counts(ordered),
        "missions": [_mission_summary(mission) for mission in ordered],
    }
    return (
        WorldFactUpdate(
            source="mc_ai_bt.task_projection",
            scope="tasks",
            key="active_mission",
            value=_mission_summary(active) if active else None,
        ),
        WorldFactUpdate(
            source="mc_ai_bt.task_projection",
            scope="tasks",
            key="mission_index",
            value=index,
        ),
    )


def _select_active(missions: list[Mission]) -> Mission | None:
    active = [mission for mission in missions if mission.state in ACTIVE_STATES]
    if not active:
        return None
    active.sort(key=lambda mission: (-_active_rank(mission.state), -mission.priority, mission.identity.mission_id))
    return active[0]


def _active_rank(state: int) -> int:
    if state == STATE_RUNNING:
        return 4
    if state == STATE_PLANNING:
        return 3
    if state == STATE_WAITING_FOR_PERMISSION:
        return 2
    if state == STATE_PAUSED:
        return 1
    return 0


def _state_counts(missions: list[Mission]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mission in missions:
        name = _state_name(mission.state).lower()
        counts[name] = counts.get(name, 0) + 1
    return counts


def _mission_summary(mission: Mission) -> dict[str, Any]:
    goal = _goal_summary(mission.goal_spec_json)
    return {
        "mission_id": mission.identity.mission_id,
        "plan_version": mission.identity.plan_version,
        "execution_id": mission.identity.execution_id,
        "parent_mission_id": mission.identity.parent_mission_id,
        "source": mission.source or mission.identity.source,
        "operator_id": mission.operator_id or mission.identity.operator_id,
        "state": mission.state,
        "state_name": _state_name(mission.state),
        "priority": mission.priority,
        "title": mission.title,
        "status_text": mission.status_text,
        "active_node": mission.active_node,
        "progress": mission.progress,
        "has_plan": bool(mission.bt_json),
        "goal": goal,
        "error_code": mission.error_code,
    }


def _goal_summary(goal_spec_json: str) -> dict[str, Any]:
    if not goal_spec_json:
        return {}
    try:
        goal = json.loads(goal_spec_json)
    except json.JSONDecodeError:
        return {"invalid": True}
    if not isinstance(goal, dict):
        return {"invalid": True}
    projected = visual_check_goal_spec(goal) if goal.get("type") == "visual" else goal
    summary = {
        "type": projected.get("type", ""),
        "predicate": projected.get("predicate", ""),
        "args": projected.get("args", {}),
        "verification": projected.get("verification", {}),
        "summary": projected.get("summary", ""),
    }
    if goal.get("type") == "visual" and projected.get("type") != "visual":
        summary["original_type"] = "visual"
        if goal.get("query"):
            summary["query"] = goal.get("query")
    return {key: value for key, value in summary.items() if value not in ("", {}, [])}


def _state_name(state: int) -> str:
    return STATE_NAMES.get(state, f"STATE_{state}")
