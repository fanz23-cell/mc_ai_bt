import json

from mc_ai_bt.identity import Identity
from mc_ai_bt.mission import Mission, STATE_PAUSED, STATE_QUEUED, STATE_RUNNING
from mc_ai_bt.task_projection import task_projection_updates


def _mission(mission_id: str, state: int, *, priority: int = 1) -> Mission:
    return Mission(
        identity=Identity(mission_id=mission_id, plan_version=1, execution_id=f"{mission_id}-exec"),
        intent_text=f"{mission_id} intent",
        source="test",
        operator_id="user",
        priority=priority,
        allow_queue=True,
        context_json="{}",
        state=state,
        title=f"{mission_id} title",
        status_text="status",
        goal_spec_json=json.dumps(
            {
                "type": "structured",
                "predicate": "robot_at_place",
                "args": {"name": "kitchen"},
                "verification": {"mode": "world_state_or_nav_result"},
            }
        ),
    )


def test_task_projection_selects_running_mission_as_active():
    queued = _mission("queued", STATE_QUEUED, priority=10)
    paused = _mission("paused", STATE_PAUSED, priority=20)
    running = _mission("running", STATE_RUNNING, priority=1)

    updates = task_projection_updates((queued, paused, running))

    active = next(update for update in updates if update.key == "active_mission")
    index = next(update for update in updates if update.key == "mission_index")
    assert active.value["mission_id"] == "running"
    assert index.value["active_mission_id"] == "running"
    assert index.value["counts"] == {"paused": 1, "queued": 1, "running": 1}


def test_task_projection_has_null_active_when_no_active_mission():
    queued = _mission("queued", STATE_QUEUED)

    updates = task_projection_updates((queued,))

    active = next(update for update in updates if update.key == "active_mission")
    assert active.value is None


def test_task_projection_includes_compact_goal_summary():
    running = _mission("running", STATE_RUNNING)

    updates = task_projection_updates((running,))
    active = next(update for update in updates if update.key == "active_mission")

    assert active.value["goal"] == {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "kitchen"},
        "verification": {"mode": "world_state_or_nav_result"},
    }


def test_task_projection_normalizes_visual_goal_for_world_events():
    running = Mission(
        identity=Identity(mission_id="running", plan_version=1, execution_id="exec"),
        intent_text="find cup",
        source="test",
        operator_id="user",
        priority=1,
        allow_queue=True,
        context_json="{}",
        state=STATE_RUNNING,
        title="find cup",
        status_text="running",
        goal_spec_json=json.dumps(
            {
                "type": "visual",
                "query": "do you see the cup?",
                "verification": {"mode": "world_state_or_visual_check"},
            }
        ),
    )

    updates = task_projection_updates((running,))
    active = next(update for update in updates if update.key == "active_mission")

    assert active.value["goal"]["type"] == "structured"
    assert active.value["goal"]["predicate"] == "object_visible"
    assert active.value["goal"]["args"] == {"name": "cup"}
    assert active.value["goal"]["original_type"] == "visual"
    assert active.value["goal"]["query"] == "do you see the cup?"
