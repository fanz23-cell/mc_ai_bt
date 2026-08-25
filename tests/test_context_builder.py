import json

from mc_ai_bt.context_builder import ContextBuilder
from mc_ai_bt.mission import MissionManager


def test_context_builder_includes_world_missions_and_skills():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json='{"locale":"en-US"}',
    )
    world_json = '{"facts":{"navigation":{"current_place":{"value":"hall"}}}}'
    builder = ContextBuilder(snapshot_provider=lambda _scopes, _age: world_json)

    context = json.loads(builder.build_json(mission, manager.all()))

    assert context["schema"] == "mc_ai_bt.context.v1"
    assert context["mission"]["intent_text"] == "go to test_place"
    assert context["caller_context"] == {"locale": "en-US"}
    assert context["world"]["facts"]["navigation"]["current_place"]["value"] == "hall"
    assert context["missions"][0]["mission_id"] == mission.identity.mission_id
    assert "go_to_place" in {skill["name"] for skill in context["skills"]}
    skills = {skill["name"]: skill for skill in context["skills"]}
    assert skills["go_to_place"]["policy_enabled"] is True
    assert skills["point_at"]["policy_enabled"] is True


def test_context_builder_ignores_invalid_caller_context():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="hello",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="not json",
    )

    context = json.loads(ContextBuilder().build_json(mission, manager.all()))

    assert context["caller_context"] == {}


def test_context_builder_requests_current_world_scopes():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="hello",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    seen = {}

    def snapshot(scopes, max_age):
        seen["scopes"] = scopes
        seen["max_age"] = max_age
        return "{}"

    ContextBuilder(snapshot_provider=snapshot).build_json(mission, manager.all())

    assert "tasks" in seen["scopes"]
    assert "task" not in seen["scopes"]
