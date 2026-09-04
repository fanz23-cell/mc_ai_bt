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


def test_context_builder_bounds_mission_history_to_the_most_recent():
    # FOUND LIVE 2026-09-03: this used to serialize EVERY mission the node
    # has EVER handled since its last cold start (replayed from the
    # persisted mission journal on every restart, so it never actually
    # shrinks) -- 204 accumulated missions from this session alone were
    # enough to push a real planner call over gpt-4o-mini's 128000-token
    # limit, live, blocking every subsequent mission on the deployed
    # system. Nothing downstream ever reads context.missions at all
    # (grep-confirmed: not planner.py's prompt, not goal_check.py,
    # anywhere) -- pure unbounded dead weight for zero benefit.
    manager = MissionManager()
    missions = []
    for i in range(15):
        _accepted, _message, mission, _event = manager.submit(
            intent_text=f"mission {i}", source="voice", operator_id="user",
            parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
        )
        missions.append(mission)
    builder = ContextBuilder()

    context = json.loads(builder.build_json(missions[-1], manager.all()))

    assert len(context["missions"]) == 10
    assert context["missions"][-1]["mission_id"] == missions[-1].identity.mission_id
    assert context["missions"][0]["mission_id"] == missions[5].identity.mission_id


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
