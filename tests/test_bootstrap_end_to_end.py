from mc_ai_bt.local_smoke import run_smoke


def test_bootstrap_end_to_end_intents():
    cases = {
        "go to test_place": {"robot_at_place": "test_place"},
        "come to me": {"robot_near_interaction_owner": True},
        "move left 90": {"relative_motion_completed": {"action": "left", "value": 90.0}},
        "wave hello": {"animation_played": "wave"},
        "look left": {"animation_played": "look_at"},
        "point at the test_object": {"animation_played": "point_at"},
        "去测试地点": {"robot_at_place": "测试地点"},
        "左转90度": {"relative_motion_completed": {"action": "left", "value": 90.0}},
        "挥手": {"animation_played": "wave"},
        "看左边": {"animation_played": "look_at"},
        "指一下测试物体": {"animation_played": "point_at"},
    }

    for intent, expected in cases.items():
        result = run_smoke(intent)

        assert result.ok, intent
        for key, value in expected.items():
            assert result.facts[key] == value


def test_bootstrap_visual_check_succeeds_with_world_json():
    result = run_smoke(
        "find the test_object",
        world_json=(
            '{"facts":{"objects":{"object:test_object":{"value":'
            '{"object_name":"test_object","score":0.9}}}}}'
        ),
    )

    assert result.ok
    assert result.stage == "done"


def test_bootstrap_visual_check_blocks_without_world_json():
    result = run_smoke("find the test_object")

    assert not result.ok
    assert result.stage == "executor"
    assert "UNKNOWN" in result.message


def test_bootstrap_compound_physical_task_succeeds():
    result = run_smoke("go to test_place then wave")

    assert result.ok
    assert result.facts["robot_at_place"] == "test_place"
    assert result.facts["animation_played"] == "wave"


def test_bootstrap_chinese_compound_physical_task_succeeds():
    result = run_smoke("去测试地点然后挥手")

    assert result.ok
    assert result.facts["robot_at_place"] == "测试地点"
    assert result.facts["animation_played"] == "wave"


def test_bootstrap_compound_visual_task_succeeds_with_world_json():
    result = run_smoke(
        "go to test_place then find the test_object",
        world_json=(
            '{"facts":{"objects":{"object:test_object":{"value":'
            '{"object_name":"test_object","score":0.9}}}}}'
        ),
    )

    assert result.ok
    assert result.facts["robot_at_place"] == "test_place"
