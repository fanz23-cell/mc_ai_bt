from mc_ai_bt.local_smoke import run_smoke


def test_bootstrap_end_to_end_intents():
    cases = {
        "go to kitchen": {"robot_at_place": "kitchen"},
        "come to me": {"robot_near_interaction_owner": True},
        "move left 90": {"relative_motion_completed": {"action": "left", "value": 90.0}},
        "wave hello": {"animation_played": "wave"},
        "look left": {"animation_played": "look_at"},
        "point at the cup": {"animation_played": "point_at"},
        "去厨房": {"robot_at_place": "kitchen"},
        "左转90度": {"relative_motion_completed": {"action": "left", "value": 90.0}},
        "挥手": {"animation_played": "wave"},
        "看左边": {"animation_played": "look_at"},
        "指一下杯子": {"animation_played": "point_at"},
    }

    for intent, expected in cases.items():
        result = run_smoke(intent)

        assert result.ok, intent
        for key, value in expected.items():
            assert result.facts[key] == value


def test_bootstrap_visual_check_succeeds_with_world_json():
    result = run_smoke(
        "find the cup",
        world_json=(
            '{"facts":{"objects":{"object:cup":{"value":'
            '{"object_name":"cup","score":0.9}}}}}'
        ),
    )

    assert result.ok
    assert result.stage == "done"


def test_bootstrap_visual_check_blocks_without_world_json():
    result = run_smoke("find the cup")

    assert not result.ok
    assert result.stage == "executor"
    assert "UNKNOWN" in result.message


def test_bootstrap_compound_physical_task_succeeds():
    result = run_smoke("go to kitchen then wave")

    assert result.ok
    assert result.facts["robot_at_place"] == "kitchen"
    assert result.facts["animation_played"] == "wave"


def test_bootstrap_chinese_compound_physical_task_succeeds():
    result = run_smoke("去厨房然后挥手")

    assert result.ok
    assert result.facts["robot_at_place"] == "kitchen"
    assert result.facts["animation_played"] == "wave"


def test_bootstrap_compound_visual_task_succeeds_with_world_json():
    result = run_smoke(
        "go to kitchen then find the cup",
        world_json=(
            '{"facts":{"objects":{"object:cup":{"value":'
            '{"object_name":"cup","score":0.9}}}}}'
        ),
    )

    assert result.ok
    assert result.facts["robot_at_place"] == "kitchen"
