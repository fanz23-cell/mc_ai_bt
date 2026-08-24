from mc_ai_bt.world_facts import world_fact_updates_for_execution


def test_world_fact_updates_map_navigation_success_to_current_place():
    updates = world_fact_updates_for_execution({"robot_at_place": "kitchen"})

    assert len(updates) == 1
    assert updates[0].scope == "navigation"
    assert updates[0].key == "current_place"
    assert updates[0].value == "kitchen"


def test_world_fact_updates_ignore_empty_values():
    updates = world_fact_updates_for_execution(
        {
            "robot_at_place": "",
            "animation_played": "",
            "relative_motion_completed": {},
        }
    )

    assert updates == ()


def test_world_fact_updates_map_multiple_skill_facts():
    updates = world_fact_updates_for_execution(
        {
            "robot_near_interaction_owner": True,
            "relative_motion_completed": {"action": "turn", "value": 90.0},
            "animation_played": "wave",
            "say_submitted": "hello",
        }
    )

    addresses = {(update.scope, update.key) for update in updates}
    assert addresses == {
        ("robot", "near_interaction_owner"),
        ("robot", "last_relative_motion"),
        ("robot", "last_animation"),
        ("robot", "last_utterance"),
    }
