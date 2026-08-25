from mc_ai_bt.world_facts import world_fact_updates_for_execution


def test_world_fact_updates_map_navigation_success_to_current_place():
    updates = world_fact_updates_for_execution({"robot_at_place": "test_place"})

    assert len(updates) == 1
    assert updates[0].scope == "navigation"
    assert updates[0].key == "current_place"
    assert updates[0].value == "test_place"


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


def test_world_fact_updates_map_human_confirmation():
    confirmation = {"request_id": "confirm-1", "approved": True}

    updates = world_fact_updates_for_execution({"human_confirmation": confirmation})

    assert len(updates) == 1
    assert updates[0].source == "mc_ai_bt.skill.human_confirmation"
    assert updates[0].scope == "tasks"
    assert updates[0].key == "last_human_confirmation"
    assert updates[0].value == confirmation


def test_world_fact_updates_map_embodied_skill_evidence():
    updates = world_fact_updates_for_execution(
        {
            "reach_completed": {"arm": "right", "matched": True},
            "contact_detected": {"target": "generic_target", "matched": True},
            "entity_following": {"entity": "person:subject", "following": True},
            "entity_at_place": {"entity": "person:subject", "place": "target_place", "at_place": True},
        }
    )

    addresses = {(update.scope, update.key) for update in updates}
    assert addresses == {
        ("robot", "reach_completed"),
        ("robot", "contact_detected"),
        ("entities", "following"),
        ("entities", "entity_at_place"),
    }
