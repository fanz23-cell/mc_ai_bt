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


def test_world_fact_updates_map_person_named_to_a_separate_scope():
    # FOUND LIVE 2026-08-31: naming a person used to write nothing at all -- a name
    # binding gets its own scope, keyed by person_id, so the next unrelated
    # pose-detection frame (people/semantic_people) never clobbers it.
    updates = world_fact_updates_for_execution(
        {"person_named": {"matched": True, "person_id": "person:track_7", "name": "Alice", "target": "left"}}
    )

    assert len(updates) == 1
    assert updates[0].scope == "people_names"
    assert updates[0].key == "person:track_7"
    assert updates[0].value == {"person_id": "person:track_7", "name": "Alice"}


def test_world_fact_updates_ignore_person_named_without_person_id_or_name():
    updates = world_fact_updates_for_execution({"person_named": {"matched": False}})

    assert updates == ()


def test_world_fact_updates_map_entity_alias_bound_to_the_entity_aliases_scope():
    # C.2: remember_entity's own, real write path -- a separate scope from
    # entity_tracks on purpose, same reasoning person_named/people_names had:
    # a name binding must not be clobbered by the next unrelated detection.
    updates = world_fact_updates_for_execution(
        {"entity_alias_bound": {
            "alias": "33", "entity_id": "person_aaaa1111", "entity_class": "person",
            "created_by": "mc_embodied_skills.skill.remember_entity",
        }}
    )

    assert len(updates) == 1
    assert updates[0].scope == "entity_aliases"
    assert updates[0].key == "33"
    assert updates[0].value == {
        "alias": "33", "entity_id": "person_aaaa1111", "entity_class": "person",
        "created_by": "mc_embodied_skills.skill.remember_entity",
    }


def test_world_fact_updates_ignore_entity_alias_bound_without_alias_or_entity_id():
    assert world_fact_updates_for_execution({"entity_alias_bound": {"entity_class": "person"}}) == ()
    assert world_fact_updates_for_execution({"entity_alias_bound": {"alias": "33"}}) == ()
