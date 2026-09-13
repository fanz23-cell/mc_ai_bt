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
    # reach_completed is deliberately NOT echoed here (Section 7, doc 61/70):
    # mc_embodied_skills' own RosActionSkillProvider._write_skill_evidence_
    # facts now writes it AUTHORITATIVELY, straight to this exact address,
    # with semantic_verified forced True the instant the real ROS action
    # succeeds -- echoing the skill's own still-False self-report here too
    # would clobber that real fact right back to unverified on every call.
    # contact_detected has no independent producer (no contact sensor), so
    # it's still echoed here unchanged, still correctly semantic_verified=False.
    assert addresses == {
        ("robot", "contact_detected"),
        ("entities", "following"),
        ("entities", "entity_at_place"),
    }


def test_world_fact_updates_does_not_echo_the_position_only_verifiable_predicates():
    # The full family mc_embodied_skills' own producer now covers directly
    # (doc 61/70's _POSITION_ONLY_VERIFIABLE_PREDICATES) -- none of them
    # should come back out of this function at all, regardless of what a
    # skill's own (still-unverified) self-report says.
    updates = world_fact_updates_for_execution(
        {
            "reach_completed": {"arm": "right", "matched": True},
            "retracted": {"arm": "left", "matched": True},
            "pose_held": {"arm": "right", "matched": True},
            "axis_aligned": {"arm": "right", "axis": "x", "matched": True},
            "axis_motion_completed": {"arm": "right", "axis": "x", "matched": True},
            "oscillation_completed": {"arm": "right", "axis": "z", "matched": True},
        }
    )

    assert updates == ()


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


def test_world_fact_updates_never_produce_a_generic_update_for_entity_alias_bound():
    # 2026-09-03 architecture consolidation: entity_alias_bound moved to a
    # dedicated domain command (WorldStateWriter.bind_entity_alias, called
    # directly by skill_adapters.py's _publish_success_facts -- see
    # test_skill_adapters.py for that path) precisely so it stops being a
    # generic WorldFactUpdate indistinguishable from any other fact write.
    # world_fact_updates_for_execution must never emit one for this fact,
    # regardless of shape.
    assert world_fact_updates_for_execution(
        {"entity_alias_bound": {
            "alias": "33", "entity_id": "person_aaaa1111", "entity_class": "person",
            "created_by": "mc_embodied_skills.skill.remember_entity",
        }}
    ) == ()
    assert world_fact_updates_for_execution({"entity_alias_bound": {"entity_class": "person"}}) == ()
    assert world_fact_updates_for_execution({"entity_alias_bound": {"alias": "33"}}) == ()
