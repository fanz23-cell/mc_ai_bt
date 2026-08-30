import json

from mc_ai_bt.executor import ExecutionResult
from mc_ai_bt.goal_check import GoalChecker, TriState


def test_structured_goal_uses_execution_facts():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "test_place"},
        "verification": {"mode": "world_state_or_nav_result"},
    }
    execution = ExecutionResult(True, "arrived", {"robot_at_place": "test_place"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_structured_goal_mismatch_is_false():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "test_place"},
        "verification": {"mode": "world_state_or_nav_result"},
    }
    execution = ExecutionResult(True, "arrived", {"robot_at_place": "office"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.FALSE


def test_missing_evidence_is_unknown_not_false():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "test_place"},
        "verification": {"mode": "world_state_or_nav_result"},
    }
    execution = ExecutionResult(True, "arrived", {})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.UNKNOWN


def test_relative_motion_execution_fact_must_match_expected_action_and_value():
    goal_spec = {
        "type": "structured",
        "predicate": "relative_motion_completed",
        "args": {"action": "left", "value": 90.0},
        "verification": {"mode": "action_result"},
    }

    ok = GoalChecker().check(
        goal_spec,
        ExecutionResult(True, "moved", {"relative_motion_completed": {"action": "left", "value": 90.0}}),
    )
    wrong = GoalChecker().check(
        goal_spec,
        ExecutionResult(True, "moved", {"relative_motion_completed": {"action": "right", "value": 90.0}}),
    )

    assert ok.state is TriState.TRUE
    assert wrong.state is TriState.FALSE


def test_world_snapshot_can_confirm_place_when_execution_fact_missing():
    world_json = json.dumps(
        {
            "facts": {
                "navigation": {
                    "current_place": {
                        "value": "test_place",
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "test_place"},
        "verification": {"mode": "world_state_or_nav_result"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "arrived", {}))

    assert result.state is TriState.TRUE


def test_world_snapshot_can_confirm_robot_action_facts_when_execution_fact_missing():
    world_json = json.dumps(
        {
            "facts": {
                "robot": {
                    "near_interaction_owner": {"value": True},
                    "last_relative_motion": {"value": {"action": "left", "value": 90.0}},
                    "last_animation": {"value": "wave"},
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)

    near = checker.check(
        {
            "type": "structured",
            "predicate": "robot_near_interaction_owner",
            "args": {},
            "verification": {"mode": "world_state_or_nav_result"},
        },
        ExecutionResult(True, "done", {}),
    )
    motion = checker.check(
        {
            "type": "structured",
            "predicate": "relative_motion_completed",
            "args": {"action": "left", "value": 90.0},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(True, "done", {}),
    )
    animation = checker.check(
        {
            "type": "structured",
            "predicate": "animation_played",
            "args": {"animation": "wave"},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(True, "done", {}),
    )

    assert near.state is TriState.TRUE
    assert motion.state is TriState.TRUE
    assert animation.state is TriState.TRUE


def test_world_snapshot_can_confirm_object_visible():
    world_json = json.dumps(
        {
            "facts": {
                "objects": {
                    "object:test_object": {
                        "value": {
                            "object_name": "test_object",
                            "score": 0.82,
                            "position": {"x": 1.0, "y": 0.5, "z": 0.2},
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "test_object", "min_score": 0.5},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_visual_goal_query_can_be_confirmed_from_world_snapshot():
    world_json = json.dumps(
        {
            "facts": {
                "objects": {
                    "object:test_object": {
                        "value": {
                            "object_name": "test_object",
                            "score": 0.82,
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "visual",
        "query": "do you see the test_object?",
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_world_snapshot_marks_low_score_object_visible_false():
    world_json = json.dumps(
        {
            "facts": {
                "objects": {
                    "object:test_object": {
                        "value": {
                            "object_name": "test_object",
                            "score": 0.21,
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "test_object", "min_score": 0.5},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.FALSE


def test_missing_object_visible_evidence_is_unknown():
    checker = GoalChecker(lambda _scopes, _max_age: json.dumps({"facts": {"objects": {}}}))
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "test_object"},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.UNKNOWN


def test_object_visible_can_be_confirmed_by_execution_fact():
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "test_object", "min_score": 0.4},
        "verification": {"mode": "world_state_or_visual_check"},
    }
    execution = ExecutionResult(
        True,
        "checked",
        {"object_visible": {"object_name": "test_object", "score": 0.9}},
    )

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_entity_located_confirmed_by_execution_fact():
    goal_spec = {
        "type": "structured",
        "predicate": "entity_located",
        "args": {"target": "mystery_gadget"},
    }
    execution = ExecutionResult(
        True, "located", {"entity_located": {"matched": True, "target": "mystery_gadget"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_entity_located_missing_evidence_is_unknown_not_false():
    goal_spec = {"type": "structured", "predicate": "entity_located", "args": {"target": "mystery_gadget"}}

    result = GoalChecker().check(goal_spec, ExecutionResult(True, "located", {}))

    assert result.state is TriState.UNKNOWN


def test_search_for_entity_completed_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "search_for_entity_completed", "args": {"target": "widget"}}
    execution = ExecutionResult(
        True, "searched", {"search_for_entity_completed": {"matched": True, "target": "widget"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_relation_checked_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "relation_checked", "args": {"relation": "mug near sink"}}
    execution = ExecutionResult(
        True, "checked", {"relation_checked": {"matched": True, "relation": "mug near sink"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_relation_checked_false_when_not_matched():
    goal_spec = {"type": "structured", "predicate": "relation_checked", "args": {"relation": "mug near sink"}}
    execution = ExecutionResult(
        True, "checked", {"relation_checked": {"matched": False, "relation": "mug near sink"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.FALSE


def test_entity_approached_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "entity_approached", "args": {"target": "person"}}
    execution = ExecutionResult(
        True, "approached", {"entity_approached": {"matched": True, "target": "person"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_place_remembered_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "place_remembered", "args": {"name": "front_desk"}}
    execution = ExecutionResult(True, "remembered", {"place_remembered": "front_desk"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_place_remembered_mismatch_is_false():
    goal_spec = {"type": "structured", "predicate": "place_remembered", "args": {"name": "front_desk"}}
    execution = ExecutionResult(True, "remembered", {"place_remembered": "back_office"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.FALSE


def test_place_remembered_missing_evidence_is_unknown_not_false():
    goal_spec = {"type": "structured", "predicate": "place_remembered", "args": {"name": "front_desk"}}

    result = GoalChecker().check(goal_spec, ExecutionResult(True, "remembered", {}))

    assert result.state is TriState.UNKNOWN


def test_entity_faced_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "entity_faced", "args": {"target": "person"}}
    execution = ExecutionResult(
        True, "faced", {"entity_faced": {"matched": True, "target": "person", "bearing_deg": 12.5}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_entity_faced_missing_evidence_is_unknown_not_false():
    goal_spec = {"type": "structured", "predicate": "entity_faced", "args": {"target": "person"}}

    result = GoalChecker().check(goal_spec, ExecutionResult(True, "faced", {}))

    assert result.state is TriState.UNKNOWN


def test_participant_ready_confirmed_by_execution_fact():
    goal_spec = {"type": "structured", "predicate": "participant_ready", "args": {"role": "customer"}}
    execution = ExecutionResult(
        True, "ready", {"participant_ready": {"matched": True, "person_id": "person:1", "role": "customer"}})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_participant_ready_missing_evidence_is_unknown_not_false():
    goal_spec = {"type": "structured", "predicate": "participant_ready", "args": {"role": "customer"}}

    result = GoalChecker().check(goal_spec, ExecutionResult(True, "ready", {}))

    assert result.state is TriState.UNKNOWN


def test_world_snapshot_can_confirm_person_visible_by_count():
    world_json = json.dumps(
        {
            "facts": {
                "people": {
                    "visible_people": {
                        "value": {
                            "count": 2,
                            "people": [{"id": "person:0"}, {"id": "person:1"}],
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "person_visible",
        "args": {"min_count": 1},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_world_snapshot_can_confirm_person_at_place_with_place_region():
    world_json = json.dumps(
        {
            "facts": {
                "places": {
                    "target_place": {
                        "value": {
                            "name": "target_place",
                            "frame_id": "map",
                            "position": {"x": 10.0, "y": 5.0, "z": 0.0},
                            "occupancy_radius_m": 2.0,
                        }
                    }
                },
                "people": {
                    "person:subject": {
                        "value": {
                            "person_id": "person:subject",
                            "visible": True,
                            "position": {"frame": "map", "x": 11.0, "y": 5.0, "z": 0.0},
                        }
                    }
                },
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)

    result = checker.check(
        {
            "type": "structured",
            "predicate": "person_at_place",
            "args": {"person_id": "person:subject", "place": "target_place"},
            "verification": {"mode": "world_state"},
        },
        ExecutionResult(True, "checked", {}),
    )

    assert result.state is TriState.TRUE


def test_world_snapshot_can_confirm_person_at_place_with_direct_semantic_fact():
    world_json = json.dumps(
        {
            "facts": {
                "people": {
                    "person_at_place": {
                        "value": {
                            "person_id": "person:1",
                            "person": "person",
                            "place": "target_place",
                            "at_place": True,
                            "semantic_verified": True,
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)

    result = checker.check(
        {
            "type": "structured",
            "predicate": "person_at_place",
            "args": {"person": "person", "place": "target_place"},
            "verification": {"mode": "world_state"},
        },
        ExecutionResult(True, "checked", {}),
    )

    assert result.state is TriState.TRUE


def test_world_snapshot_marks_object_at_place_false_outside_region():
    world_json = json.dumps(
        {
            "facts": {
                "places": {
                    "target_place": {
                        "value": {
                            "name": "target_place",
                            "frame_id": "map",
                            "position": {"x": 10.0, "y": 5.0, "z": 0.0},
                            "occupancy_radius_m": 1.0,
                        }
                    }
                },
                "objects": {
                    "object:test_object": {
                        "value": {
                            "object_id": "object:test_object",
                            "visible": True,
                            "position": {"frame": "map", "x": 12.0, "y": 5.0, "z": 0.0},
                        }
                    }
                },
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)

    result = checker.check(
        {
            "type": "structured",
            "predicate": "object_at_place",
            "args": {"object": "test_object", "place": "target_place"},
            "verification": {"mode": "world_state"},
        },
        ExecutionResult(True, "checked", {}),
    )

    assert result.state is TriState.FALSE


def test_person_following_uses_explicit_following_fact():
    world_json = json.dumps(
        {
            "facts": {
                "people": {
                    "following": {
                        "value": {
                            "person_id": "person:subject",
                            "following": True,
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)

    result = checker.check(
        {
            "type": "structured",
            "predicate": "person_following",
            "args": {"person_id": "person:subject"},
            "verification": {"mode": "world_state"},
        },
        ExecutionResult(True, "checked", {}),
    )

    assert result.state is TriState.TRUE


def test_embodied_completion_predicates_use_verified_execution_evidence():
    checker = GoalChecker()

    contact = checker.check(
        {
            "type": "structured",
            "predicate": "contact_detected",
            "args": {"target": "generic_target"},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(
            True,
            "contact",
            {
                "contact_detected": {
                    "target": "generic_target",
                    "matched": True,
                    "semantic_verified": True,
                    "contact_verified": True,
                    "contact_duration_sec": 0.2,
                }
            },
        ),
    )
    changed = checker.check(
        {
            "type": "structured",
            "predicate": "target_state_changed",
            "args": {"target": "generic_target"},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(
            True,
            "state changed",
            {
                "target_state_changed": {
                    "target": "generic_target",
                    "matched": True,
                    "semantic_verified": True,
                    "target_state_verified": True,
                    "changed": True,
                }
            },
        ),
    )

    assert contact.state is TriState.TRUE
    assert changed.state is TriState.TRUE


def test_embodied_completion_predicates_reject_weak_execution_evidence():
    checker = GoalChecker()

    contact = checker.check(
        {
            "type": "structured",
            "predicate": "contact_detected",
            "args": {"target": "generic_target"},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(True, "contact", {"contact_detected": {"target": "generic_target", "matched": True}}),
    )
    changed = checker.check(
        {
            "type": "structured",
            "predicate": "target_state_changed",
            "args": {"target": "generic_target"},
            "verification": {"mode": "action_result"},
        },
        ExecutionResult(True, "state", {"target_state_changed": {"target": "generic_target", "matched": True}}),
    )

    assert contact.state is TriState.UNKNOWN
    assert changed.state is TriState.UNKNOWN


def test_embodied_completion_requires_semantic_verification_when_evidence_says_missing():
    result = GoalChecker().check(
        {
            "type": "structured",
            "predicate": "contact_detected",
            "args": {"target": "generic_target"},
            "verification": {"mode": "action_result_then_world_state"},
        },
        ExecutionResult(
            True,
            "contact",
            {
                "contact_detected": {
                    "target": "generic_target",
                    "matched": True,
                    "semantic_verified": False,
                }
            },
        ),
    )

    assert result.state is TriState.UNKNOWN


def test_person_following_requires_semantic_verification_when_evidence_says_missing():
    result = GoalChecker().check(
        {
            "type": "structured",
            "predicate": "person_following",
            "args": {"person_id": "person:subject"},
            "verification": {"mode": "action_result_then_world_state"},
        },
        ExecutionResult(
            True,
            "following",
            {
                "person_following": {
                    "person_id": "person:subject",
                    "following": True,
                    "semantic_verified": False,
                }
            },
        ),
    )

    assert result.state is TriState.UNKNOWN


def test_world_snapshot_can_confirm_person_visible_by_id():
    world_json = json.dumps(
        {
            "facts": {
                "people": {
                    "visible_people": {
                        "value": {
                            "count": 2,
                            "people": [{"id": "person:0"}, {"id": "person:1"}],
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "person_visible",
        "args": {"id": "person:1"},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_world_snapshot_marks_person_visible_false_when_count_too_low():
    world_json = json.dumps(
        {
            "facts": {
                "people": {
                    "visible_people": {
                        "value": {
                            "count": 0,
                            "people": [],
                        }
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "person_visible",
        "args": {"min_count": 1},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.FALSE


def test_execution_failure_makes_goal_false():
    result = GoalChecker().check_json(
        json.dumps({"type": "human", "verification": {"mode": "implicit"}}),
        ExecutionResult(False, "skill failed"),
    )

    assert result.state is TriState.FALSE


def test_human_confirmation_goal_requires_approved_evidence():
    goal_spec = {
        "type": "human",
        "verification": {"mode": "confirmation", "request_id": "confirm-1"},
    }

    missing = GoalChecker().check(goal_spec, ExecutionResult(True, "done", {}))
    approved = GoalChecker().check(
        goal_spec,
        ExecutionResult(
            True,
            "done",
            {"human_confirmation": {"request_id": "confirm-1", "approved": True}},
        ),
    )
    denied = GoalChecker().check(
        goal_spec,
        ExecutionResult(
            True,
            "done",
            {"human_confirmation": {"request_id": "confirm-1", "approved": False}},
        ),
    )

    assert missing.state is TriState.UNKNOWN
    assert approved.state is TriState.TRUE
    assert denied.state is TriState.FALSE


def test_human_confirmation_goal_can_use_world_state_snapshot():
    world_json = json.dumps(
        {
            "facts": {
                "tasks": {
                    "last_human_confirmation": {
                        "value": {"request_id": "confirm-2", "decision": "APPROVED"}
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "human",
        "verification": {"mode": "operator_confirmation", "request_id": "confirm-2"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "done", {}))

    assert result.state is TriState.TRUE
