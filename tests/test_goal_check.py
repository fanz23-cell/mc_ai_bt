import json

from mc_ai_bt.executor import ExecutionResult
from mc_ai_bt.goal_check import GoalChecker, TriState


def test_structured_goal_uses_execution_facts():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "kitchen"},
        "verification": {"mode": "world_state_or_nav_result"},
    }
    execution = ExecutionResult(True, "arrived", {"robot_at_place": "kitchen"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


def test_structured_goal_mismatch_is_false():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "kitchen"},
        "verification": {"mode": "world_state_or_nav_result"},
    }
    execution = ExecutionResult(True, "arrived", {"robot_at_place": "office"})

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.FALSE


def test_missing_evidence_is_unknown_not_false():
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "kitchen"},
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
                        "value": "kitchen",
                    }
                }
            }
        }
    )
    checker = GoalChecker(lambda _scopes, _max_age: world_json)
    goal_spec = {
        "type": "structured",
        "predicate": "robot_at_place",
        "args": {"name": "kitchen"},
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
                    "object:cup": {
                        "value": {
                            "object_name": "cup",
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
        "args": {"name": "cup", "min_score": 0.5},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_visual_goal_query_can_be_confirmed_from_world_snapshot():
    world_json = json.dumps(
        {
            "facts": {
                "objects": {
                    "object:cup": {
                        "value": {
                            "object_name": "cup",
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
        "query": "do you see the cup?",
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.TRUE


def test_world_snapshot_marks_low_score_object_visible_false():
    world_json = json.dumps(
        {
            "facts": {
                "objects": {
                    "object:cup": {
                        "value": {
                            "object_name": "cup",
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
        "args": {"name": "cup", "min_score": 0.5},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.FALSE


def test_missing_object_visible_evidence_is_unknown():
    checker = GoalChecker(lambda _scopes, _max_age: json.dumps({"facts": {"objects": {}}}))
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "cup"},
        "verification": {"mode": "world_state_or_visual_check"},
    }

    result = checker.check(goal_spec, ExecutionResult(True, "checked", {}))

    assert result.state is TriState.UNKNOWN


def test_object_visible_can_be_confirmed_by_execution_fact():
    goal_spec = {
        "type": "structured",
        "predicate": "object_visible",
        "args": {"name": "cup", "min_score": 0.4},
        "verification": {"mode": "world_state_or_visual_check"},
    }
    execution = ExecutionResult(
        True,
        "checked",
        {"object_visible": {"object_name": "cup", "score": 0.9}},
    )

    result = GoalChecker().check(goal_spec, execution)

    assert result.state is TriState.TRUE


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
