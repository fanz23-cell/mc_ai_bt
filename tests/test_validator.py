import json

from mc_ai_bt.validator import PlanValidator


def test_accepts_minimal_say_plan_with_human_goal():
    plan = {
        "root": {
            "type": "Action",
            "skill": "say",
            "args": {"text": "hello"},
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok


def test_rejects_unknown_skill():
    plan = {
        "root": {
            "type": "Action",
            "skill": "publish_arbitrary_topic",
            "args": {},
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert "unknown" in result.errors[0]


def test_rejects_goal_without_verification():
    plan = {
        "root": {
            "type": "Action",
            "skill": "say",
            "args": {"text": "hello"},
        },
        "goal_spec": {
            "type": "human",
        },
    }

    result = PlanValidator().validate_json(json.dumps(plan))

    assert not result.ok
    assert any("verification" in error for error in result.errors)


def test_accepts_executable_condition_and_goal_check_nodes():
    plan = {
        "root": {
            "type": "Sequence",
            "children": [
                {
                    "type": "Condition",
                    "predicate": "robot_at_place",
                    "args": {"name": "test_place"},
                },
                {
                    "type": "GoalCheck",
                    "check": {
                        "predicate": "robot_at_place",
                        "args": {"name": "test_place"},
                    },
                },
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok


def test_rejects_condition_with_invented_predicate():
    # Found live 2026-08-30: a real submitted mission's plan used
    # predicate "visible_people" -- not a real predicate name (it's an internal
    # world-state fact-storage key that person_visible's own evaluation reads,
    # not something Condition/GoalCheck/WaitForEvent can check). The validator
    # previously only checked that `predicate` was a non-empty string, so this
    # passed schema validation, ran, and could only ever resolve to an eternal,
    # unanswerable UNKNOWN -- wasting the mission regardless of whether a woman
    # was actually present. This is the systemic guard: any invented predicate
    # name is now rejected at plan-validation time, not just this one string.
    plan = {
        "root": {
            "type": "Condition",
            "predicate": "visible_people",
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("not a known predicate" in error and "visible_people" in error
               for error in result.errors)


def test_rejects_goal_check_with_invented_predicate():
    plan = {
        "root": {
            "type": "GoalCheck",
            "check": {"predicate": "visible_people"},
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("not a known predicate" in error for error in result.errors)


def test_rejects_wait_for_event_with_invented_predicate():
    plan = {
        "root": {"type": "WaitForEvent", "predicate": "visible_people", "timeout_sec": 60.0},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("not a known predicate" in error for error in result.errors)


def test_rejects_structured_goal_spec_with_invented_predicate():
    plan = {
        "root": {"type": "Action", "skill": "say", "args": {"text": "hi"}},
        "goal_spec": {
            "type": "structured",
            "predicate": "visible_people",
            "verification": {"mode": "world_state"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("not a known predicate" in error for error in result.errors)


def test_accepts_wait_for_event_node():
    plan = {
        "root": {
            "type": "WaitForEvent",
            "predicate": "participant_ready",
            "args": {"role": "customer"},
            "poll_interval_sec": 1.0,
            "timeout_sec": 8.0,
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok


def test_accepts_wait_for_event_default_poll_interval():
    plan = {
        "root": {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 60.0},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    assert PlanValidator().validate(plan).ok


def test_rejects_wait_for_event_missing_predicate():
    plan = {
        "root": {"type": "WaitForEvent", "timeout_sec": 60.0},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("predicate" in error for error in result.errors)


def test_rejects_wait_for_event_timeout_over_1800():
    # This is exactly the bug found live 2026-08-29: a WaitForEvent plan with
    # timeout_sec > 600 (but <= 1800, its own documented ceiling) must not be
    # rejected by the generic node-level timeout_sec check, which caps at 600
    # for every other node type.
    plan = {
        "root": {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 1801.0},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("timeout_sec" in error for error in result.errors)


def test_accepts_wait_for_event_timeout_above_600_below_1800():
    plan = {
        "root": {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 1200.0},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    assert PlanValidator().validate(plan).ok


def test_rejects_wait_for_event_poll_interval_out_of_bounds():
    plan = {
        "root": {
            "type": "WaitForEvent",
            "predicate": "participant_ready",
            "poll_interval_sec": 0.1,
            "timeout_sec": 60.0,
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("poll_interval_sec" in error for error in result.errors)


def test_accepts_visual_check_nodes():
    plan = {
        "root": {
            "type": "VisualCheck",
            "check": {"query": "is the test_object visible?"},
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok


def test_accepts_parallel_timeout_and_noaction_nodes():
    plan = {
        "root": {
            "type": "Sequence",
            "children": [
                {
                    "type": "Parallel",
                    "children": [
                        {"type": "Action", "skill": "say", "args": {"text": "checking"}},
                        {
                            "type": "Timeout",
                            "timeout_sec": 2.0,
                            "child": {"type": "Wait", "duration_sec": 0.1},
                        },
                    ],
                },
                {"type": "NoAction", "reason": "world event handled locally"},
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok


def test_rejects_timeout_without_valid_child_or_timeout():
    plan = {
        "root": {
            "type": "Timeout",
            "timeout_sec": 0,
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("timeout_sec" in error for error in result.errors)
    assert any("child" in error for error in result.errors)


def test_rejects_parallel_branches_with_conflicting_skill_resources():
    plan = {
        "root": {
            "type": "Parallel",
            "children": [
                {"type": "Action", "skill": "look_at", "args": {"direction": "front"}},
                {"type": "Action", "skill": "point_at", "args": {"object": "test_object"}},
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("conflicting resources" in error and "body" in error for error in result.errors)


def test_rejects_parallel_branches_with_cross_family_resource_conflict():
    # Found live 2026-08-29 (§C5): "body" and "gaze" are different resource
    # strings, but mc_resource_authority's own runtime lease arbiter treats
    # them as conflicting (body's children include gaze) -- a plan pairing
    # play_animation (body) with look_at_static (gaze) used to clear this
    # validator with zero errors and only get denied later, at the resource
    # lease, or worse race silently for skills that skip leasing entirely.
    plan = {
        "root": {
            "type": "Parallel",
            "children": [
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
                {"type": "Action", "skill": "look_at_static", "args": {"target": "left"}},
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("conflicting resources" in error for error in result.errors)


def test_rejects_parallel_branches_pairing_search_for_entity_with_a_base_skill():
    # FOUND LIVE 2026-09-01 (3rd GPT review): locate_entity/get_pose/search_for_entity all
    # physically turn the robot base (mc_embodied_skills' _locate_with_scan) when the
    # target isn't in the current frame, but skill_registry.py used to declare no "base"
    # resource for any of them -- this exact plan shape used to clear the validator with
    # zero errors, while at runtime both branches genuinely fought over the base.
    plan = {
        "root": {
            "type": "Parallel",
            "children": [
                {"type": "Action", "skill": "search_for_entity", "args": {"target": "alice"}},
                {"type": "Action", "skill": "go_to_place", "args": {"place": "kitchen"}},
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert not result.ok
    assert any("conflicting resources" in error and "base" in error for error in result.errors)


def test_accepts_parallel_branches_with_genuinely_unrelated_resources():
    plan = {
        "root": {
            "type": "Parallel",
            "children": [
                {"type": "Action", "skill": "say", "args": {"text": "hi"}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "forward", "value": 0.2}},
            ],
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    assert PlanValidator().validate(plan).ok
