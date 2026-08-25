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
