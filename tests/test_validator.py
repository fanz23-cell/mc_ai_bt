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
                    "args": {"name": "kitchen"},
                },
                {
                    "type": "GoalCheck",
                    "check": {
                        "predicate": "robot_at_place",
                        "args": {"name": "kitchen"},
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
            "check": {"query": "is the cup visible?"},
        },
        "goal_spec": {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }

    result = PlanValidator().validate(plan)

    assert result.ok
