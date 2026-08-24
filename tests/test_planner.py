import json

from mc_ai_bt.planner import BootstrapPlanner
from mc_ai_bt.validator import PlanValidator


def _plan(text: str) -> dict:
    return json.loads(BootstrapPlanner().plan(text))


def test_go_to_place_intent_becomes_valid_bt():
    plan = _plan("go to kitchen")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["type"] == "Sequence"
    assert plan["root"]["children"][1]["skill"] == "go_to_place"
    assert plan["root"]["children"][1]["args"]["name"] == "kitchen"
    assert plan["goal_spec"]["predicate"] == "robot_at_place"


def test_come_to_me_intent_becomes_valid_bt():
    plan = _plan("Codey, come to me")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["children"][1]["skill"] == "come_to_me"
    assert plan["goal_spec"]["predicate"] == "robot_near_interaction_owner"


def test_simple_move_defaults_value_by_direction():
    forward = _plan("move forward")
    left = _plan("turn and move left")

    assert forward["root"]["children"][1]["args"] == {"action": "forward", "value": 0.5}
    assert left["root"]["children"][1]["args"] == {"action": "left", "value": 90.0}


def test_wave_intent_becomes_animation():
    plan = _plan("wave hello")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["skill"] == "play_animation"
    assert plan["root"]["args"]["animation"] == "wave"


def test_look_at_intent_becomes_look_at_skill():
    plan = _plan("look left")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["skill"] == "look_at"
    assert plan["root"]["args"] == {"direction": "left"}
    assert plan["goal_spec"]["predicate"] == "animation_played"
    assert plan["goal_spec"]["args"] == {"animation": "look_at"}


def test_point_at_intent_becomes_point_at_skill():
    plan = _plan("point at the cup with left hand")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["skill"] == "point_at"
    assert plan["root"]["args"] == {"arm": "left", "object": "cup"}
    assert plan["goal_spec"]["predicate"] == "animation_played"
    assert plan["goal_spec"]["args"] == {"animation": "point_at"}


def test_find_object_intent_becomes_visual_check():
    plan = _plan("find the cup")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["type"] == "Sequence"
    assert plan["root"]["children"][1]["type"] == "VisualCheck"
    assert plan["root"]["children"][1]["check"]["query"] == "do you see the cup?"
    assert plan["goal_spec"]["predicate"] == "object_visible"
    assert plan["goal_spec"]["args"] == {"name": "cup"}


def test_look_for_someone_intent_becomes_person_visual_check():
    plan = _plan("look for someone")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["children"][1]["type"] == "VisualCheck"
    assert plan["goal_spec"]["predicate"] == "person_visible"
    assert plan["goal_spec"]["args"] == {"min_count": 1}


def test_compound_intent_becomes_multi_step_sequence_with_final_goal():
    plan = _plan("go to kitchen then wave")

    assert PlanValidator().validate(plan).ok
    assert [child["type"] for child in plan["root"]["children"]] == [
        "Action",
        "Action",
        "Action",
    ]
    assert plan["root"]["children"][1]["skill"] == "go_to_place"
    assert plan["root"]["children"][2]["skill"] == "play_animation"
    assert plan["goal_spec"]["predicate"] == "animation_played"


def test_compound_visual_intent_uses_visual_goal_as_final_goal():
    plan = _plan("go to kitchen then find the cup")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["children"][1]["skill"] == "go_to_place"
    assert plan["root"]["children"][3]["type"] == "VisualCheck"
    assert plan["goal_spec"]["predicate"] == "object_visible"
    assert plan["goal_spec"]["args"] == {"name": "cup"}


def test_chinese_go_to_place_intent_becomes_valid_bt():
    plan = _plan("去厨房")

    assert PlanValidator().validate(plan).ok
    assert plan["root"]["children"][1]["skill"] == "go_to_place"
    assert plan["root"]["children"][1]["args"]["name"] == "kitchen"
    assert plan["goal_spec"]["predicate"] == "robot_at_place"


def test_chinese_embodied_intents_use_existing_skills():
    cases = {
        "来我这里": ("come_to_me", {}),
        "左转90度": ("simple_move", {"action": "left", "value": 90.0}),
        "挥手": ("play_animation", {"animation": "wave"}),
        "看左边": ("look_at", {"direction": "left"}),
        "指一下杯子": ("point_at", {"arm": "right", "object": "cup"}),
    }

    for intent, (skill, args) in cases.items():
        plan = _plan(intent)

        assert PlanValidator().validate(plan).ok
        node = plan["root"]
        if node["type"] == "Sequence":
            node = node["children"][1]
        assert node["skill"] == skill
        assert node["args"] == args


def test_chinese_visual_and_compound_intents_use_existing_goal_shapes():
    visual = _plan("找杯子")

    assert PlanValidator().validate(visual).ok
    assert visual["root"]["children"][1]["type"] == "VisualCheck"
    assert visual["root"]["children"][1]["check"]["query"] == "do you see the cup?"
    assert visual["goal_spec"]["predicate"] == "object_visible"
    assert visual["goal_spec"]["args"] == {"name": "cup"}

    compound = _plan("去厨房然后挥手")

    assert PlanValidator().validate(compound).ok
    assert compound["root"]["children"][1]["skill"] == "go_to_place"
    assert compound["root"]["children"][2]["skill"] == "play_animation"
    assert compound["goal_spec"]["predicate"] == "animation_played"
