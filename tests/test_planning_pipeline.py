import json

from mc_ai_bt.context_builder import ContextBuilder
from mc_ai_bt.mission import MissionManager
from mc_ai_bt.planner import BootstrapPlanner
from mc_ai_bt.planning_pipeline import PlanningPipeline
from mc_ai_bt.policy_guard import PolicyGuard
from mc_ai_bt.validator import PlanValidator


class StaticPlanner:
    def __init__(self, plan_json: str):
        self.plan_json = plan_json

    def plan(self, intent_text: str, context_json: str = "") -> str:
        return self.plan_json


class RaisingPlanner:
    def plan(self, intent_text: str, context_json: str = "") -> str:
        raise RuntimeError("no model")


def _mission(intent: str = "go to test_place"):
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text=intent,
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json='{"language":"en-US"}',
    )
    return mission, manager.all()


def _pipeline(planner):
    return PlanningPipeline(
        planner=planner,
        context_builder=ContextBuilder(),
        validator=PlanValidator(),
        policy_guard=PolicyGuard(),
    )


def test_planning_pipeline_returns_normalized_artifacts():
    mission, missions = _mission()

    result = _pipeline(BootstrapPlanner()).plan(mission, missions)

    assert result.ok
    assert result.stage == "done"
    assert json.loads(result.bt_json)["type"] == "Sequence"
    assert json.loads(result.goal_spec_json)["predicate"] == "robot_at_place"
    assert json.loads(result.context_json)["caller_context"] == {"language": "en-US"}


def test_planning_pipeline_accepts_bootstrap_visual_check_plan():
    mission, missions = _mission("find the test_object")

    result = _pipeline(BootstrapPlanner()).plan(mission, missions)

    assert result.ok
    assert json.loads(result.bt_json)["children"][1]["type"] == "VisualCheck"
    assert json.loads(result.goal_spec_json)["predicate"] == "object_visible"


def test_planning_pipeline_reports_planner_exception():
    mission, missions = _mission()

    result = _pipeline(RaisingPlanner()).plan(mission, missions)

    assert not result.ok
    assert result.stage == "planner"
    assert "RuntimeError" in result.message


def test_planning_pipeline_reports_validator_error():
    mission, missions = _mission()

    result = _pipeline(StaticPlanner('{"root": {"type": "Nope"}}')).plan(mission, missions)

    assert not result.ok
    assert result.stage == "validator"
    assert "goal_spec" in result.message


# --- deterministic goal_spec auto-fill (2026-09-02, E.1 follow-up) --------
# The gap this closes: a planner emitting implicit/human verification for a
# mission containing exactly one physical skill used to be an unconditional
# PolicyGuard rejection -- correct when the goal is genuinely ambiguous, but
# wrong when there is only ONE possible predicate the skill could mean (its
# own result_predicates has exactly one entry). This is the specific gap
# that blocked E.1's own live acceptance test ("go check on 33") from ever
# reaching a real plan.

def _implicit_plan_with_single_action(skill: str, args: dict) -> str:
    return json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": skill, "args": args},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })


def test_planning_pipeline_fills_in_the_only_possible_goal_predicate():
    mission, missions = _mission("go check on 33")
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_bad0fefe"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["type"] == "structured"
    assert goal_spec["predicate"] == "entity_approached"
    assert goal_spec["args"] == {"target": "33", "entity_id": "person_bad0fefe"}


def test_planning_pipeline_fills_in_remember_entity_goal_predicate():
    mission, missions = _mission("remember this as 33")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity", {"target": "the person over there", "alias": "33"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["predicate"] == "entity_alias_bound"
    assert goal_spec["args"]["alias"] == "33"


def test_planning_pipeline_does_not_guess_with_two_physical_actions():
    mission, missions = _mission("go there and then wave")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
            ],
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    # Unchanged pre-existing behavior: still correctly rejected, not guessed.
    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_guess_when_the_skill_has_multiple_result_predicates():
    mission, missions = _mission("look left")
    # look_at declares TWO result_predicates (look_at_static_completed,
    # animation_played, see skill_registry.py) -- genuinely ambiguous which
    # one a mission meant, must not guess either one.
    plan_json = _implicit_plan_with_single_action("look_at", {"direction": "left"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_leaves_an_already_structured_goal_spec_alone():
    mission, missions = _mission()
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
        "goal_spec": {
            "type": "structured",
            "predicate": "robot_at_place",
            "args": {"name": "test_place"},
            "verification": {"mode": "world_state_or_nav_result"},
        },
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok
    assert json.loads(result.goal_spec_json)["verification"]["mode"] == "world_state_or_nav_result"


def test_planning_pipeline_reports_policy_error():
    mission, missions = _mission()
    plan_json = json.dumps(
        {
            "schema": "mc_ai_bt.plan.v1",
            "root": {
                "type": "Sequence",
                "children": [
                    {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
                    {
                        "type": "Action",
                        "skill": "simple_move",
                        "args": {"action": "forward", "value": 2.0},
                    },
                ],
            },
            "goal_spec": {
                "type": "structured",
                "predicate": "robot_at_place",
                "args": {"name": "test_place"},
                "verification": {"mode": "world_state_or_nav_result"},
            },
        }
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"
    assert "1m" in result.message
