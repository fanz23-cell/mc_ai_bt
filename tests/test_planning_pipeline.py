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


def _mission(intent: str = "go to test_place", context_json: str = '{"language":"en-US"}'):
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text=intent,
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json=context_json,
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
    mission, missions = _mission(
        "go check on 33",
        context_json=json.dumps({
            "grounded_entities": [{"alias": "33", "entity_id": "person_bad0fefe"}],
        }),
    )
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


def _implicit_plan_with_actions(actions: list) -> str:
    return json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Sequence", "children": actions},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })


# --- remember_person/remember_entity preceded by a redundant locate action
# (2026-09-03, E.1 follow-up) -- remember_person/remember_entity already
# locate the target internally, so a real gpt-4o-mini planner reliably (and
# a prompt instruction against it measurably did not stop it) prepends a
# look_at/search_for_entity/locate_entity Action first. The single-action
# auto-fill above never covers this (2 physical actions), so it used to
# fall straight through to PolicyGuard's rejection every time.

def test_planning_pipeline_absorbs_a_redundant_look_at_into_relation():
    # 2026-09-03 (GPT review, D0, then a Phase C follow-up found live one
    # round later): look_at used to be dropped unconditionally (a real
    # information-loss bug -- D0 reverted that), then left completely
    # untouched (safe, but blocked a real live mission that no longer even
    # needed to be blocked once ResolveEntityReference/relation existed).
    # The actual fix: "front" is copied into remember_entity's own
    # `relation` arg (which ResolveEntityReference can now genuinely
    # consume to disambiguate 2+ candidates), and ONLY THEN is look_at
    # removed -- no information lost, unlike the original bug, and no
    # longer needlessly blocked either.
    mission, missions = _mission("remember this as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "front"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "33"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert bt == {
        "type": "Sequence",
        "children": [
            {
                "type": "Action", "skill": "remember_entity",
                "args": {"target": "the person", "alias": "33", "relation": "front"},
            },
        ],
    }
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["args"]["relation"] == "front"


def test_planning_pipeline_does_not_absorb_a_look_at_with_an_unmappable_direction():
    # No relation corresponds to a bare "up"/"down" bearing (ResolveEntityReference
    # only understands ground-plane front/left/right/nearest) -- never guess,
    # same conservative fallback as before the absorption feature existed.
    mission, missions = _mission("remember this as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "up"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "33"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_override_an_already_specific_terminal_action():
    # remember_entity already specifies its own relation -- a preceding
    # look_at's bearing must never override or duplicate that; left
    # completely untouched (still 2 physical actions, still policy-rejected).
    mission, missions = _mission("remember this as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
        {"type": "Action", "skill": "remember_entity",
         "args": {"target": "the person", "alias": "33", "relation": "nearest"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_fills_in_goal_predicate_past_a_redundant_search_for_entity():
    mission, missions = _mission("remember this person as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "search_for_entity", "args": {"target": "the person"}},
        {"type": "Action", "skill": "remember_person", "args": {"target": "the person", "name": "33"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["predicate"] == "person_named"
    bt = json.loads(result.bt_json)
    assert bt["children"] == [
        {"type": "Action", "skill": "remember_person", "args": {"target": "the person", "name": "33"}},
    ]


def test_planning_pipeline_does_not_drop_a_search_for_entity_targeting_something_else():
    # 2026-09-03 (GPT review): search_for_entity/locate_entity DO carry a
    # real target entity reference (unlike look_at, which is direction-only)
    # -- a search for something UNRELATED to the remember action's own
    # target must never be silently dropped, since it may be there for a
    # genuinely different reason.
    mission, missions = _mission("look for the chair, then remember this person as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "search_for_entity", "args": {"target": "chair"}},
        {"type": "Action", "skill": "remember_person", "args": {"target": "the person", "name": "33"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    # Not canonicalized (different targets) -- still 2 real physical
    # actions, correctly rejected exactly as before this round's fixes.
    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_guess_when_remember_entity_follows_a_non_locate_action():
    mission, missions = _mission("go there and remember this as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "33"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    # go_to_place is a real, non-redundant physical action -- must not guess.
    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_guess_with_two_remember_actions():
    mission, missions = _mission("remember both of them")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "remember_entity", "args": {"target": "person A", "alias": "A"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "person B", "alias": "B"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    # Two remember-type actions -- genuinely ambiguous which one the goal means.
    assert not result.ok
    assert result.stage == "policy"


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


# --- GroundingNormalizer (2026-09-03, E.1 follow-up) -----------------------
# planner.py's system prompt already INSTRUCTS the model to copy an exact
# entity_id from context_json.caller_context.grounded_entities and never
# invent one -- these tests cover the deterministic enforcement of that same
# rule, independent of whether the LLM actually complied.

def _grounded_context(*entries: dict) -> str:
    return json.dumps({"grounded_entities": list(entries)})


def test_grounding_normalizer_accepts_an_entity_id_that_matches_grounded_entities():
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context({"alias": "33", "entity_id": "person_real0001"}),
    )
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_real0001"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message


def test_grounding_normalizer_rejects_a_hallucinated_entity_id():
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context({"alias": "33", "entity_id": "person_real0001"}),
    )
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_made_up"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "grounding"
    assert "person_made_up" in result.message


def test_grounding_normalizer_rejects_an_entity_id_when_nothing_is_grounded():
    mission, missions = _mission("go check on 33")  # default context has no grounded_entities
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_made_up"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "grounding"


def test_grounding_normalizer_injects_entity_id_from_an_exact_alias_match():
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context({"alias": "33", "entity_id": "person_real0001"}),
    )
    # No entity_id -- as if the planner correctly picked the right target but
    # forgot to also copy the id, despite the instruction.
    plan_json = _implicit_plan_with_single_action("approach_entity", {"target": "33"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["args"]["entity_id"] == "person_real0001"


def test_grounding_normalizer_leaves_a_target_alone_when_no_alias_matches():
    mission, missions = _mission(
        "go check on the stranger",
        context_json=_grounded_context({"alias": "33", "entity_id": "person_real0001"}),
    )
    plan_json = _implicit_plan_with_single_action("approach_entity", {"target": "the stranger"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert "entity_id" not in goal_spec["args"]


def test_grounding_normalizer_rejects_an_entity_id_cross_wired_to_a_different_alias():
    # 2026-09-03 (GPT review): found live -- with TWO aliases grounded in
    # the same mission, the old check only asked "is this entity_id
    # SOMEWHERE in the grounded set", which a cross-wired id (the real
    # entity_id of a DIFFERENT alias than the one the target names) would
    # pass. target="33" names person_A specifically; entity_id=person_B
    # (22's real id) must be rejected, not treated as merely "some real id".
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context(
            {"alias": "33", "entity_id": "person_A"},
            {"alias": "22", "entity_id": "person_B"},
        ),
    )
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_B"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "grounding"
    assert "person_B" in result.message
    assert "person_A" in result.message


def test_grounding_normalizer_accepts_the_correct_id_when_multiple_aliases_are_grounded():
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context(
            {"alias": "33", "entity_id": "person_A"},
            {"alias": "22", "entity_id": "person_B"},
        ),
    )
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_A"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
