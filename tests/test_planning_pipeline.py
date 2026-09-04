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

def _implicit_plan_with_single_action(
    skill: str, args: dict, *, reference_constraints: list | None = None,
) -> str:
    plan = {
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": skill, "args": args},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    }
    if reference_constraints is not None:
        plan["reference_constraints"] = reference_constraints
    return json.dumps(plan)


def test_planning_pipeline_fills_in_the_only_possible_goal_predicate():
    mission, missions = _mission(
        "go check on 33",
        context_json=json.dumps({
            "grounded_entities": [{"alias": "33", "entity_id": "person_bad0fefe", "grounding_state": "RESOLVED"}],
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

def test_planning_pipeline_drops_a_redundant_look_at_without_absorbing_its_direction():
    # Gate-1.1 architecture round (2026-09-03, GPT re-review): a look_at
    # immediately before remember_entity is still dropped as redundant
    # motion (remember_entity locates internally regardless) -- but its
    # direction is no longer harvested into `relation` at all. Confirmed
    # here with intent_text that says nothing about position: the plan
    # still succeeds (look_at genuinely is pure waste here), and the final
    # args carry NO relation key, because nothing established one.
    mission, missions = _mission("Remember this person as 33")
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
                "args": {"target": "the person", "alias": "33"},
            },
        ],
    }
    goal_spec = json.loads(result.goal_spec_json)
    assert "relation" not in goal_spec["args"]


def test_planning_pipeline_still_drops_a_redundant_look_at_when_a_constraint_supplies_relation():
    # The two mechanisms are properly decoupled: look_at is dropped purely
    # because it is redundant motion; relation comes ONLY from the
    # validated reference_constraint, never from look_at's own direction
    # (which here points a DIFFERENT way -- "left" -- than the constraint's
    # "front", proving the constraint, not the look_at, is the real source).
    mission, missions = _mission("There is a person right in front of you. Remember this as 33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
                {
                    "type": "Action", "skill": "remember_entity",
                    "args": {
                        "target": "the person", "alias": "33",
                        "reference_constraint_id": "ref_1",
                    },
                },
            ],
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
        "reference_constraints": [
            {
                "constraint_id": "ref_1", "entity_class": "person", "relation": "front",
                "reference_frame": "robot", "source_span": "right in front of you",
            },
        ],
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert len(bt["children"]) == 1  # look_at dropped -- only remember_entity remains
    assert bt["children"][0]["args"]["relation"] == "front"


def test_planning_pipeline_does_not_drop_a_look_at_before_a_terminal_with_an_already_specific_entity_id():
    # remember_entity already specifies its own entity_id (a MORE specific
    # disambiguation than anything look_at could add) -- left completely
    # untouched (still 2 physical actions, still policy-rejected).
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context({"alias": "33", "entity_id": "person_A"}),
    )
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
        {"type": "Action", "skill": "remember_entity",
         "args": {"target": "33", "alias": "33", "entity_id": "person_A"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_absorb_a_look_at_that_comes_after_the_terminal_action():
    # P1-1 (2026-09-03, GPT review, Gate-1): _terminal_self_locating_action
    # used to operate on a flat, order-blind list of physical actions --
    # look_at AFTER remember_entity in the same Sequence (not a real
    # "redundant prefix" at all) could previously still be absorbed. Now
    # requires a strictly SMALLER index than the terminal action in the
    # same Sequence; this plan has none, so it is left completely untouched.
    mission, missions = _mission("remember this as 33")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "33"}},
        {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"


def test_planning_pipeline_does_not_absorb_a_look_at_from_a_different_fallback_branch():
    # P1-1 (2026-09-03, GPT review, Gate-1): a look_at and a remember_entity
    # that are alternatives in a Fallback (mutually exclusive branches, not
    # a sequence at all) must never be treated as a "prefix" relationship --
    # neither is a direct child of any Sequence, so the terminal action
    # itself fails the same-Sequence-child requirement and the whole
    # pattern is left completely untouched.
    mission, missions = _mission("remember this as 33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {
            "type": "Fallback",
            "children": [
                {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
                {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "33"}},
            ],
        },
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

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
    # Auto-fill grounding_state=RESOLVED whenever an entry has a real
    # entity_id and doesn't say otherwise -- every pre-existing call site
    # below only ever set alias/entity_id, matching a real Bridge-resolved
    # alias; a test that cares about the UNRESOLVED case sets it explicitly.
    filled = []
    for entry in entries:
        entry = dict(entry)
        entry.setdefault("grounding_state", "RESOLVED" if entry.get("entity_id") else "UNRESOLVED")
        filled.append(entry)
    return json.dumps({"grounded_entities": filled})


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


def test_grounding_normalizer_rejects_an_entity_id_cross_wired_from_an_unresolved_alias():
    # 2026-09-03 (GPT review, Gate-1): found live -- once UNRESOLVED aliases
    # stopped being dropped from grounded_entities entirely, the ORIGINAL
    # cross-wire check above stopped applying to them (an UNRESOLVED alias
    # was simply absent from the map, so its target fell through to the
    # weaker flat-membership fallback, which happily accepted a DIFFERENT,
    # RESOLVED alias's real entity_id). "33" is a known alias, currently
    # UNRESOLVED (no live entity -- see mc_world_state/entity_identity.py);
    # "22" is RESOLVED to person_B. target="33" + entity_id=person_B (22's
    # real id) must be rejected just as hard as the RESOLVED-vs-RESOLVED
    # cross-wire above -- an unresolved alias can never legitimately carry
    # ANY entity_id, cross-wired or otherwise.
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context(
            {"alias": "33", "entity_id": "", "grounding_state": "UNRESOLVED"},
            {"alias": "22", "entity_id": "person_B"},
        ),
    )
    plan_json = _implicit_plan_with_single_action(
        "approach_entity", {"target": "33", "entity_id": "person_B"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "grounding"
    assert "unresolved" in result.message.lower()


def test_grounding_normalizer_passes_through_an_unresolved_alias_with_no_entity_id():
    # The legitimate UNRESOLVED case: the planner correctly did not invent
    # an entity_id for a known-but-currently-unresolved alias -- nothing to
    # inject (there is no live entity to inject), and nothing to reject.
    mission, missions = _mission(
        "go check on 33",
        context_json=_grounded_context({"alias": "33", "entity_id": "", "grounding_state": "UNRESOLVED"}),
    )
    plan_json = _implicit_plan_with_single_action("approach_entity", {"target": "33"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    goal_spec = json.loads(result.goal_spec_json)
    assert "entity_id" not in goal_spec["args"]


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


# --- reference constraint guard (2026-09-03, GPT re-review, Gate-1.1
# architecture round) --------------------------------------------------------
# Gate-1's P0-2 fix made `relation` authoritative against real geometry once
# supplied. A first Gate-1.1 round tried a keyword-lexicon backstop (does
# SOME phrase supporting this relation appear ANYWHERE in intent_text) --
# GPT's re-review found this could still FALSE-ACCEPT a cross-wired claim
# (not just miss a legitimate one): "Look to your left, then remember this
# person" (left describes a look_at MOVEMENT, not the person), "The chair is
# on your left, remember this person" (left describes the CHAIR), "remember
# the person in front of the sofa" (front is relative to the SOFA, not the
# robot) would all have passed the old keyword check. These tests are the
# regression backstop for the structured reference_constraints contract that
# replaced it -- each one encodes exactly one of those counter-examples.

def test_reference_constraint_guard_rejects_a_relation_set_directly():
    # The literal fix for "the planner can invent relation=left": there is
    # no longer anywhere for a directly-set relation to go that the guard
    # will accept, regardless of intent_text content.
    mission, missions = _mission("The person on your left, remember them as 44")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity", {"target": "the person", "alias": "44", "relation": "left"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "must never be set directly" in result.message


def test_reference_constraint_guard_rejects_an_unknown_constraint_id():
    mission, missions = _mission("Remember this person as 44")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_missing"},
        reference_constraints=[],
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "does not match any entry" in result.message


def test_reference_constraint_guard_rejects_a_hallucinated_source_span():
    # GPT's original worked example, now via the structured path: the
    # user's own words never mention a side at all -- a constraint CLAIMING
    # "on your left" as its source_span is citing text that was never there.
    mission, missions = _mission("Remember this person as 44")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
        reference_constraints=[{
            "constraint_id": "ref_1", "entity_class": "person", "relation": "left",
            "reference_frame": "robot", "source_span": "on your left",
        }],
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "was not found" in result.message


def test_reference_constraint_guard_rejects_a_constraint_relative_to_a_different_entity():
    # GPT's sofa counter-example: source_span is genuinely, verbatim
    # present in intent_text -- but it describes a relation to the SOFA,
    # not to the robot. ResolveEntityReference only ever understands
    # robot-relative bearings; reinterpreting this as reference_frame=
    # "robot" would silently answer a different question than the one the
    # user actually asked.
    mission, missions = _mission("Remember the person in front of the sofa as 44")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
        reference_constraints=[{
            "constraint_id": "ref_1", "entity_class": "person", "relation": "front",
            "reference_frame": "sofa", "source_span": "in front of the sofa",
        }],
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "robot" in result.message


def test_reference_constraint_guard_rejects_a_constraint_describing_a_different_entity_class():
    # GPT's chair counter-example: source_span is genuinely present -- but
    # it describes the CHAIR's position, not the person being bound.
    # remember_person requires entity_class="person"; a constraint
    # extracted for "chair" must never be borrowed for it.
    mission, missions = _mission("The chair is on your left. Remember this person as 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_person",
        {"target": "the person", "name": "44", "reference_constraint_id": "ref_1"},
        reference_constraints=[{
            "constraint_id": "ref_1", "entity_class": "chair", "relation": "left",
            "reference_frame": "robot", "source_span": "on your left",
        }],
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "does not match" in result.message


def test_reference_constraint_guard_accepts_a_valid_constraint_and_fills_in_relation():
    mission, missions = _mission("The person on your left, remember them as 44")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
        reference_constraints=[{
            "constraint_id": "ref_1", "entity_class": "person", "relation": "left",
            "reference_frame": "robot", "source_span": "on your left",
        }],
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    assert json.loads(result.goal_spec_json)["args"]["relation"] == "left"


def test_reference_constraint_guard_ignores_check_relations_own_unrelated_relation_arg():
    # check_relation's own `relation` arg is a free-text predicate string
    # ("the mug near the sink"), not a front/left/right/nearest spatial
    # disambiguator -- the guard only inspects remember_person/
    # remember_entity (_SELF_LOCATING_TERMINAL_SKILLS) and must never
    # false-trigger on this unrelated, same-named arg.
    mission, missions = _mission("is the mug near the sink")
    plan_json = _implicit_plan_with_single_action(
        "check_relation", {"relation": "the mug near the sink"})

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
