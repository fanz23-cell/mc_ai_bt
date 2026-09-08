import json

from mc_ai_bt.context_builder import ContextBuilder
from mc_ai_bt.mission import MissionManager
from mc_ai_bt.planner import BootstrapPlanner
from mc_ai_bt.planning_pipeline import (
    PlanningPipeline, _all_actions_in, _apply_reference_constraint_guard,
    _append_missing_required_producers, _INTERNAL_GROUNDING_FIELD,
    _parse_mission_goal_contract, _strip_internal_grounding_field,
)
from mc_ai_bt.policy_guard import PolicyGuard
from mc_ai_bt.validator import PlanValidator


def _context_json_with_reference_constraints(*constraints: dict) -> str:
    """A hand-built context_json carrying an arbitrary (possibly invalid)
    reference_constraints list -- for unit-testing
    _apply_reference_constraint_guard's OWN defensive checks directly.
    The real extractor (reference_extraction.py) never actually produces
    an invalid entry -- these tests exist because the guard must not
    silently trust context_json's shape either, exactly the same
    "verify, don't just trust the source" principle applied one layer
    further out."""
    return json.dumps({"reference_constraints": list(constraints)})


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


def _pipeline(planner, *, reference_resolver=None):
    return PlanningPipeline(
        planner=planner,
        context_builder=ContextBuilder(),
        validator=PlanValidator(),
        policy_guard=PolicyGuard(),
        reference_resolver=reference_resolver,
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
    # because it is redundant motion (intent_text has no "look/turn left"
    # instruction, so has_explicit_look_instruction is False for it);
    # relation comes ONLY from the real, deterministically-extracted
    # reference_constraint (context_json.reference_constraints, computed
    # by reference_extraction.py from intent_text's "in front of you"),
    # never from look_at's own direction (which here points a DIFFERENT
    # way -- "left" -- than the resolved "front", proving the constraint,
    # not the look_at, is the real source).
    #
    # FOUND while applying the v4 fix (2026-09-04): the original sentence
    # here used "Remember THIS as 33" -- _CROSS_SENTENCE_ALIAS_PATTERNS
    # only recognizes "them/him/her/it", not "this", so it captured
    # bind_alias="" under the real, UNCHANGED extractor grammar. That was
    # fine before v4 (a lone constraint with an empty bind_alias was still
    # accepted), but v4 rejects an empty bind_alias unconditionally -- so
    # this test, whose actual purpose is entirely the look_at/relation
    # decoupling above and has nothing to do with bind_alias ownership,
    # needs a sentence whose bind_alias is actually captured. Swapped the
    # pronoun to "them" (an already-supported form) -- nothing else about
    # the test changed.
    mission, missions = _mission("There is a person right in front of you. Remember them as 33")
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


def test_planning_pipeline_preserves_an_explicit_look_instruction_as_a_real_action():
    # Identity/Grounding foundation finalization (2026-09-03, GPT
    # re-review): "Look to your left, then remember this person as 44" is
    # two real, distinct user intentions -- turn left, THEN remember
    # whoever you then see. The look_at here is an EXPLICIT physical
    # action the user asked for, not planner filler -- has_explicit_look_
    # instruction recognizes "Look to your left" and keeps it in the tree.
    # With 2 real physical actions and only an implicit goal_spec, this
    # correctly reaches PolicyGuard's rejection rather than a bare
    # look_at(left) x1 -- proving look_at survived canonicalization intact
    # (a silently-dropped look_at would instead leave exactly 1 physical
    # action, and this plan would succeed).
    mission, missions = _mission("Look to your left, then remember this person as 44.")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "44"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "policy"
    assert "physical skill" in result.message


def test_planning_pipeline_still_drops_a_look_at_with_no_explicit_instruction_of_its_own():
    # Contrast case for the test above: a bare look_at with no textual
    # support in intent_text at all (the planner added it speculatively,
    # with no basis in the user's words) is still dropped as redundant --
    # this is the ORIGINAL, still-working E.1 behavior, not a regression.
    mission, missions = _mission("Remember this person as 44.")
    plan_json = _implicit_plan_with_actions([
        {"type": "Action", "skill": "look_at", "args": {"direction": "left"}},
        {"type": "Action", "skill": "remember_entity", "args": {"target": "the person", "alias": "44"}},
    ])

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert len(bt["children"]) == 1


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
    # Direct unit test of the guard's own defensive check -- the real
    # extractor (reference_extraction.py) never produces an entry whose
    # source_span is not really in intent_text, but the guard must not
    # blindly trust context_json's shape either (the same "verify, don't
    # just trust the source" principle applied one layer further out).
    # GPT's original worked example: the user's own words never mention a
    # side at all -- a constraint CLAIMING "on your left" as its
    # source_span is citing text that was never there.
    plan = {"root": {"type": "Action", "skill": "remember_entity", "args": {
        "target": "the person", "alias": "44", "reference_constraint_id": "ref_1",
    }}}
    context_json = _context_json_with_reference_constraints({
        "constraint_id": "ref_1", "entity_class": "person", "relation": "left",
        "reference_frame": "robot", "source_span": "on your left",
    })

    error = _apply_reference_constraint_guard(
        plan, context_json=context_json, intent_text="Remember this person as 44")

    assert error is not None
    assert "was not found" in error


def test_reference_constraint_guard_rejects_a_constraint_relative_to_a_different_entity():
    # Direct unit test, same reasoning as above. GPT's sofa counter-example:
    # source_span is genuinely, verbatim present in intent_text -- but it
    # describes a relation to the SOFA, not to the robot. ResolveEntity
    # Reference only ever understands robot-relative bearings;
    # reinterpreting this as reference_frame="robot" would silently answer
    # a different question than the one the user actually asked.
    plan = {"root": {"type": "Action", "skill": "remember_entity", "args": {
        "target": "the person", "alias": "44", "reference_constraint_id": "ref_1",
    }}}
    context_json = _context_json_with_reference_constraints({
        "constraint_id": "ref_1", "entity_class": "person", "relation": "front",
        "reference_frame": "sofa", "source_span": "in front of the sofa",
    })

    error = _apply_reference_constraint_guard(
        plan, context_json=context_json, intent_text="Remember the person in front of the sofa as 44")

    assert error is not None
    assert "robot" in error


def test_reference_constraint_guard_rejects_a_constraint_not_owned_by_this_binding():
    # Direct unit test. GPT's chair counter-example: source_span is
    # genuinely present -- but it describes the CHAIR's position, not the
    # person being bound. FOUND (2026-09-03, GPT re-review): a prior round
    # left remember_entity with NO entity_class check at all here -- any
    # constraint, regardless of declared class, would be accepted for it.
    #
    # CHANGE APPROVAL 1 (2026-09-08, this run): entity_class no longer has
    # to equal "person" -- reference_extraction.py now also produces
    # object-noun constraints (real counter-example: two co-visible potted
    # plants, "the nearest plant", remember_entity correctly refusing to
    # guess). The safety property this test actually cares about --
    # "a constraint that isn't genuinely about THIS binding must never be
    # used for it" -- was never enforced by the entity_class check alone;
    # it is (and was) enforced independently by the bind_alias/ownership
    # check below, which this test still exercises unchanged: the fixture
    # constraint carries no bind_alias, so it is still unconditionally
    # rejected, just via that check now instead of a since-widened
    # entity_class check. See the companion test right below this one for
    # the new, previously-impossible acceptance case this change enables,
    # and the one after that for confirmation the ownership check still
    # closes the door even when entity_class alone would now pass.
    plan = {"root": {"type": "Action", "skill": "remember_entity", "args": {
        "target": "the person", "alias": "44", "reference_constraint_id": "ref_1",
    }}}
    context_json = _context_json_with_reference_constraints({
        "constraint_id": "ref_1", "entity_class": "chair", "relation": "left",
        "reference_frame": "robot", "source_span": "on your left",
    })

    error = _apply_reference_constraint_guard(
        plan, context_json=context_json, intent_text="The chair is on your left. Remember this person as 44.")

    assert error is not None
    assert "not deterministically associated" in error


def test_reference_constraint_guard_now_accepts_a_genuine_object_class_constraint():
    # CHANGE APPROVAL 1 (2026-09-08): the new, intended capability this
    # change adds -- binding an OBJECT via a real, ownership-matched
    # reference_constraint, exactly like remember_person already could for
    # people. Real utterance shape: "Remember the plant on your left as
    # Greenie." -- entity_class="plant" (not "person"), and bind_alias
    # correctly names THIS action's own alias.
    plan = {"root": {"type": "Action", "skill": "remember_entity", "args": {
        "target": "the plant", "alias": "Greenie", "reference_constraint_id": "ref_1",
    }}}
    context_json = _context_json_with_reference_constraints({
        "constraint_id": "ref_1", "entity_class": "plant", "relation": "left",
        "reference_frame": "robot", "source_span": "the plant on your left",
        "bind_alias": "Greenie",
    })

    error = _apply_reference_constraint_guard(
        plan, context_json=context_json,
        intent_text="Remember the plant on your left as Greenie.")

    assert error is None
    assert plan["root"]["args"]["relation"] == "left"


def test_reference_constraint_guard_still_rejects_wrong_owner_even_with_a_valid_object_class():
    # Confirms entity_class no longer being restricted to "person" did not
    # quietly widen WHICH binding a real object constraint can be used
    # for: same constraint as above (a real, validly-shaped plant/left
    # constraint with its own bind_alias="Greenie"), but this action binds
    # a DIFFERENT alias ("Fern") -- must still be rejected on ownership
    # grounds, entity_class validity alone is never sufficient.
    plan = {"root": {"type": "Action", "skill": "remember_entity", "args": {
        "target": "the plant", "alias": "Fern", "reference_constraint_id": "ref_1",
    }}}
    context_json = _context_json_with_reference_constraints({
        "constraint_id": "ref_1", "entity_class": "plant", "relation": "left",
        "reference_frame": "robot", "source_span": "the plant on your left",
        "bind_alias": "Greenie",
    })

    error = _apply_reference_constraint_guard(
        plan, context_json=context_json,
        intent_text="Remember the plant on your left as Greenie.")

    assert error is not None
    assert "Greenie" in error and "Fern" in error


def test_reference_constraint_guard_accepts_a_valid_constraint_and_fills_in_relation():
    # FOUND while applying the v4 fix below (2026-09-04): this test's
    # original sentence -- "The person on your left, remember them as
    # 44" -- is comma-joined, not the tight same-clause "as" marker
    # (_ALIAS_AFTER_PATTERNS requires immediate adjacency after the
    # relation phrase) and not the period-separated cross-sentence
    # pattern either -- so it captures bind_alias="" under the real,
    # UNCHANGED extractor grammar. Once the v4 fix below removes the
    # empty-bind_alias singleton exception, that sentence would now be
    # correctly REJECTED, not accepted -- this basic accept-path smoke
    # test needs a sentence whose bind_alias is actually captured to keep
    # testing what it always intended to (plain accept + relation
    # fill-in), so the phrasing changed to the tight same-clause "as"
    # form. Nothing about relation/alias expectations changed.
    mission, missions = _mission("Remember the person on your left as 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    assert json.loads(result.goal_spec_json)["args"]["relation"] == "left"


# --- constraint <-> alias/binding ownership (2026-09-03, GPT re-review v2) --
# A constraint's own relation/frame/class being real is not enough on its
# own -- when 2+ constraints exist in the same mission, a confused (not
# malicious) planner could still borrow a genuinely real constraint that
# belongs to a DIFFERENT mention for THIS action's alias/name. GPT's own
# worked example, reproduced exactly below.

def test_reference_constraint_guard_rejects_a_real_constraint_that_belongs_to_a_different_alias():
    # "The person on your left is waving. Remember the person on your
    # right as 44." extracts TWO real constraints (left, right, via the
    # real extractor). Referencing the LEFT one for alias "44" -- which
    # actually belongs to "right" -- must be rejected, even though the
    # LEFT constraint itself is completely real and independently valid.
    mission, missions = _mission(
        "The person on your left is waving. Remember the person on your right as 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        # ref_1 = left (extracted first); deliberately using it for "44",
        # which the sentence actually assigns to ref_2 = right.
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "not deterministically associated" in result.message


def test_reference_constraint_guard_accepts_the_correctly_matched_constraint_among_several():
    # Same sentence, same two constraints -- but this time referencing the
    # RIGHT constraint (ref_2) for alias "44", which is what the sentence
    # actually says. Must succeed.
    mission, missions = _mission(
        "The person on your left is waving. Remember the person on your right as 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_2"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    assert json.loads(result.goal_spec_json)["args"]["relation"] == "right"


def test_reference_constraint_guard_rejects_a_single_constraint_used_for_the_wrong_alias():
    # THE core fix (2026-09-03, GPT re-review v3): a v2 round's own
    # ownership check only ran when 2+ constraints existed, reasoning a
    # single constraint could never be ambiguous about which action it
    # belongs to -- GPT's re-review found that incomplete. "Remember the
    # person on your left as 33" extracts exactly ONE constraint, with
    # bind_alias="33" captured directly -- a planner mistakenly writing
    # name="44" while still referencing that SAME constraint must be
    # rejected too, exactly like the 2+-constraint case, not silently
    # let through just because nothing else was around to confuse it with.
    mission, missions = _mission("Remember the person on your left as 33.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "bound to alias/name '33'" in result.message


def test_reference_constraint_guard_accepts_a_single_constraint_used_for_its_own_correct_alias():
    # The positive counterpart -- the SAME captured bind_alias, referenced
    # for the alias it actually names, must still succeed.
    mission, missions = _mission("Remember the person on your left as 33.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "33", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    assert json.loads(result.goal_spec_json)["args"]["relation"] == "left"


def test_reference_constraint_guard_still_accepts_the_cross_sentence_e1_phrasing():
    # Confirms the original E.1 live-tested phrasing keeps working -- now
    # via reference_extraction.py's own explicit cross-sentence pattern
    # (bind_alias="44" captured directly), not a singleton bypass.
    mission, missions = _mission("There is a person right in front of you. Remember them as 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    assert json.loads(result.goal_spec_json)["args"]["relation"] == "front"


def test_reference_constraint_guard_rejects_a_single_constraint_with_no_captured_bind_alias():
    # THE v4 fix (2026-09-04, GPT re-review): v3 only required a
    # non-empty bind_alias to match the action's alias/name -- an EMPTY
    # bind_alias was still silently accepted whenever it was the ONLY
    # constraint in the mission (elif len(constraints) > 1), reasoning a
    # lone constraint could never be ambiguous about which action it
    # belongs to. GPT's re-review found that exception itself lets an
    # unowned real spatial constraint (relation/frame/class all genuinely
    # verified, but the sentence never actually names WHO it is for) be
    # used for identity binding under ANY alias the planner happens to
    # write, as long as nothing else is around to be confused with. Fixed:
    # empty bind_alias is now rejected unconditionally, no matter how many
    # constraints exist. "There is a person right in front of you. They
    # seem friendly." extracts exactly ONE constraint (front), with no
    # naming marker anywhere -- bind_alias="" -- so referencing it for
    # ANY alias must now be rejected.
    mission, missions = _mission("There is a person right in front of you. They seem friendly.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_1"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "not deterministically associated" in result.message


# NOTE: the "single constraint, real bind_alias, accept" regression asked
# for alongside the v4 fix above -- the original E.1 cross-sentence
# phrasing ("There is a person right in front of you. Remember them as
# 44."), bind_alias="44", referenced for alias "44" -- is already covered
# verbatim by test_reference_constraint_guard_still_accepts_the_cross_
# sentence_e1_phrasing above (added in v3, when the cross-sentence
# pattern was introduced); not duplicated here.


def test_reference_constraint_guard_rejects_any_constraint_without_a_captured_alias_when_multiple_exist():
    # Neither constraint has a captured bind_alias here (both use "is",
    # never a trusted marker) -- with 2+ constraints in play, NEITHER is
    # usable for identity binding at all, even for its own genuinely
    # correct alias.
    mission, missions = _mission(
        "The person on your left is 33. The person on your right is 44.")
    plan_json = _implicit_plan_with_single_action(
        "remember_entity",
        {"target": "the person", "alias": "44", "reference_constraint_id": "ref_2"},
    )

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert not result.ok
    assert result.stage == "reference_constraint"
    assert "not deterministically associated" in result.message


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


# --- CHANGE APPROVAL A (2026-09-05): append-all-missing-producer, zero -----
# pruning. Root cause: a trusted, pre-planner reference_constraint can
# already prove a mission requires a specific alias bound to a person, but
# Stage-2 can omit any producer of it entirely -- no existing deterministic
# layer required a plan to contain one. The fix only ever APPENDS a missing
# producer; it never deletes, prunes, reorders, or rewrites any existing
# Action. An earlier candidate that also pruned "redundant" same-target
# actions was rejected on adversarial review: this registry has no
# precondition/delete-effect model, so nothing here can formally prove an
# existing action irrelevant, and that candidate was shown to silently
# discard a real user request in exactly that case. These tests exist to
# pin the ZERO-PRUNING safety property directly, not just its net effect
# through the rest of the pipeline.
#
# B (multi-physical-action deterministic goal_spec composition) is a
# separate, already-registered, orthogonal gap -- NOT GRANTED, not touched
# here. A plan left with 2+ surviving physical actions and no structured
# goal_spec still correctly fails at PolicyGuard, exactly as it did before
# this change; that failure is expected and out of scope, not a regression.

def _reference_constraint(
    constraint_id: str, *, bind_alias: str, relation: str = "nearest",
    entity_class: str = "person", source_span: str,
) -> dict:
    return {
        "constraint_id": constraint_id, "entity_class": entity_class,
        "relation": relation, "reference_frame": "robot", "source_span": source_span,
        "bind_alias": bind_alias,
    }


def _skills_in(plan: dict) -> list[str]:
    return [str(a.get("skill") or "") for a in _all_actions_in(plan.get("root"))]


def test_append_missing_required_producers_is_a_no_op_when_a_producer_already_exists():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "remember_person", "args": {"name": "33", "target": "person"}},
    ]}}
    before = json.dumps(plan, sort_keys=True)
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住离你最近的人叫33")

    assert json.dumps(plan, sort_keys=True) == before


def test_append_missing_required_producers_appends_exactly_one_producer():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "search_for_entity", "args": {"target": "person"}},
        {"type": "Action", "skill": "approach_entity", "args": {"target": "person"}},
        {"type": "Action", "skill": "say", "args": {"text": "好的"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住离你最近的人叫33")

    remember_actions = [a for a in _all_actions_in(plan["root"]) if a["skill"] == "remember_person"]
    assert len(remember_actions) == 1
    assert remember_actions[0]["args"]["name"] == "33"
    assert remember_actions[0]["args"]["reference_constraint_id"] == "ref_1"
    assert "relation" not in remember_actions[0]["args"], (
        "relation must be left for the existing reference-constraint guard to fill in, "
        "never set directly here"
    )


def test_append_missing_required_producers_appends_both_of_two_missing_aliases():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "say", "args": {"text": "好的"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [
        _reference_constraint("ref_left", bind_alias="23", relation="left", source_span="你左边的人"),
        _reference_constraint("ref_right", bind_alias="44", relation="right", source_span="你右边的人"),
    ]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住你左边的人叫23，你右边的人叫44")

    names = sorted(
        a["args"]["name"] for a in _all_actions_in(plan["root"]) if a["skill"] == "remember_person")
    assert names == ["23", "44"], "both aliases must get their own producer, not just the first"


def test_append_missing_required_producers_does_nothing_for_an_invalid_constraint():
    # entity_class "chair" fails _validate_reference_constraint (the only
    # class reference_extraction.py ever produces is "person") -- existing
    # fail-closed behavior downstream is untouched, this stage does not
    # invent a producer for a class it cannot uniquely resolve.
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "search_for_entity", "args": {"target": "chair"}},
    ]}}
    before = json.dumps(plan, sort_keys=True)
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="99", entity_class="chair", source_span="离你最近的椅子")]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住离你最近的椅子叫99")

    assert json.dumps(plan, sort_keys=True) == before


def test_append_missing_required_producers_preserves_an_unrelated_gesture_exactly():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json,
        intent_text="记住离你最近的人叫33，然后跟他挥手")

    skills = _skills_in(plan)
    assert skills.count("play_animation") == 1
    wave = next(a for a in _all_actions_in(plan["root"]) if a["skill"] == "play_animation")
    assert wave["args"] == {"animation": "wave"}, "the gesture's own args must survive untouched"
    assert "remember_person" in skills


def test_append_missing_required_producers_preserves_a_same_target_physical_action():
    # The rejected v3 candidate would have pruned this: an approach_entity
    # whose own target happens to match the class the synthesized producer
    # also targets. This stage never inspects `target` at all -- it cannot
    # be tricked into deleting a same-target action because it has no
    # deletion logic of any kind.
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "approach_entity", "args": {"target": "person"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住离你最近的人叫33")

    skills = _skills_in(plan)
    assert skills.count("approach_entity") == 1
    assert "remember_person" in skills


def test_append_missing_required_producers_preserves_a_different_target_physical_action():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "approach_entity",
         "args": {"target": "entity_id", "entity_id": "person_known_other"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json,
        intent_text="去找那个人，然后把离你最近的人叫33")

    approach_actions = [a for a in _all_actions_in(plan["root"]) if a["skill"] == "approach_entity"]
    assert len(approach_actions) == 1
    assert approach_actions[0]["args"] == {"target": "entity_id", "entity_id": "person_known_other"}


def test_append_missing_required_producers_leaves_a_redundant_search_for_entity_for_later_stages():
    # This new stage's own job is only to append; whether an existing
    # search_for_entity is later cleaned up as redundant is
    # _canonicalize_redundant_locate_prefix's pre-existing, unmodified
    # responsibility (a separate stage, run after this one) -- not
    # something this function does or needs to know about.
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "search_for_entity", "args": {"target": "person"}},
        {"type": "Action", "skill": "say", "args": {"text": "好的"}},
    ]}}
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    _append_missing_required_producers(
        plan, context_json=context_json, intent_text="记住离你最近的人叫33")

    assert "search_for_entity" in _skills_in(plan)


def test_append_missing_required_producers_does_nothing_without_a_bind_alias_constraint():
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "go_to_place", "args": {"place": "kitchen"}},
    ]}}
    before = json.dumps(plan, sort_keys=True)

    _append_missing_required_producers(
        plan, context_json="{}", intent_text="go to the kitchen")

    assert json.dumps(plan, sort_keys=True) == before


def test_append_missing_required_producers_does_not_depend_on_intent_text_wording():
    # Producer selection reads only the already-validated constraint dict
    # (bind_alias, entity_class, constraint_id) -- never a literal keyword
    # in intent_text ("记住"/"叫"/a specific alias). Two very differently
    # worded sentences that both genuinely contain the same source_span
    # must produce byte-identical synthesis.
    context_json = json.dumps({"reference_constraints": [_reference_constraint(
        "ref_1", bind_alias="33", source_span="离你最近的人")]})

    plan_a = {"root": {"type": "Sequence", "children": [{"type": "Action", "skill": "say", "args": {}}]}}
    plan_b = {"root": {"type": "Sequence", "children": [{"type": "Action", "skill": "say", "args": {}}]}}

    _append_missing_required_producers(
        plan_a, context_json=context_json, intent_text="记住离你最近的人叫33")
    _append_missing_required_producers(
        plan_b, context_json=context_json,
        intent_text="麻烦你把离你最近的人记下来，他的名字是33")

    assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)


def test_planning_pipeline_end_to_end_synthesizes_the_missing_naming_terminal():
    # The exact E1 root-cause scenario, through the REAL pipeline: Stage-2
    # completely omits any alias-binding producer (a bare `say`, today's
    # ALSO-latent false-success case: 0 physical actions, implicit
    # goal_spec, previously accepted while binding nothing). The trusted
    # reference_constraint (real extractor output, not hand-built) proves
    # this mission requires binding alias "33" -- this change makes that
    # requirement real: the plan now actually binds it and the mission can
    # reach genuine, verifiable SUCCESS instead of a no-op false success.
    mission, missions = _mission("记住离你最近的人叫33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "say", "args": {"text": "好的"}},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    remember_actions = [a for a in bt["children"] if a["skill"] == "remember_person"]
    assert len(remember_actions) == 1
    assert remember_actions[0]["args"]["name"] == "33"
    assert remember_actions[0]["args"]["relation"] == "nearest"
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["type"] == "structured"
    assert goal_spec["predicate"] == "person_named"


def test_planning_pipeline_end_to_end_is_unaffected_when_the_producer_is_already_planned():
    mission, missions = _mission("记住离你最近的人叫33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "remember_person",
                  "args": {"name": "33", "target": "person"}},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert len([a for a in _all_actions_in(bt) if a["skill"] == "remember_person"]) == 1


def test_planning_pipeline_end_to_end_ordinary_navigation_is_unaffected():
    mission, missions = _mission()

    result = _pipeline(BootstrapPlanner()).plan(mission, missions)

    assert result.ok
    assert json.loads(result.bt_json)["type"] == "Sequence"
    assert json.loads(result.goal_spec_json)["predicate"] == "robot_at_place"


# --- B (MissionGoalContract V1, GPT-approved architecture amendment to ----
# FROZEN #14/#15): Omega's submit_mission stays one natural-language
# argument, but that argument may now carry a terminal typed trailer Bridge
# parses into context_json.caller_context.mission_goal_contract (same
# location grounded_entities already lives at -- see _grounded_entity_ids).
# A VALID single-goal contract is deterministically synthesized WITHOUT
# ever consulting the LLM candidate planner's own action list for
# authorization; ABSENT falls straight through to Change A and everything
# before it, completely unchanged; PRESENT_BUT_INVALID and >1 goals both
# REJECT outright, never a silent downgrade to legacy execution.

class _FakeResolver:
    def __init__(self, state="RESOLVED", entity_id="person_x1", grounding_ref="gnd_test1"):
        self.state, self.entity_id, self.grounding_ref = state, entity_id, grounding_ref
        self.calls = []

    def resolve(self, *, entity_class, relation, reference_frame):
        self.calls.append((entity_class, relation, reference_frame))
        if self.state != "RESOLVED":
            return self.state, "", ""
        return self.state, self.entity_id, self.grounding_ref


class _RaisingPlanner:
    def plan(self, intent_text: str, context_json: str = "") -> str:
        raise AssertionError(
            "the LLM candidate planner must never be called when a VALID "
            "single-goal MissionGoalContract is present")


def _goal_contract(*, targets: dict, goals: list) -> dict:
    return {"schema": "mc_ai_bt.mission_goal_contract.v1", "targets": targets, "goals": goals}


def _mission_with_contract(intent: str, contract: dict | None, *, grounded_entities=None):
    bridge_context: dict = {"schema": "mc.mission_context.v1"}
    if contract is not None:
        bridge_context["mission_goal_contract"] = contract
    if grounded_entities is not None:
        bridge_context["grounded_entities"] = grounded_entities
    return _mission(intent, json.dumps(bridge_context))


_SPATIAL_T1 = {"t1": {"kind": "spatial_reference", "entity_class": "person",
                       "relation": "nearest", "reference_frame": "robot"}}


def test_mission_goal_contract_turn1_naming_is_synthesized_without_calling_the_planner():
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33",
        _goal_contract(targets=_SPATIAL_T1, goals=[
            {"goal_id": "g1", "predicate": "person_named", "target_id": "t1", "alias": "33"}]))
    resolver = _FakeResolver()

    result = _pipeline(_RaisingPlanner(), reference_resolver=resolver).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert bt["skill"] == "remember_person"
    assert bt["args"]["name"] == "33"
    assert bt["args"]["entity_id"] == "person_x1"
    assert bt["args"][_INTERNAL_GROUNDING_FIELD] == "gnd_test1"
    goal_spec = json.loads(result.goal_spec_json)
    assert goal_spec["predicate"] == "person_named"
    assert _INTERNAL_GROUNDING_FIELD not in goal_spec["args"], (
        "the internal grounding field must never leak into goal_spec.args")
    assert resolver.calls == [("person", "nearest", "robot")]


def test_mission_goal_contract_turn2_alias_reference_uses_existing_grounded_entities():
    mission, missions = _mission_with_contract(
        "去33那里",
        _goal_contract(
            targets={"t1": {"kind": "alias_reference", "alias": "33"}},
            goals=[{"goal_id": "g1", "predicate": "entity_approached", "target_id": "t1"}]),
        grounded_entities=[{"alias": "33", "entity_id": "person_a1", "grounding_state": "RESOLVED"}])

    result = _pipeline(_RaisingPlanner()).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert bt["skill"] == "approach_entity"
    assert bt["args"]["entity_id"] == "person_a1"
    assert _INTERNAL_GROUNDING_FIELD not in bt["args"], (
        "alias-reference targets carry no grounding_ref -- no new identity decision is made")


def test_mission_goal_contract_turn2_unresolved_alias_fails_closed_no_nearest_fallback():
    mission, missions = _mission_with_contract(
        "去33那里",
        _goal_contract(
            targets={"t1": {"kind": "alias_reference", "alias": "33"}},
            goals=[{"goal_id": "g1", "predicate": "entity_approached", "target_id": "t1"}]),
        grounded_entities=[{"alias": "33", "entity_id": "", "grounding_state": "UNRESOLVED"}])

    result = _pipeline(_RaisingPlanner()).plan(mission, missions)

    assert not result.ok
    assert result.stage == "target_resolution"
    assert not result.bt_json


def test_mission_goal_contract_ambiguous_resolution_authorizes_no_physical_bt():
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33",
        _goal_contract(targets=_SPATIAL_T1, goals=[
            {"goal_id": "g1", "predicate": "person_named", "target_id": "t1", "alias": "33"}]))

    result = _pipeline(_RaisingPlanner(), reference_resolver=_FakeResolver(state="AMBIGUOUS")).plan(
        mission, missions)

    assert not result.ok
    assert result.stage == "target_resolution"
    assert not result.bt_json


def test_mission_goal_contract_not_found_authorizes_no_physical_bt():
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33",
        _goal_contract(targets=_SPATIAL_T1, goals=[
            {"goal_id": "g1", "predicate": "person_named", "target_id": "t1", "alias": "33"}]))

    result = _pipeline(_RaisingPlanner(), reference_resolver=_FakeResolver(state="NOT_FOUND")).plan(
        mission, missions)

    assert not result.ok
    assert result.stage == "target_resolution"


def test_mission_goal_contract_unknown_resolution_authorizes_no_physical_bt():
    # No resolver injected at all -- _resolve_mission_target must report
    # UNKNOWN, never fabricate RESOLVED.
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33",
        _goal_contract(targets=_SPATIAL_T1, goals=[
            {"goal_id": "g1", "predicate": "person_named", "target_id": "t1", "alias": "33"}]))

    result = _pipeline(_RaisingPlanner(), reference_resolver=None).plan(mission, missions)

    assert not result.ok
    assert result.stage == "target_resolution"
    assert "UNKNOWN" in result.message


def test_mission_goal_contract_present_but_malformed_rejects_not_falls_back_to_legacy():
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33", {"schema": "mc_ai_bt.mission_goal_contract.v1", "targets": {}, "goals": []})

    result = _pipeline(_RaisingPlanner()).plan(mission, missions)

    assert not result.ok
    assert result.stage == "mission_goal_contract"
    assert not result.bt_json


def test_mission_goal_contract_unknown_predicate_rejects():
    mission, missions = _mission_with_contract(
        "记住离你最近的人叫33",
        _goal_contract(targets=_SPATIAL_T1, goals=[
            {"goal_id": "g1", "predicate": "not_a_real_predicate", "target_id": "t1"}]))

    result = _pipeline(_RaisingPlanner()).plan(mission, missions)

    assert not result.ok
    assert result.stage == "mission_goal_contract"


def test_mission_goal_contract_multi_goal_is_unsupported_v1_fail_closed():
    targets = dict(_SPATIAL_T1)
    targets["t2"] = {"kind": "spatial_reference", "entity_class": "person",
                      "relation": "left", "reference_frame": "robot"}
    mission, missions = _mission_with_contract(
        "走到你左边的人那里，然后记住你右边的人叫33",
        _goal_contract(targets=targets, goals=[
            {"goal_id": "g1", "predicate": "entity_approached", "target_id": "t2"},
            {"goal_id": "g2", "predicate": "person_named", "target_id": "t1", "alias": "33"},
        ]))

    result = _pipeline(_RaisingPlanner(), reference_resolver=_FakeResolver()).plan(mission, missions)

    assert not result.ok
    assert result.stage == "mission_goal_contract"
    assert "UNSUPPORTED_V1" in result.message
    assert not result.bt_json, "no partial physical execution for an unsupported multi-goal contract"


def test_mission_goal_contract_absent_leaves_legacy_change_a_path_completely_unchanged():
    mission, missions = _mission("记住离你最近的人叫33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "say", "args": {"text": "好的"}},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    remember_actions = [a for a in json.loads(result.bt_json)["children"] if a["skill"] == "remember_person"]
    assert len(remember_actions) == 1
    assert remember_actions[0]["args"]["name"] == "33"


def test_planner_forged_internal_grounding_field_is_stripped_on_legacy_path():
    plan = {"root": {"type": "Action", "skill": "remember_person",
                      "args": {"name": "33", "entity_id": "person_x1",
                               _INTERNAL_GROUNDING_FIELD: "FORGED_BY_LLM"}}}
    _strip_internal_grounding_field(plan)
    assert _INTERNAL_GROUNDING_FIELD not in plan["root"]["args"]


def test_planner_forged_internal_grounding_field_end_to_end_legacy_path():
    # No MissionGoalContract at all -- the LLM's own plan_json directly
    # supplies a forged internal field. It must never survive to bt_json.
    mission, missions = _mission("记住离你最近的人叫33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "remember_person",
                  "args": {"name": "33", "target": "person",
                           _INTERNAL_GROUNDING_FIELD: "FORGED_BY_LLM"}},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert _INTERNAL_GROUNDING_FIELD not in bt["args"]


def test_mission_goal_contract_normal_relation_remember_path_is_untouched():
    # The existing relation/reference_constraint_id remember path (its own
    # execution-time ResolveEntityReference call, fresher than any
    # planning-time resolution) is not part of this change's scope at all --
    # confirm ordinary Change-A behavior for it is bit-for-bit unaffected.
    mission, missions = _mission("记住离你最近的人叫33")
    plan_json = json.dumps({
        "schema": "mc_ai_bt.plan.v1",
        "root": {"type": "Action", "skill": "remember_person",
                  "args": {"name": "33", "target": "person"}},
        "goal_spec": {"type": "human", "verification": {"mode": "implicit_conversation"}},
    })

    result = _pipeline(StaticPlanner(plan_json)).plan(mission, missions)

    assert result.ok, result.message
    bt = json.loads(result.bt_json)
    assert _INTERNAL_GROUNDING_FIELD not in bt["args"]


def test_parse_mission_goal_contract_reads_caller_context_same_place_as_grounded_entities():
    context_json = json.dumps({
        "caller_context": {
            "mission_goal_contract": _goal_contract(targets=_SPATIAL_T1, goals=[
                {"goal_id": "g1", "predicate": "person_named", "target_id": "t1", "alias": "33"}]),
        },
    })
    state, goals, targets = _parse_mission_goal_contract(context_json)
    assert state == "VALID"
    assert len(goals) == 1
    assert "t1" in targets
