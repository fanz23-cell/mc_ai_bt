"""FOUND LIVE 2026-09-05 (E1 TURN1, third real attempt): the planner put an
approach_entity walk in front of remember_person and invented an entity_id out
of the alias the mission was supposed to CREATE. The prompt rule meant to stop
the first half had been written as a hand-typed sentence naming exactly two
skills (look_at, search_for_entity), so a third skill stepped around it.

That is the third time in this codebase a hand-maintained skill-name set went
stale (see skill_registry.py's own comments on `dispatch` and
`subsumes_locate_skills`, and the omegaclaw prompt's hard-coded skill list).
These tests exist so it cannot be the fourth: the property is declared once on
SkillSpec and the prompt sentence is derived from it, and if someone adds a
self-sufficient terminal skill without the prompt following along, this fails.
"""


from mc_ai_bt.planner import _target_acquisition_rule, build_planner_messages
from mc_ai_bt.skill_registry import DEFAULT_SKILLS, SkillRegistry


def _registry_declared_terminals() -> set[str]:
    return {
        name for name, spec in DEFAULT_SKILLS.items()
        if spec.resolves_own_target_acquisition
    }


def _prompt_text() -> str:
    messages = build_planner_messages(
        "remember the nearest person as 33", "{}", skill_registry=SkillRegistry())
    # build_planner_messages returns (role, content) tuples.
    return "\n".join(content for _role, content in messages)


def test_the_registry_actually_declares_the_remember_skills_as_self_sufficient():
    # If this ever shrinks to nothing the derived rule silently disappears from
    # the prompt, which is exactly the regression that would let the planner go
    # back to walking to someone before naming them.
    assert _registry_declared_terminals() == {"remember_person", "remember_entity"}


def test_every_declared_terminal_appears_in_the_derived_rule():
    rule = _target_acquisition_rule()
    for name in _registry_declared_terminals():
        assert name in rule, f"{name} declares resolves_own_target_acquisition but the derived rule omits it"


def test_the_derived_rule_actually_reaches_the_planner_prompt():
    prompt = _prompt_text()
    rule = _target_acquisition_rule()
    assert rule, "the registry declares terminals, so the rule must be non-empty"
    assert rule in prompt


def test_the_rule_forbids_travelling_to_a_target_just_to_name_it():
    # The specific live failure: approach_entity in front of remember_person.
    rule = _target_acquisition_rule()
    assert "approach_entity" in rule
    assert "go_to_place" in rule
    assert "SEEING" in rule


def test_the_prompt_states_the_alias_is_never_an_entity_id_unconditionally():
    # The live plan used the not-yet-bound alias "33" as an entity_id. The old
    # prohibition existed but sat inside an "if grounded_entities is present"
    # clause, so for a pure binding task -- where the alias is by definition NOT
    # in grounded_entities -- the whole paragraph read as inapplicable.
    prompt = _prompt_text()
    assert "A name or alias the user SPOKE is never an" in prompt
    assert "This holds unconditionally" in prompt


def test_the_catalog_no_longer_teaches_approach_then_remember():
    # remember_entity's own catalog description used to say that getting an
    # entity_id from locate/search/approach_entity first was "the ONLY way" to
    # bind 11/22/33-style aliases -- the prompt was literally teaching the
    # planner to approach before remembering.
    description = DEFAULT_SKILLS["remember_entity"].description
    assert "approach_entity" not in description
    assert "ONLY way" not in description
    assert "The alias you are creating is NEVER an entity_id" in description


def test_approach_entitys_args_do_not_conflate_a_label_with_an_id():
    args = DEFAULT_SKILLS["approach_entity"].args_schema
    assert "entity_id" not in str(args["target"]).split("(")[0], (
        "target is a human-readable label; listing entity_id as one of its accepted "
        "values is what let the planner put a spoken alias there"
    )
    assert "never a name/alias the user spoke" in str(args["entity_id"])


def test_self_sufficient_terminals_are_a_superset_of_nothing_unexpected():
    # A skill that genuinely needs the robot to travel (approach_entity,
    # go_to_place, come_to_me) must never be declared self-sufficient -- the
    # rule tells the planner such skills are exactly what NOT to plan alongside.
    for name in ("approach_entity", "go_to_place", "come_to_me", "simple_move"):
        assert not DEFAULT_SKILLS[name].resolves_own_target_acquisition


def test_the_rule_also_forbids_a_precondition_gate_in_front_of_the_terminal():
    # FOUND LIVE 2026-09-05, fourth attempt: with the physical-Action half of
    # this rule in place the planner stopped adding approach_entity -- and put a
    # Condition(entity_located) in front of remember_person instead. Nothing in
    # that plan produces entity_located (remember_person produces person_named),
    # so the gate could only ever be UNKNOWN and the mission stalled awaiting a
    # decision. Same concept as the Action half: a self-sufficient terminal
    # establishes its own preconditions, so nothing precedes it -- the rule has
    # to say that about node types, not only about skills.
    rule = _target_acquisition_rule()
    assert "entity_located" in rule
    assert "produces rather than needs" in rule
    # and it must NOT over-claim about the rest of the mission
    assert "ONLY physical Action" not in rule


def test_an_unsatisfiable_gate_before_a_terminal_is_removed_deterministically():
    # The live plan, verbatim, from the fourth and fifth E1 TURN1 attempts.
    from mc_ai_bt.planning_pipeline import _canonicalize_gate_the_terminal_resolves_itself
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Condition", "predicate": "entity_located"},
        {"type": "Action", "skill": "remember_person", "args": {"name": "33", "target": "person"}},
    ]}}
    _canonicalize_gate_the_terminal_resolves_itself(plan)
    kinds = [c["type"] for c in plan["root"]["children"]]
    assert kinds == ["Action"], "the unsatisfiable gate must be gone"
    assert plan["root"]["children"][0]["skill"] == "remember_person"


def test_a_gate_whose_predicate_an_earlier_action_really_produces_is_kept():
    # "locate, then confirm that locate worked" is a legitimate pair and must
    # survive untouched -- this is the whole reason the check is producer-aware
    # instead of just deleting Conditions in front of remember_*.
    from mc_ai_bt.planning_pipeline import _canonicalize_gate_the_terminal_resolves_itself
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "locate_entity", "args": {"target": "person"}},
        {"type": "Condition", "predicate": "entity_located"},
        {"type": "Action", "skill": "remember_person", "args": {"name": "33", "target": "person"}},
    ]}}
    _canonicalize_gate_the_terminal_resolves_itself(plan)
    kinds = [c["type"] for c in plan["root"]["children"]]
    assert kinds == ["Action", "Condition", "Action"]


def test_a_gate_in_front_of_an_ordinary_skill_is_never_touched():
    from mc_ai_bt.planning_pipeline import _canonicalize_gate_the_terminal_resolves_itself
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Condition", "predicate": "entity_located"},
        {"type": "Action", "skill": "approach_entity", "args": {"target": "person"}},
    ]}}
    before = json_like = str(plan)
    _canonicalize_gate_the_terminal_resolves_itself(plan)
    assert str(plan) == before


def test_the_owned_reference_constraint_is_attached_when_the_planner_omits_it():
    # The live sixth-attempt plan: correct single action, but no
    # reference_constraint_id, so "离你最近的人" was lost and the skill refused
    # between two visible people.
    import json as _json
    from mc_ai_bt.planning_pipeline import _inject_owned_reference_constraint
    from mc_ai_bt.reference_extraction import extract_reference_constraints

    intent = "记住离你最近的人叫33"
    context = _json.dumps({"reference_constraints": extract_reference_constraints(intent)})
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "remember_person", "args": {"name": "33", "target": "person"}},
    ]}}
    _inject_owned_reference_constraint(plan, context_json=context, intent_text=intent)
    args = plan["root"]["children"][0]["args"]
    assert args.get("reference_constraint_id") == "ref_1"


def test_nothing_is_attached_when_the_constraint_names_a_different_alias():
    # Same ownership test the guard enforces, applied in the other direction:
    # a constraint whose sentence names 33 must never be attached to an action
    # binding 44.
    import json as _json
    from mc_ai_bt.planning_pipeline import _inject_owned_reference_constraint
    from mc_ai_bt.reference_extraction import extract_reference_constraints

    intent = "记住离你最近的人叫33"
    context = _json.dumps({"reference_constraints": extract_reference_constraints(intent)})
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "remember_person", "args": {"name": "44", "target": "person"}},
    ]}}
    _inject_owned_reference_constraint(plan, context_json=context, intent_text=intent)
    assert "reference_constraint_id" not in plan["root"]["children"][0]["args"]


def test_an_action_that_already_chose_a_constraint_is_left_alone():
    import json as _json
    from mc_ai_bt.planning_pipeline import _inject_owned_reference_constraint
    from mc_ai_bt.reference_extraction import extract_reference_constraints

    intent = "记住离你最近的人叫33"
    context = _json.dumps({"reference_constraints": extract_reference_constraints(intent)})
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Action", "skill": "remember_person",
         "args": {"name": "33", "target": "person", "reference_constraint_id": "ref_9"}},
    ]}}
    _inject_owned_reference_constraint(plan, context_json=context, intent_text=intent)
    assert plan["root"]["children"][0]["args"]["reference_constraint_id"] == "ref_9"


# --- the P0 this file exists to prevent recurring -------------------------
# A first version of the canonicalizer decided a gate was unsatisfiable
# whenever no EARLIER ACTION produced its predicate. That inference is false:
# a Condition may check a fact that already holds in WorldState. Written that
# way it silently deleted legitimate pre-execution checks -- a worse defect
# than the stall it fixed. These are the cases that catch it.

def test_a_legitimate_worldstate_gate_with_no_producer_is_never_removed():
    # person_visible is produced by NO skill at all -- it is answered from
    # WorldState. Gating a remember on it is entirely legitimate, and the old
    # "no earlier producer means unsatisfiable" rule would have deleted it.
    from mc_ai_bt.planning_pipeline import _canonicalize_gate_the_terminal_resolves_itself
    from mc_ai_bt.skill_registry import DEFAULT_SKILLS
    assert not any("person_visible" in s.result_predicates for s in DEFAULT_SKILLS.values()), (
        "this test's premise is that person_visible has no producing skill")
    plan = {"root": {"type": "Sequence", "children": [
        {"type": "Condition", "predicate": "person_visible"},
        {"type": "Action", "skill": "remember_person", "args": {"name": "33", "target": "person"}},
    ]}}
    _canonicalize_gate_the_terminal_resolves_itself(plan)
    kinds = [c["type"] for c in plan["root"]["children"]]
    assert kinds == ["Condition", "Action"], "a legitimate WorldState gate must survive"


def test_only_predicates_the_terminal_itself_resolves_are_removable():
    from mc_ai_bt.skill_registry import internally_resolved_predicates
    resolved = internally_resolved_predicates("remember_person")
    assert resolved == {"entity_located", "search_for_entity_completed"}
    # everything else a plan might gate on stays outside that set
    for predicate in ("person_visible", "person_named", "robot_at_place", "entity_approached"):
        assert predicate not in resolved


def test_a_skill_that_does_not_acquire_its_own_target_resolves_nothing():
    from mc_ai_bt.skill_registry import internally_resolved_predicates
    assert internally_resolved_predicates("approach_entity") == frozenset()


def test_the_rule_no_longer_claims_the_mission_may_hold_nothing_else():
    # "wave at the nearest person, then remember them as 33" is a legitimate
    # two-action mission; the rule must not forbid it.
    rule = _target_acquisition_rule()
    assert "rest of the mission" in rule
    assert "if the user genuinely asked for another action too, plan it" in rule
