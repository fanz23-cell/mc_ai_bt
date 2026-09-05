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


from mc_ai_bt.planner import _self_sufficient_terminal_rule, build_planner_messages
from mc_ai_bt.skill_registry import DEFAULT_SKILLS, SkillRegistry


def _registry_declared_terminals() -> set[str]:
    return {
        name for name, spec in DEFAULT_SKILLS.items()
        if spec.self_sufficient_physical_terminal
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
    rule = _self_sufficient_terminal_rule()
    for name in _registry_declared_terminals():
        assert name in rule, f"{name} declares self_sufficient_physical_terminal but the derived rule omits it"


def test_the_derived_rule_actually_reaches_the_planner_prompt():
    prompt = _prompt_text()
    rule = _self_sufficient_terminal_rule()
    assert rule, "the registry declares terminals, so the rule must be non-empty"
    assert rule in prompt


def test_the_rule_forbids_travelling_to_a_target_just_to_name_it():
    # The specific live failure: approach_entity in front of remember_person.
    rule = _self_sufficient_terminal_rule()
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
        assert not DEFAULT_SKILLS[name].self_sufficient_physical_terminal
