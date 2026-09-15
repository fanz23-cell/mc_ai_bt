import json
from types import SimpleNamespace

from mc_ai_bt.planner import LlmJsonPlanner, build_planner_messages


class FakeModel:
    def __init__(self, content: str):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_llm_json_planner_extracts_json_from_model_response():
    model = FakeModel(
        """
        ```json
        {"root":{"type":"Action","skill":"say","args":{"text":"hi"}},
         "goal_spec":{"type":"human","verification":{"mode":"implicit_conversation"}}}
        ```
        """
    )

    plan = json.loads(LlmJsonPlanner(model).plan("say hi", '{"world":{}}'))

    assert plan["schema"] == "mc_ai_bt.plan.v1"
    assert plan["root"]["skill"] == "say"
    assert plan["context_json"] == '{"world":{}}'
    assert model.messages[0][0] == "system"


def test_llm_json_planner_recovers_from_a_trailing_line_comment():
    # FOUND LIVE 2026-09-09: root-caused via full-session log analysis plus a
    # captured raw sample -- 175/531 (~33%) of all mission submissions that
    # session failed with an "Expecting ',' delimiter" JSONDecodeError. The
    # actual cause: the planner LLM sometimes appends a JavaScript-style
    # "// ..." comment after a value, hedging about a placeholder entity_id
    # it wasn't fully sure of. This reproduces that exact shape.
    model = FakeModel(
        """
        {"root":{"type":"Sequence","children":[
         {"type":"Action","skill":"search_for_entity","args":{"target":"bottle"},"timeout_sec":30},
         {"type":"Action","skill":"approach_entity","args":{
           "target":"the bottle",
           "entity_id":"bottle_entity_id"  // placeholder, replace with the real grounded id
         },"timeout_sec":30}
        ]},
         "goal_spec":{"type":"human","verification":{"mode":"implicit_conversation"}}}
        """
    )

    plan = json.loads(LlmJsonPlanner(model).plan("find the bottle", "{}"))

    assert plan["root"]["type"] == "Sequence"
    assert [child["skill"] for child in plan["root"]["children"]] == [
        "search_for_entity",
        "approach_entity",
    ]
    assert plan["root"]["children"][1]["args"]["entity_id"] == "bottle_entity_id"


def test_llm_json_planner_comment_stripping_ignores_urls_in_strings():
    # A literal "//" inside a real string value (e.g. a URL) must survive --
    # the comment stripper only fires outside of string content.
    model = FakeModel(
        '{"root":{"type":"Action","skill":"say",'
        '"args":{"text":"see https://example.com/path for details"}}}'
    )

    plan = json.loads(LlmJsonPlanner(model).plan("say the url", "{}"))

    assert plan["root"]["args"]["text"] == "see https://example.com/path for details"


def test_llm_json_planner_recovers_from_a_missing_comma():
    # A genuinely dropped comma (no comment involved) is kept as a second-line
    # fallback repair, distinct from the comment-stripping path above.
    model = FakeModel(
        """
        {"root":{"type":"Sequence","children":[
         {"type":"Action","skill":"say","args":{"text":"hi"},"timeout_sec":30}
         {"type":"Action","skill":"approach_entity","args":{"entity_id":"x"},"timeout_sec":30}
        ]},
         "goal_spec":{"type":"human","verification":{"mode":"implicit_conversation"}}}
        """
    )

    plan = json.loads(LlmJsonPlanner(model).plan("say hi then approach x", "{}"))

    assert plan["root"]["type"] == "Sequence"
    assert [child["skill"] for child in plan["root"]["children"]] == [
        "say",
        "approach_entity",
    ]


def test_llm_json_planner_still_rejects_json_broken_for_another_reason():
    # The comma-repair path must not mask a genuinely different malformation --
    # confirm an unrelated syntax error still raises instead of being swallowed.
    model = FakeModel('{"root": {"type": }}')

    try:
        LlmJsonPlanner(model).plan("say hi", "{}")
    except ValueError as exc:
        assert "unparseable JSON" in str(exc)
    else:
        raise AssertionError("expected a ValueError for genuinely broken JSON")


def test_planner_prompt_no_longer_claims_a_same_plan_locate_as_an_entity_id_source():
    # FOUND LIVE 2026-09-09: this claim taught the model to invent a plausible-looking
    # entity_id (e.g. "bottle_entity_id") for a same-plan search_for_entity target with no
    # bound alias -- _apply_grounding_normalizer in planning_pipeline.py never accepts an
    # entity_id from any source but context_json.caller_context.grounded_entities, so every
    # such plan was rejected. The prompt must no longer claim that second source exists, and
    # must instead say to omit entity_id and rely on target alone in that situation.
    messages = build_planner_messages("find the water bottle", "{}")
    system = messages[0][1]

    assert "earlier locate-type Action in this same plan" not in system
    assert "exactly two legitimate" not in system
    assert "OMITS entity_id entirely" in system
    assert "only legitimate source is an entry of context_json.caller_context.grounded_entities" in system


def test_planner_prompt_teaches_a_structured_goal_spec_for_search_then_approach():
    # FOUND LIVE 2026-09-09: companion to the entity_id fix above -- once the model stops
    # inventing an entity_id, it still defaulted to a human-type goal_spec for a
    # search_for_entity + approach_entity plan, which PolicyGuard correctly rejects (the
    # deterministic auto-fill never covers a 2-physical-action plan). The prompt must now
    # teach the one, non-ambiguous correct structured goal_spec for this exact shape.
    messages = build_planner_messages("find the water bottle", "{}")
    system = messages[0][1]

    assert '"predicate": "entity_approached"' in system
    assert "do not default to a human-type" in system


def test_planner_prompt_names_allowed_skills_and_constraints():
    messages = build_planner_messages("go to test_place", "{}")
    system = messages[0][1]

    assert "mc_ai_bt.plan.v1" in system
    assert "go_to_place" in system
    assert "point_at:" in system
    assert "look_at may use only direction" in system
    assert "Do not invent ROS topics" in system
    assert "UNKNOWN" in system


def test_planner_prompt_omits_blocked_skills():
    # FOUND LIVE 2026-08-31: align_axis/move_along_axis/maintain_distance/
    # wait_for_contact/detect_contact (status="blocked" in skill_registry.py) used to be
    # serialized into this same prompt line, indistinguishable from a working skill.
    messages = build_planner_messages("go to test_place", "{}")
    system = messages[0][1]

    for blocked in ("align_axis", "move_along_axis", "maintain_distance", "wait_for_contact", "detect_contact"):
        assert f"- {blocked}:" not in system, f"{blocked} is status=blocked and must not be in the planner prompt"
    # A representative still-available skill must still be present, proving this isn't
    # just an empty/broken catalog.
    assert "- go_to_place:" in system


def test_planner_prompt_lists_known_predicate_names():
    # Found live 2026-08-29 (§C2): the planner had no idea which predicate names
    # actually exist, so it repeatedly either invented one ("root.children[1].
    # predicate is required" validator rejections) or gave up on a structured
    # goal_spec entirely and pre-wrote a guessed conclusion into a say node instead.
    messages = build_planner_messages("is the red_mug near the blue_shelf", "{}")
    system = messages[0][1]

    assert "relation_checked" in system
    assert "robot_at_place" in system
    assert "never invented or guessed" in system


def test_planner_prompt_warns_against_pre_written_check_conclusions():
    messages = build_planner_messages("check whether the door is open", "{}")
    system = messages[0][1]

    assert "fixed at planning time" in system


def test_planner_prompt_explains_grounded_entities():
    # E.1 (2026-09-02): the model must be told to USE a resolved entity_id
    # from context_json.caller_context.grounded_entities when present, and
    # explicitly told never to invent one or fall back to a class-based
    # search once entity_id resolution might fail at execution time.
    messages = build_planner_messages("go to 33", "{}")
    system = messages[0][1]

    assert "grounded_entities" in system
    assert "entity_id" in system
    assert "never invent" in system.lower()


def test_planner_prompt_teaches_the_reference_constraints_contract():
    # Gate-1.1 architecture round (2026-09-03, GPT re-review): the planner
    # must be taught the structured reference_constraints schema AND told
    # never to set relation directly / never to treat look_at as evidence
    # -- planning_pipeline.py's _apply_reference_constraint_guard is the
    # deterministic backstop, but the model needs to know the contract
    # exists at all to comply with it in the first place.
    messages = build_planner_messages("remember this person as 44", "{}")
    system = messages[0][1]

    assert "reference_constraints" in system
    assert "constraint_id" in system
    assert "source_span" in system
    assert "must NEVER be set directly" in system
    assert "look_at" in system and "NEVER evidence" in system


def test_planner_prompt_lists_wait_for_event_node_type():
    messages = build_planner_messages("wait until a customer raises a hand", "{}")
    system = messages[0][1]

    assert "WaitForEvent" in system
    assert "poll_interval_sec" in system


def test_planner_prompt_includes_args_schema_for_every_skill():
    # Found live 2026-08-29 building remember_place: the skill catalog line used to
    # carry only resources+description, never args_schema, so the model had no idea
    # what argument keys a brand-new skill took and generated a node missing
    # args.name outright. Confirmed live: adding args_schema to this line fixed it
    # 3/3 tries where it had failed before. This is the systemic guard against that
    # class of bug regressing for ANY future skill, not just this one.
    messages = build_planner_messages("remember this spot as front_desk", "{}")
    system = messages[0][1]

    assert '"name": "place name to save"' in system
    assert '"target": "entity|entity_id|object|person"' in system


def test_planner_prompt_warns_against_visible_people_and_wrong_predicate_after_search():
    # Found live 2026-08-30: a real submitted mission ("find a woman in the room")
    # planned search_for_entity(target="woman") followed by a Condition checking
    # predicate "visible_people" -- not a real predicate name at all (it's an
    # internal world-state fact key person_visible's own evaluation reads), and
    # even if it had been the real person_visible, that answers a different
    # question than "did the search just find her". The mission could never
    # succeed as a result, regardless of whether a woman was actually present.
    messages = build_planner_messages("search the room for a woman", "{}")
    system = messages[0][1]

    assert '"visible_people" is never a valid predicate name' in system
    assert "search_for_entity_completed" in system
    assert "did that just" in system


def test_planner_prompt_steers_generic_person_presence_to_person_visible():
    # Found live 2026-08-30: search_for_entity(target="woman") always fails --
    # confirmed by calling the real object/person classifier directly: it only
    # recognizes a fixed 80-class vocabulary (person, chair, bottle, ... zebra),
    # no gender-specific classes, and rejects "woman" outright regardless of
    # whether a woman is actually in the room. person_visible, by contrast, is
    # fed by a continuously-running pose detector with real evidence -- for a
    # plain "is anyone there" question, that is the predicate that can actually
    # answer, not a search_for_entity call the classifier was never going to
    # resolve.
    messages = build_planner_messages("look for a woman in the room", "{}")
    system = messages[0][1]

    assert "closed, fixed set of" in system
    assert "go straight to a Condition/GoalCheck with predicate person_visible" in system
    assert '"cardboard package box"' in system


def test_planner_prompt_warns_there_is_no_stop_skill():
    # Found live 2026-08-31, twice, on two different plans: the planner tried
    # to satisfy "...and stop right next to it"/"...stop right beside it" with
    # a dedicated "stop" step -- once as an invented top-level skill (rejected:
    # "root.children[2].skill is unknown: stop"), once as simple_move's action
    # arg (rejected: "args.action is not allowed: 'stop'"). Both plans were
    # correctly rejected by the validator, but the mission the person actually
    # asked for never got planned at all -- approach_entity/go_to_place/
    # come_to_me already stop on arrival, no separate step is ever needed.
    messages = build_planner_messages(
        "navigate to the plant and stop right next to it", "{}")
    system = messages[0][1]

    assert "no 'stop' skill" in system
    assert "already stop on arrival" in system


def test_planner_prompt_warns_sequential_steps_are_not_parallel():
    # Found live 2026-08-31: "turn right, then look for what's there" (an
    # explicitly two-step, sequential instruction) planned as two children
    # both needing the "gaze" resource under what the validator's own error
    # ("root.children[1].children[0] and ...children[1] have conflicting
    # resources: ['gaze']") identified as a concurrent-execution node --
    # correctly rejected, but the mission the person asked for was never
    # planned at all.
    messages = build_planner_messages(
        "turn right, then look for what's there", "{}")
    system = messages[0][1]

    assert "is a Sequence, not" in system
    assert "Parallel means both children run at once" in system


# --- P0.5 (z-doc 92): VisualCheck mode="observe" vs mode="condition" -------
# CONFIRMED root cause: the planner's own capability guidance predates the
# observe-mode/visual_check_completed contract (z-doc 91, already LIVE
# VERIFIED at the runtime/GoalCheck/MissionOutcome/Bridge/Omega layer) -- the
# model has no way to know this shape exists unless the prompt teaches it.
# These tests check PROMPT CONTENT only (build_planner_messages is a pure,
# static-per-catalog function of the skill registry, not of intent_text --
# same convention every other test in this file already uses), since the
# real LlmJsonPlanner's actual choice depends on a live LLM call this test
# suite cannot make deterministically.

def test_planner_prompt_teaches_observe_mode_for_an_observation_request():
    messages = build_planner_messages(
        "check whether the plant on the table is dry", "{}")
    system = messages[0][1]

    assert '"mode": "observe"' in system
    assert "visual_check_completed" in system
    assert "OBSERVE (mode=\"observe\")" in system


def test_planner_prompt_covers_paraphrased_observation_requests():
    # Different surface wording for the same underlying request shape --
    # the prompt's own trigger-phrase list must cover more than one phrasing,
    # not just the single canonical example.
    messages = build_planner_messages(
        "look and tell me whether the window is closed", "{}")
    system = messages[0][1]

    assert "'look and tell me whether X'" in system
    assert "'verify whether X and report the answer'" in system


def test_planner_prompt_teaches_false_is_a_valid_completed_observe_outcome():
    messages = build_planner_messages(
        "determine whether the delivery box is empty", "{}")
    system = messages[0][1]

    assert "a definitive FALSE is still a fully successful, completed mission" in system
    assert "does NOT mean the answer was" in system


def test_planner_prompt_teaches_condition_mode_for_a_conditional_branch():
    messages = build_planner_messages(
        "if the door is open, go through it", "{}")
    system = messages[0][1]

    assert 'CONDITION (mode absent, or mode="condition"' in system
    assert "TRUE lets that branch succeed, FALSE fails it" in system
    assert "there genuinely is no 'go through a closed door' outcome to report" in system


def test_planner_prompt_still_reserves_search_for_entity_for_ordinary_search():
    # An ordinary "find X" request is not a yes/no observation question at
    # all -- the pre-existing search_for_entity guidance (unchanged by this
    # round) must still be present and intact.
    messages = build_planner_messages("find the red mug", "{}")
    system = messages[0][1]

    assert "search_for_entity/locate_entity's real target vocabulary is a closed" in system
    assert "Do NOT substitute search_for_entity just because the query happens to name a class" in system


def test_planner_prompt_still_reserves_check_relation_for_already_grounded_entities():
    # Structured relation checks between two ALREADY-NAMED/tracked entities
    # remain check_relation's job -- VisualCheck is for the open-ended/bare-
    # class case only. Both halves of this distinction must survive intact.
    messages = build_planner_messages("is the mug near the sink", "{}")
    system = messages[0][1]

    assert "Both sides must already be named, tracked world-state entities" in system
    assert "check_relation cannot resolve it" in system
    assert "use VisualCheck instead, with a query phrased as the relation itself" in system


def test_planner_prompt_forbids_implicit_goal_spec_for_a_physical_observation_mission():
    messages = build_planner_messages(
        "go check whether the guest is holding a ticket and tell me", "{}")
    system = messages[0][1]

    assert "never fall back to a human-type/implicit goal_spec either" in system


def test_planner_prompt_teaches_approach_then_observe_keeps_the_observation_as_terminal():
    # 2a: an enabling approach (needed only to SEE the subject) precedes the
    # observation, but nothing physical is asked for AFTER the answer -- the
    # observation itself stays the mission's terminal criterion.
    messages = build_planner_messages(
        "go to the visitor and check whether they are wearing a badge, then tell me", "{}")
    system = messages[0][1]

    assert "OBSERVATION-TERMINAL" in system
    assert "enabling actions, if any (e.g. approaching the subject so the camera can see " \
        "them), go BEFORE the VisualCheck, which stays the LAST node" in system
    assert "Never substitute an enabling action's own predicate (entity_approached, " \
        "search_for_entity_completed, ...) as the goal_spec here" in system


# --- P0.6 (z-doc 93): observation followed by a required physical post-action --
# CONFIRMED root cause (z-doc 92's own live acceptance): the prior guidance treated
# VisualCheck(mode="observe") as ALWAYS terminal, which is only correct when no
# physical action is required after the answer -- a live replay of the original
# compound request showed the model instead baking "if TRUE say .../if FALSE say
# ..." into a broken Fallback rather than terminating on the required return.

def test_planner_prompt_makes_visual_check_non_terminal_when_a_physical_return_is_required():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "OBSERVATION FOLLOWED BY A REQUIRED PHYSICAL POST-ACTION" in system
    assert "Here the VisualCheck is NOT the last node" in system


def test_planner_prompt_ties_the_final_goal_to_the_return_actions_own_predicate():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "goal_spec matches THAT FINAL physical action's own predicate instead " \
        "(e.g. entity_approached for an approach_entity(target=Y) ending)" in system
    assert "never visual_check_completed here" in system


def test_planner_prompt_forbids_conditional_say_nodes_for_the_observed_answer():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "Do NOT pre-author the report as BT speech" in system
    assert "'if TRUE say ...' / 'if FALSE say ...'" in system
    assert "a bare say Action essentially always succeeds, so a Fallback built from " \
        "two say nodes never actually branches on the real answer" in system


def test_planner_prompt_true_false_wording_never_becomes_pre_authored_speech():
    # The prohibition is against BAKING the answer into say text at plan-authoring
    # time -- not against discussing TRUE/FALSE in prose. Confirms the actual
    # forbidden pattern (quoted literally) appears only as a NEGATIVE example.
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    idx = system.find("'if TRUE say ...' / 'if FALSE say ...'")
    assert idx != -1
    # It must be introduced as something forbidden, not offered as a valid shape.
    preceding = system[max(0, idx - 80):idx]
    assert "never add a say node" in preceding


def test_planner_prompt_preserves_execution_facts_carrying_the_visual_result_forward():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "already preserved as structured data in the mission's own execution facts" in system
    assert "nothing extra needs to be authored to carry it forward" in system


def test_planner_prompt_a_failed_return_is_not_success_even_with_a_real_visual_answer():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "the mission correctly is NOT successful merely because a visual answer " \
        "already exists" in system
    assert "goal_spec targets the return's own predicate precisely so a failed " \
        "return still fails the mission" in system


def test_planner_prompt_still_allows_a_say_node_for_genuinely_fixed_speech():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "A say node is still fine here for genuinely fixed speech that does not " \
        "depend on the not-yet-executed observation" in system


def test_planner_prompt_unknown_still_pauses_and_never_triggers_the_return():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "Both 2a and 2b: UNKNOWN is unchanged" in system
    assert "never allowed to trigger the physical return as if the observation had " \
        "already succeeded" in system


def test_planner_prompt_condition_mode_still_covers_conditional_physical_branching():
    # A genuinely different shape from 2b: here the branch itself (not a spoken
    # report) depends on the answer -- condition mode, unchanged by this round.
    messages = build_planner_messages(
        "if the parcel has arrived, bring it to the front desk", "{}")
    system = messages[0][1]

    assert 'CONDITION (mode absent, or mode="condition"' in system
    assert "TRUE lets that branch succeed, FALSE fails it" in system


def test_planner_prompt_never_hardcodes_any_demo_alias_for_the_return_case():
    messages = build_planner_messages(
        "check whether the parcel has arrived, then go back to the front desk and "
        "tell them the result", "{}")
    system = messages[0][1]

    assert "bob_live_test" not in system
    assert "potted plant" not in system
    assert "watering" not in system
    assert "zzz_planner_test" not in system


def test_planner_prompt_ties_mode_observe_to_visual_check_completed_as_one_decision():
    # FOUND LIVE 2026-09-14: a real planner call correctly chose goal_spec.
    # predicate=visual_check_completed for an approach-then-observe plan but
    # omitted mode="observe" on the VisualCheck node itself -- silently
    # reintroducing the exact bug this whole contract exists to fix (a plain
    # VisualCheck defaults to condition mode, which never writes the fact
    # visual_check_completed depends on, so it can only ever resolve
    # UNKNOWN). Generalized (z-doc 93) to also cover 2b, where a plain
    # VisualCheck's condition-mode default would silently cancel a required
    # physical return on a real FALSE answer.
    messages = build_planner_messages(
        "check whether the guest is holding a ticket and tell me", "{}")
    system = messages[0][1]

    assert 'mode="observe" is required whenever EITHER' in system
    assert "goal_spec.predicate is visual_check_completed (2a)" in system
    assert "a required physical return (2b)" in system


def test_planner_prompt_never_hardcodes_the_original_acceptance_scenario():
    # z-doc 92's own explicit requirement: the generic capability contract
    # must not name the specific acceptance utterance/entities it was
    # motivated by.
    messages = build_planner_messages("check whether the door is open", "{}")
    system = messages[0][1]

    assert "bob_live_test" not in system
    assert "potted plant" not in system
    assert "watering" not in system


# --- P0.7 (z-doc 94-97, Candidate D, GPT-approved CHANGE APPROVAL): the
# planner now teaches node_id/goal_node_id, and no longer relies solely on
# the model's own free-typed goal_spec.predicate to determine success.

def test_planner_prompt_declares_goal_node_id_as_a_top_level_key():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "Top-level keys: schema, root, goal_spec, goal_node_id." in system


def test_planner_prompt_requires_node_id_on_action_and_visualcheck_nodes():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert (
        "Every Action node and every VisualCheck node must also carry a unique "
        'string "node_id" field'
    ) in system


def test_planner_prompt_explains_goal_node_id_overrides_predicate_trust():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert (
        "goal_spec.predicate is no longer trusted as a free-form string for "
        "deciding mission success"
    ) in system
    assert "compiles the real predicate from whichever node goal_node_id names" in system


def test_planner_prompt_forbids_a_say_node_as_goal_node_id():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "never a say node" in system
    assert (
        "speaking the answer is not the same thing as the obligation the "
        "answer is about"
    ) in system


def test_planner_prompt_forbids_an_earlier_enabling_step_as_goal_node_id():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "never an earlier enabling/orientation step if something else follows it" in system


def test_planner_prompt_allows_null_goal_node_id_for_genuine_ambiguity():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "set goal_node_id to null" in system
    assert "a null goal_node_id is a correct, safe answer" in system
    assert "naming the wrong node is not" in system


def test_planner_prompt_required_output_example_demonstrates_node_id_and_goal_node_id():
    messages = build_planner_messages("say hi", "{}")
    user = json.loads(messages[1][1])
    example = user["required_output_example"]

    assert example["root"]["node_id"] == "n1"
    assert example["goal_node_id"] is None


def test_planner_prompt_goal_node_instruction_never_hardcodes_any_demo_alias():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "bob_live_test" not in system
    assert "obsret_05" not in system
    assert "小W" not in system


# --- Found live 2026-09-15: a real mission wrote {"type": "Action", "skill":
# "VisualCheck"} -- a node TYPE used as a skill name, the same general
# confusion class as the historical "stop"/"NoAction" rejections.

def test_planner_prompt_forbids_node_type_names_as_action_skill_values():
    messages = build_planner_messages("go to the kitchen", "{}")
    system = messages[0][1]

    assert "are BT node TYPES, never skill names" in system
    assert (
        '{"type": "Action", "skill": "VisualCheck", ...} is always wrong'
    ) in system
    for node_type in (
        "Sequence", "Fallback", "Parallel", "Timeout", "Action", "Wait", "Retry",
        "Condition", "GoalCheck", "VisualCheck", "NoAction", "WaitForEvent",
    ):
        assert node_type in system
