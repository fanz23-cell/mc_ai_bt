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
