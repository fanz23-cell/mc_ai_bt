from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .goal_check import PREDICATE_REGISTRY
from .policy_guard import LOOK_AT_DIRECTIONS, POLICY_ENABLED_SKILLS, PolicyLimits
from .skill_registry import DEFAULT_SKILLS, SkillRegistry
from .visual_check import visual_check_goal_spec

# The complete set of real predicate names Condition/GoalCheck/WaitForEvent/goal_spec
# can actually check -- PREDICATE_REGISTRY covers every one except place_remembered
# (deliberately bespoke-only, §10.5: no world-state fact backs it, so it isn't in the
# generic-fallthrough registry, but it's still a real, checkable predicate name).
_KNOWN_PREDICATE_NAMES = tuple(sorted({*PREDICATE_REGISTRY.keys(), "place_remembered"}))


def _predicate_names_with_producing_skill() -> str:
    # Found live 2026-08-29 (§C2): even after being told the exact list of valid
    # predicate names, the model still confused a *skill* name (check_relation) for
    # the *predicate* it produces (relation_checked) -- a predicate name alone,
    # with no indication of which skill's result_predicates it comes from, wasn't
    # enough to stop that specific mix-up. Pairing each predicate with its skill
    # directly is the same fix §10.5 already made for args_schema, applied here.
    predicate_to_skills: dict[str, list[str]] = {}
    for name, spec in DEFAULT_SKILLS.items():
        for predicate in spec.result_predicates:
            predicate_to_skills.setdefault(predicate, []).append(name)
    parts = []
    for predicate in _KNOWN_PREDICATE_NAMES:
        skills = predicate_to_skills.get(predicate)
        parts.append(f"{predicate} (from {'/'.join(skills)})" if skills else predicate)
    return ", ".join(parts)


PERSON_WORDS = {"person", "people", "someone", "anyone", "人", "某人", "一个人"}


class Planner(Protocol):
    def plan(self, intent_text: str, context_json: str = "") -> str:
        ...


class BootstrapPlanner:
    """Deterministic planner used until the production LLM planner is enabled."""

    def plan(self, intent_text: str, context_json: str = "") -> str:
        text = intent_text.strip()
        lowered = text.lower()
        root = self._route_intent(text, lowered)
        goal_spec = self._goal_spec_for(root, text)
        plan = {
            "schema": "mc_ai_bt.plan.v1",
            "root": root,
            "goal_spec": goal_spec,
            "context_json": context_json or "{}",
        }
        return json.dumps(plan, sort_keys=True, separators=(",", ":"))

    def _route_intent(self, text: str, lowered: str) -> dict:
        steps = _split_compound_intent(lowered)
        if len(steps) > 1:
            children: list[dict] = []
            for step in steps:
                child = self._route_single_intent(step, step)
                if child.get("type") == "Sequence":
                    children.extend(child.get("children", []))
                else:
                    children.append(child)
            return {"type": "Sequence", "children": children}
        return self._route_single_intent(text, lowered)

    def _route_single_intent(self, text: str, lowered: str) -> dict:
        if self._looks_like_come_to_me(lowered):
            return self._sequence(
                self._say("I'm coming to you."),
                {"type": "Action", "skill": "come_to_me", "args": {}, "timeout_sec": 300},
            )

        visual_query = self._extract_visual_query(lowered)
        if visual_query:
            return self._sequence(
                self._say("I'll check what I can see."),
                {
                    "type": "VisualCheck",
                    "check": {"query": visual_query},
                    "timeout_sec": 30,
                },
            )

        look_direction = self._extract_look_at(lowered)
        if look_direction:
            return {
                "type": "Action",
                "skill": "look_at",
                "args": {"direction": look_direction},
                "timeout_sec": 120,
            }

        point_args = self._extract_point_at(lowered)
        if point_args:
            return {
                "type": "Action",
                "skill": "point_at",
                "args": point_args,
                "timeout_sec": 120,
            }

        embodied = self._extract_embodied_skill(lowered)
        if embodied:
            skill, args = embodied
            return {
                "type": "Action",
                "skill": skill,
                "args": args,
                "timeout_sec": 180,
            }

        place = self._extract_place(lowered)
        if place:
            return self._sequence(
                self._say(f"I'm going to {place}."),
                {
                    "type": "Action",
                    "skill": "go_to_place",
                    "args": {"name": place},
                    "timeout_sec": 300,
                },
            )

        move = self._extract_simple_move(lowered)
        if move is not None:
            action, value = move
            return self._sequence(
                self._say(f"Moving {action}."),
                {
                    "type": "Action",
                    "skill": "simple_move",
                    "args": {"action": action, "value": value},
                    "timeout_sec": 120,
                },
            )

        animation = self._extract_animation(lowered)
        if animation:
            return {
                "type": "Action",
                "skill": "play_animation",
                "args": {"animation": animation},
                "timeout_sec": 120,
            }

        return self._say(text)

    @staticmethod
    def _looks_like_come_to_me(lowered: str) -> bool:
        phrases = (
            "come to me",
            "come here",
            "come over",
            "come closer",
            "walk to me",
            "过来",
            "来我这",
            "来我这里",
            "靠近我",
        )
        return any(phrase in lowered for phrase in phrases)

    @staticmethod
    def _extract_place(lowered: str) -> str:
        patterns = (
            r"\bgo to (?P<place>[a-z0-9 _-]+)$",
            r"\bnavigate to (?P<place>[a-z0-9 _-]+)$",
            r"\btake me to (?P<place>[a-z0-9 _-]+)$",
            r"\bdrive to (?P<place>[a-z0-9 _-]+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return _slug(match.group("place"))
        chinese_patterns = (
            r"(?:带我去|带我到|去|到|开到|走到|移动到)(?P<place>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
        )
        for pattern in chinese_patterns:
            match = re.search(pattern, lowered)
            if match:
                return _normalise_place(match.group("place"))
        return ""

    @staticmethod
    def _extract_simple_move(lowered: str) -> tuple[str, float] | None:
        match = re.search(
            r"\b(?:move|drive|go) (?P<action>forward|backward|left|right)"
            r"(?: (?P<value>[0-9]+(?:\.[0-9]+)?))?",
            lowered,
        )
        if not match:
            chinese_move = _extract_chinese_simple_move(lowered)
            if chinese_move is not None:
                return chinese_move
            return None
        action = match.group("action")
        raw_value = match.group("value")
        if raw_value is not None:
            return action, float(raw_value)
        return action, 0.5 if action in {"forward", "backward"} else 90.0

    @staticmethod
    def _extract_animation(lowered: str) -> str:
        if "wave" in lowered or "挥手" in lowered:
            return "wave"
        match = re.search(r"\bplay animation (?P<name>[a-z0-9 _-]+)$", lowered)
        if match:
            return _slug(match.group("name"))
        return ""

    @staticmethod
    def _extract_look_at(lowered: str) -> str:
        direction_pattern = (
            r"front(?:[_ -]?(?:up|down))?|"
            r"left(?:[_ -]?(?:up|down))?|"
            r"right(?:[_ -]?(?:up|down))?|"
            r"up|down|ahead|forward|straight"
        )
        patterns = (
            rf"\blook (?P<direction>{direction_pattern})$",
            rf"\blook at (?P<direction>{direction_pattern})$",
            rf"\bglance (?P<direction>{direction_pattern})$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            direction = _normalize_look_direction(match.group("direction"))
            if direction in LOOK_AT_DIRECTIONS:
                return direction
        for fragment, direction in (
            ("看左边", "left"),
            ("看右边", "right"),
            ("往左看", "left"),
            ("往右看", "right"),
            ("向左看", "left"),
            ("向右看", "right"),
            ("往上看", "front_up"),
            ("向上看", "front_up"),
            ("往下看", "front_down"),
            ("向下看", "front_down"),
            ("向前看", "front"),
            ("往前看", "front"),
        ):
            if fragment in lowered:
                return direction
        return ""

    @staticmethod
    def _extract_point_at(lowered: str) -> dict[str, Any]:
        patterns = (
            r"\bpoint (?:at|to) (?P<target2>[a-z0-9 _-]+?) with (?P<arm2>left|right) (?:hand|arm)$",
            r"\bpoint (?:(?P<arm1>left|right) (?:hand|arm) )?(?:at|to) (?P<target1>[a-z0-9 _-]+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            groups = match.groupdict()
            raw_target = groups.get("target1") or groups.get("target2") or ""
            target = _clean_point_target(raw_target)
            if not target:
                return {}
            args: dict[str, Any] = {"arm": groups.get("arm1") or groups.get("arm2") or "right"}
            if target.startswith("place "):
                args["place"] = _slug(target.removeprefix("place "))
            else:
                args["object"] = target
            return args
        chinese_match = re.search(
            r"(?:用(?P<arm>左|右)(?:手|胳膊|手臂))?(?:指一下|指向|指)(?P<target>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
            lowered,
        )
        if chinese_match:
            target = _normalise_object(chinese_match.group("target"))
            if not target:
                return {}
            arm = "left" if chinese_match.group("arm") == "左" else "right"
            return {"arm": arm, "object": target}
        return {}

    @staticmethod
    def _extract_embodied_skill(lowered: str) -> tuple[str, dict[str, Any]] | None:
        reach = re.search(r"\breach (?:to|for) (?:the )?(?P<object>[a-z0-9 _-]+)$", lowered)
        if reach:
            return "reach_to", {"object": _clean_point_target(reach.group("object"))}
        if "follow me" in lowered or "跟着我" in lowered or "跟我走" in lowered:
            return "follow_entity", {"entity": "interaction_owner"}
        guide = re.search(r"\bguide (?:me|person|someone)? ?(?:to )(?P<place>[a-z0-9 _-]+)$", lowered)
        if guide:
            return "guide_entity_to_place", {"entity": "interaction_owner", "place": _slug(guide.group("place"))}
        chinese_guide = re.search(r"(?:带人去|带我去|引导.*去)(?P<place>[\u4e00-\u9fffA-Za-z0-9 _-]+)$", lowered)
        if chinese_guide:
            return "guide_entity_to_place", {
                "entity": "interaction_owner",
                "place": _normalise_place(chinese_guide.group("place")),
            }
        return None

    @staticmethod
    def _extract_visual_query(lowered: str) -> str:
        patterns = (
            r"\bfind (?P<target>[a-z0-9 _-]+)$",
            r"\blook for (?P<target>[a-z0-9 _-]+)$",
            r"\bcheck for (?P<target>[a-z0-9 _-]+)$",
            r"\bcheck (?:if|whether) (?:you )?(?:can )?see (?P<target>[a-z0-9 _-]+)$",
            r"\bsee if (?:you )?(?:can )?see (?P<target>[a-z0-9 _-]+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            target = _clean_visual_target(match.group("target"))
            if not target:
                return ""
            if target in {"anyone", "someone", "person", "a person", "people"}:
                return "do you see anyone?"
            return f"do you see the {target}?"
        chinese_patterns = (
            r"(?:寻找|找一下|找|看看有没有|检查有没有)(?P<target>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
        )
        for pattern in chinese_patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            target = _normalise_object(match.group("target"))
            if not target:
                return ""
            if target in PERSON_WORDS:
                return "do you see anyone?"
            return f"do you see the {target}?"
        return ""

    @staticmethod
    def _say(text: str) -> dict:
        return {
            "type": "Action",
            "skill": "say",
            "args": {"text": text},
            "timeout_sec": 30,
        }

    @staticmethod
    def _sequence(*children: dict) -> dict:
        return {"type": "Sequence", "children": list(children)}

    @staticmethod
    def _goal_spec_for(root: dict, text: str) -> dict:
        kind, node = _last_goal_relevant_node(root)
        if kind == "action":
            goal = _goal_spec_for_action(node, text)
            if goal:
                return goal
        if kind == "visual":
            check = node.get("check") if isinstance(node.get("check"), dict) else {}
            goal = visual_check_goal_spec(check)
            if goal:
                goal["summary"] = text
                return goal
        return {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
            "summary": text,
        }


class LlmJsonPlanner:
    """Prompt an injected chat model for a constrained JSON BT plan.

    The node does not instantiate this by default. It exists so the production
    LLM caller can be wired without changing the planner/validator contract.
    """

    def __init__(
        self,
        model: Any,
        *,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self._model = model
        self._skills = skill_registry or SkillRegistry()

    def plan(self, intent_text: str, context_json: str = "") -> str:
        messages = build_planner_messages(
            intent_text,
            context_json,
            skill_registry=self._skills,
        )
        raw = self._invoke(messages)
        plan = _extract_json_object(raw)
        plan.setdefault("schema", "mc_ai_bt.plan.v1")
        plan.setdefault("context_json", context_json or "{}")
        return json.dumps(plan, sort_keys=True, separators=(",", ":"))

    def _invoke(self, messages: list[tuple[str, str]]) -> str:
        invoke = getattr(self._model, "invoke", None)
        if not callable(invoke):
            raise TypeError("planner model must expose invoke(messages)")
        result = invoke(messages)
        content = getattr(result, "content", result)
        return str(content)


def build_planner_messages(
    intent_text: str,
    context_json: str,
    *,
    skill_registry: SkillRegistry | None = None,
) -> list[tuple[str, str]]:
    skills = skill_registry or SkillRegistry()
    limits = PolicyLimits()
    skill_lines = []
    for name in skills.names():
        if name not in POLICY_ENABLED_SKILLS:
            continue
        spec = skills.get(name)
        resources = ",".join(spec.resources) if spec.resources else "none"
        # Found live 2026-08-29 building remember_place: this line used to omit
        # args_schema entirely, so the model had no idea what argument keys a new
        # skill actually took and had to guess from the free-text description alone
        # -- it generated a remember_place node missing args.name outright. The
        # handful of skills below with their own hardcoded sentences (look_at,
        # point_at, go_to_place/approach_entity, etc.) exist BECAUSE this line
        # never carried that information; surfacing args_schema here directly, for
        # every skill, is the systemic fix (not one more hardcoded sentence for
        # remember_place specifically) -- it may make some of those sentences
        # redundant over time, but removing them isn't done here since they still
        # carry semantic guidance (e.g. "don't invent a place name") beyond arg shape.
        args_repr = json.dumps(spec.args_schema) if spec.args_schema else "{}"
        skill_lines.append(f"- {spec.name}: resources=[{resources}] args={args_repr} {spec.description}")

    system = "\n".join(
        [
            "You are the mc_ai_bt planner.",
            "Return only one JSON object. Do not include markdown.",
            "The JSON object must use schema mc_ai_bt.plan.v1.",
            "Top-level keys: schema, root, goal_spec, context_json.",
            "root must be a Behavior Tree made only from executable node types.",
            (
                "Executable node types: Sequence, Fallback, Parallel, Timeout, Action, Wait, Retry, "
                "Condition, GoalCheck, VisualCheck, NoAction, WaitForEvent."
            ),
            "Parallel children must be independently safe and must not require overlapping skill resources.",
            "Timeout wraps exactly one child and must set timeout_sec in seconds.",
            "NoAction is only for an explicit already-satisfied/no-op plan with a short reason.",
            (
                "WaitForEvent shape (flat object, no nesting): "
                '{"type": "WaitForEvent", "predicate": "<predicate name string, e.g. participant_ready>", '
                '"args": {<that predicate\'s own args, e.g. role, condition>}, "poll_interval_sec": <0.5-30, '
                'default 2>, "timeout_sec": <required, max 1800>}. predicate is a plain string naming an '
                "existing structured predicate (the same ones Condition/GoalCheck use) -- never a nested "
                "Condition/GoalCheck object. Polls that predicate every poll_interval_sec until TRUE or "
                "timeout_sec elapses; use it instead of Retry(Sequence[Wait,Condition]) for 'wait until X "
                "happens', since that pattern caps out around 1500s and has no real interval control. Only "
                "use predicate names that already exist as a result of some skill's result_predicates "
                "(e.g. robot_at_place, entity_faced, participant_ready) -- never invent one. For 'wait for "
                "a person/role to become ready/raise a hand/etc', do NOT use WaitForEvent at all: call the "
                "wait_for_participant skill directly as a plain Action (args role/condition; set the "
                "Action's own top-level timeout_sec, max 600, for how long to wait, NOT inside args) -- it "
                "already blocks internally up to its own timeout, so wrapping it in WaitForEvent or "
                "inventing a predicate for it is always wrong."
            ),
            (
                "The predicate field on Condition, GoalCheck, WaitForEvent, and a structured goal_spec must "
                "always be one of these exact strings, never invented or guessed, and never a skill name -- "
                "a predicate is what a skill's execution *produces evidence for*, not the skill's own name "
                "(e.g. the check_relation skill produces the relation_checked predicate; use relation_checked "
                "in predicate fields, never check_relation): "
                + _predicate_names_with_producing_skill()
                + ". If none of these fits what you actually need to check, do not invent a new name -- use "
                "a human-type goal_spec (verification mode implicit_conversation) instead of a structured one."
            ),
            (
                # Found live 2026-08-30: a plan called search_for_entity(target=\"woman\"), then checked
                # a Condition with predicate \"visible_people\" -- not in the list above at all. That exact
                # string IS real inside this codebase, just not as a predicate: it is the raw fact-storage
                # key under the world-state \"people\" scope that person_visible's own evaluation reads
                # internally (goal_check.py), so it plausibly *looks* like a valid name without being one.
                # Whether the model meant person_visible or genuinely invented it, checking either one here
                # was the wrong call: search_for_entity's OWN result (search_for_entity_completed) is what
                # actually reflects whether that search just found \"woman\" -- person_visible/visible_people
                # answer a different, unrelated question (is a person visible in the current camera frame
                # right now, independent of any search this plan ran).
                '"visible_people" is never a valid predicate name -- it is an internal fact key, not '
                "something a Condition/GoalCheck/WaitForEvent can check; do not write it. More generally: "
                "immediately after calling a locate-type skill (search_for_entity, locate_entity, get_pose, "
                "wait_for_participant, check_relation), the Condition/GoalCheck that asks \"did that just "
                "find/confirm it\" must check THAT skill's own result predicate from the (from <skill>) list "
                "above (search_for_entity_completed, entity_located, pose_available, participant_ready, "
                "relation_checked respectively) -- not person_visible/object_visible, which answer a "
                "different question (is something visible in the current camera view right now) and are "
                "correct ONLY when nothing in this plan already ran a locate-type skill for the same target."
            ),
            (
                # Found live 2026-08-30: "find/look for a woman in the room" planned
                # search_for_entity(target="woman") -> Condition(entity_located). entity_located's real
                # evidence comes from mc_perception's object/person classifier, which only recognizes a
                # fixed, closed vocabulary of class names (a standard 80-class detector: person, chair,
                # bottle, cup, laptop, ... -- no gender-specific classes, no "box"/"package" class either).
                # Asking it to locate "woman" or "cardboard package box" fails outright, every time,
                # regardless of what is actually in the room -- confirmed live by calling the detector
                # directly with both a real class name (succeeded) and "woman" (rejected, listing its
                # known classes). No amount of the room actually containing a woman fixes this: the
                # target string itself is unrecognizable to the classifier search_for_entity queries.
                "search_for_entity/locate_entity's real target vocabulary is a closed, fixed set of "
                "generic object/creature classes (airplane, apple, backpack, ..., person, ..., zebra -- "
                "the standard 80-class set; person is the only human-related class, with no gender, "
                "age, or clothing distinction). Before using search_for_entity/locate_entity/entity_located "
                "for anything human, ask: does the actual intent reduce to \"is any person present/nearby\", "
                "with no further distinguishing detail the classifier could resolve (a name, a role, "
                "hand-raised, facing the robot)? If so, do not call search_for_entity at all -- go straight "
                "to a Condition/GoalCheck with predicate person_visible (fed by a continuously-running "
                "pose detector, real evidence, not gated on having called any locate skill first). Reserve "
                "search_for_entity/locate_entity for a target that is either one of the fixed class names "
                "verbatim, or a role/description wait_for_participant's own condition argument already "
                "covers (hand_raised, facing_robot) -- never a free-text description like \"woman\" or "
                "\"cardboard package box\" that the classifier was never going to recognize. When the "
                "target is a specific object with no matching class name in that fixed set, there is no "
                "structured predicate that can verify it -- use a human-type goal_spec (verification mode "
                "implicit_conversation) and a VisualCheck/robot_observe-backed description instead of "
                "forcing search_for_entity on a name it cannot possibly resolve."
            ),
            (
                "Never write a say node whose text states the outcome of a Condition/GoalCheck/"
                "VisualCheck/check_relation/locate_entity-family step that has not run yet -- a say node's "
                "text is fixed at planning time, before any check has actually executed, so writing 'Yes, "
                "X is true' there is always a guess, never a verified fact. For a plan whose whole point is "
                "answering a check (e.g. 'is X near Y', 'check whether the door is open'), do not add a "
                "say node for the conclusion at all: leave the goal_spec structured with the real predicate, "
                "and let the mission's own real result reach the user afterward through the existing "
                "terminal-event channel, not a pre-written guess baked into this plan."
            ),
            (
                "VisualCheck must be a concise true/false visual query. It may use structured world-state "
                "evidence or the configured visual checker; UNKNOWN blocks instead of being guessed."
            ),
            "Do not use VisualCheck for unrestricted scene description or hidden-state inference.",
            (
                "look_at may use only direction={front,front_up,front_down,left,left_up,left_down,"
                "right,right_up,right_down} and optional hold seconds."
            ),
            (
                "point_at must use exactly one target form: object/target name, place name, "
                "or explicit x/y/z coordinates; optional arm is left or right."
            ),
            (
                "locate_entity, get_pose and search_for_entity take exactly one arg, target: "
                "the plain name of the object or person to look for, e.g. target=mystery_gadget. "
                "They check current world-state knowledge, not an active physical search; an "
                "UNKNOWN/not-found result means the entity has not been perceived recently, not "
                "that it does not exist."
            ),
            (
                "check_relation takes exactly one arg, relation: two entity/object names joined "
                "by one of near/at/by/next to/close to, e.g. relation='the mug near the sink'. "
                "It only checks entity-to-entity proximity, not containment in a named place."
            ),
            (
                "go_to_place ONLY works for a place already configured by name -- if there is no "
                "known place for what the person means (e.g. 'go to her', 'walk over to the box'), "
                "do not invent a place name. Use approach_entity instead: one arg, target, the "
                "plain name of the person/object to walk to; it navigates to that entity's live "
                "perceived position. It fails if the entity has not been perceived recently -- "
                "that is a real, reportable outcome, not something to retry with a guessed place."
            ),
            (
                "There is no 'stop' skill and no 'stop' action on any skill, including "
                "simple_move -- go_to_place/approach_entity/come_to_me already stop on arrival by "
                "themselves. An intent phrased as '...and stop right next to it' or '...then stop' "
                "is satisfied entirely by the navigation skill alone; do not add a separate step "
                "for the word 'stop' in the intent text."
            ),
            "Do not invent ROS topics, action names, Python code, frames, joint commands or expressions.",
            "For physical actions, use only skills from the skill catalog.",
            (
                "Sequential multi-step plans are allowed, but stay within policy budgets: "
                f"max_total_nodes={limits.max_total_nodes}, "
                f"max_total_actions={limits.max_total_actions}, "
                f"max_physical_actions={limits.max_physical_actions}, "
                f"max_base_actions={limits.max_base_actions}, "
                f"max_body_actions={limits.max_body_actions}, "
                f"max_visual_checks={limits.max_visual_checks}."
            ),
            "For multi-step plans, goal_spec must describe the final desired outcome, not an intermediate step.",
            "A missing or uncertain condition is UNKNOWN, not FALSE.",
            "Skill catalog:",
            "\n".join(skill_lines),
        ]
    )
    user = json.dumps(
        {
            "intent_text": intent_text,
            "context_json": context_json or "{}",
            "required_output_example": {
                "schema": "mc_ai_bt.plan.v1",
                "root": {
                    "type": "Action",
                    "skill": "say",
                    "args": {"text": "I will do that."},
                    "timeout_sec": 30,
                },
                "goal_spec": {
                    "type": "human",
                    "verification": {"mode": "implicit_conversation"},
                    "summary": intent_text,
                },
                "context_json": context_json or "{}",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return [("system", system), ("human", user)]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        value, _end = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("planner output must be a JSON object")
    return value


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _last_goal_relevant_node(node: dict) -> tuple[str, dict]:
    found: tuple[str, dict] = ("", {})
    if node.get("type") == "Action" and node.get("skill") != "say":
        found = ("action", node)
    if node.get("type") == "VisualCheck":
        found = ("visual", node)
    for key in ("children",):
        for child in node.get(key, []) or []:
            child_found = _last_goal_relevant_node(child)
            if child_found[0]:
                found = child_found
    child = node.get("child")
    if isinstance(child, dict):
        child_found = _last_goal_relevant_node(child)
        if child_found[0]:
            found = child_found
    return found


def _goal_spec_for_action(action: dict, text: str) -> dict[str, Any]:
    skill = action.get("skill")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if skill == "go_to_place":
        return {
            "type": "structured",
            "predicate": "robot_at_place",
            "args": {"name": args.get("name", "")},
            "verification": {"mode": "world_state_or_nav_result"},
            "summary": text,
        }
    if skill == "come_to_me":
        return {
            "type": "structured",
            "predicate": "robot_near_interaction_owner",
            "verification": {"mode": "world_state_or_nav_result"},
            "summary": text,
        }
    if skill == "simple_move":
        return {
            "type": "structured",
            "predicate": "relative_motion_completed",
            "args": args,
            "verification": {"mode": "action_result_then_world_state"},
            "summary": text,
        }
    if skill in {"play_animation", "look_at", "point_at"}:
        animation = args.get("animation") or args.get("name") or ""
        if skill == "look_at":
            animation = "look_at"
        elif skill == "point_at":
            animation = "point_at"
        return {
            "type": "structured",
            "predicate": "animation_played",
            "args": {"animation": animation},
            "verification": {"mode": "action_result"},
            "summary": text,
        }
    if skill == "reach_to":
        return {
            "type": "structured",
            "predicate": "reach_completed",
            "args": args,
            "verification": {"mode": "action_result"},
            "summary": text,
        }
    if skill == "follow_entity":
        return {
            "type": "structured",
            "predicate": "entity_following",
            "args": args,
            "verification": {"mode": "world_state_or_action_result"},
            "summary": text,
        }
    if skill == "guide_entity_to_place":
        return {
            "type": "structured",
            "predicate": "entity_at_place",
            "args": args,
            "verification": {"mode": "world_state_or_action_result"},
            "summary": text,
        }
    return {}


def _slug(raw: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", raw.strip().lower()).strip("_")


def _generic_label(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(raw or "").strip())
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return cleaned.replace(" ", "_")
    return _slug(cleaned)


def _split_compound_intent(lowered: str) -> list[str]:
    parts = re.split(r"\s+(?:and\s+then|then)\s+|(?:然后|再)", lowered)
    return [part.strip() for part in parts if part.strip()]


def _clean_visual_target(raw: str) -> str:
    cleaned = re.sub(r"\b(the|a|an|right now|nearby|around|please)\b", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _normalise_object(cleaned)


def _normalize_look_direction(raw: str) -> str:
    cleaned = re.sub(r"[\s-]+", "_", raw.strip().lower())
    aliases = {
        "up": "front_up",
        "down": "front_down",
        "ahead": "front",
        "forward": "front",
        "straight": "front",
    }
    return aliases.get(cleaned, cleaned)


def _clean_point_target(raw: str) -> str:
    cleaned = re.sub(r"\b(the|a|an|nearby|around|please)\b", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _normalise_object(cleaned)


def _normalise_place(raw: str) -> str:
    cleaned = re.sub(r"(这里|那里|那边|这边|一下|吧|请)", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    return _generic_label(cleaned)


def _normalise_object(raw: str) -> str:
    cleaned = re.sub(r"(这里|那里|那边|这边|一下|吧|请)", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    return cleaned


def _extract_chinese_simple_move(lowered: str) -> tuple[str, float] | None:
    value = _extract_first_number(lowered)
    if any(fragment in lowered for fragment in ("左转", "向左转", "往左转", "向左", "往左")):
        return "left", value if value is not None else 90.0
    if any(fragment in lowered for fragment in ("右转", "向右转", "往右转", "向右", "往右")):
        return "right", value if value is not None else 90.0
    if any(fragment in lowered for fragment in ("前进", "向前", "往前")):
        return "forward", value if value is not None else 0.5
    if any(fragment in lowered for fragment in ("后退", "向后", "往后")):
        return "backward", value if value is not None else 0.5
    return None


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"(?P<value>[0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group("value"))
