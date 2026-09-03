from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .context_builder import ContextBuilder
from .goal_check import _normalize_alias
from .mission import Mission
from .planner import Planner
from .policy_guard import PHYSICAL_SKILLS, PolicyGuard
from .skill_registry import DEFAULT_SKILLS
from .validator import PlanValidator


@dataclass(frozen=True)
class PlanningResult:
    ok: bool
    stage: str
    message: str
    context_json: str = ""
    plan_json: str = ""
    bt_json: str = ""
    goal_spec_json: str = ""


def _physical_actions_in(node: Any) -> list[dict[str, Any]]:
    """Every {"type": "Action", "skill": <a PHYSICAL_SKILLS member>} node
    anywhere in the tree, walked the same way policy_guard.py's own _walk
    does (Sequence/Fallback/Parallel children, Retry/Timeout child) --
    duplicated here rather than imported/reused because PolicyGuard's walker
    is fused with its stats/error accumulation, not a standalone tree query."""
    found: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return found
    node_type = node.get("type")
    if node_type in {"Sequence", "Fallback", "Parallel"}:
        for child in node.get("children", []) or []:
            found.extend(_physical_actions_in(child))
        return found
    if node_type in {"Retry", "Timeout"}:
        return _physical_actions_in(node.get("child"))
    if node_type == "Action":
        skill = str(node.get("skill") or "")
        if skill in PHYSICAL_SKILLS:
            found.append(node)
    return found


def _all_actions_in(node: Any) -> list[dict[str, Any]]:
    """Every {"type": "Action"} node anywhere in the tree, regardless of
    skill -- broader than _physical_actions_in above, since entity_id
    grounding (below) applies to any skill that accepts an entity_id arg,
    not just physical ones."""
    found: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return found
    node_type = node.get("type")
    if node_type in {"Sequence", "Fallback", "Parallel"}:
        for child in node.get("children", []) or []:
            found.extend(_all_actions_in(child))
        return found
    if node_type in {"Retry", "Timeout"}:
        return _all_actions_in(node.get("child"))
    if node_type == "Action":
        found.append(node)
    return found


def _grounded_entity_ids(context_json: str) -> dict[str, str]:
    """normalized alias -> entity_id, from context_json.caller_context.
    grounded_entities (E.1, mc_voice_pipeline_legacy's RobotGatewayBridge --
    a real, Bridge-constructed, never-Omega-authored identity source, see
    that repo's own _grounded_entities_for_intent). Empty dict if absent or
    malformed -- never raises."""
    try:
        context = json.loads(context_json) if context_json else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    caller_context = context.get("caller_context") if isinstance(context, dict) else None
    if not isinstance(caller_context, dict):
        return {}
    grounded = caller_context.get("grounded_entities")
    if not isinstance(grounded, list):
        return {}
    result: dict[str, str] = {}
    for entry in grounded:
        if not isinstance(entry, dict):
            continue
        alias = str(entry.get("alias") or "")
        entity_id = str(entry.get("entity_id") or "")
        if alias and entity_id:
            result[_normalize_alias(alias)] = entity_id
    return result


def _apply_grounding_normalizer(plan: dict[str, Any], context_json: str) -> str | None:
    """E.1 follow-up (2026-09-03, GPT spec): planner.py's own system prompt
    already INSTRUCTS the model to copy an exact entity_id from
    context_json.caller_context.grounded_entities when the intent
    references a bound alias, and to never invent one -- but instruction
    compliance from a real LLM is not a guarantee, and entity identity is
    exactly the kind of physical-truth boundary A1's evidence policy
    already established must never rest on a generative model's say-so
    alone. This is the deterministic enforcement layer:

    - VALIDATION: any entity_id an Action node actually specifies must be
      one of THIS mission's real grounded_entities -- anything else
      (hallucinated outright, or copied from a different alias than the
      one the intent actually mentioned) is rejected before the plan ever
      reaches PolicyGuard or execution. Returns a non-empty error message
      in this case.
    - INJECTION: any Action node whose `target` (a plain human-readable
      label) exactly matches -- after the SAME NFKC+strip+casefold
      normalization the alias binding itself uses -- one grounded alias,
      and does not already carry an entity_id, gets the correct entity_id
      filled in automatically. This is not a guess: it is copying identity
      data the mission's own real grounding source already established:
      the model correctly identified WHO was meant (used the right target
      text) but simply did not also copy entity_id despite the
      instruction. A target that does not exactly match any grounded
      alias is left alone entirely -- normal class-based behavior,
      unchanged from before this fix.

    FOUND LIVE 2026-09-03 (GPT review): the check below used to be flat
    set-membership against ALL of this mission's grounded entity_ids
    (`entity_id not in grounded.values()`) -- correct with only one alias
    grounded, wrong the moment two or more are: with both "33"->A and
    "22"->B grounded, target="33"+entity_id=B used to PASS, because B is a
    real grounded id, just for the WRONG alias -- exactly the cross-wired
    mistake this stage exists to catch. Now: when `target` itself exactly
    matches a grounded alias, entity_id (if present) must equal THAT
    alias's own entity_id, not merely be somewhere in the grounded set. The
    flat membership check remains as the fallback for a target that is not
    itself a recognizable alias.

    Mutates `plan` in place for the injection case. Returns None when the
    plan needs no rejection (whether or not anything was injected).
    """
    grounded = _grounded_entity_ids(context_json)
    for action in _all_actions_in(plan.get("root")):
        args = action.get("args") if isinstance(action.get("args"), dict) else None
        if args is None:
            continue
        entity_id = str(args.get("entity_id") or "").strip()
        target = str(args.get("target") or "").strip()
        expected_entity_id = grounded.get(_normalize_alias(target)) if target else None

        if entity_id:
            skill = str(action.get("skill") or "")
            if expected_entity_id is not None:
                if entity_id != expected_entity_id:
                    return (
                        f"plan uses entity_id {entity_id!r} on skill {skill!r} whose target "
                        f"{target!r} is grounded to a DIFFERENT entity_id ({expected_entity_id!r}) "
                        "-- entity_id must match the specific alias actually mentioned, never a "
                        "different one from this mission's grounded_entities"
                    )
            elif entity_id not in grounded.values():
                return (
                    f"plan uses entity_id {entity_id!r} on skill {skill!r} that does not match "
                    "any of this mission's grounded_entities -- entity_id must come from "
                    "context_json.caller_context.grounded_entities, never invented"
                )
            continue  # already has a validated entity_id -- nothing to inject
        if expected_entity_id:
            args["entity_id"] = expected_entity_id
    return None


# E.1 follow-up (2026-09-03): remember_person/remember_entity's own
# descriptions already say they locate the target internally (turning to
# look if needed) -- confirmed live, three separate real E.1 test attempts
# through the deployed gpt-4o-mini planner all still planned a redundant
# look_at or search_for_entity Action immediately before remember_person/
# remember_entity despite an added planner.py prompt instruction against
# it (prompting an LLM is never a guarantee, and this one measurably did
# not change its behavior on retry). A first fix only made the goal_spec
# auto-fill below TREAT the redundant action as if absent -- but the BT
# tree itself was untouched, so the redundant action still actually
# EXECUTED (confirmed live: active_node became the redundant look_at, which
# then failed for an unrelated reason -- a robot/sim torso-control
# degradation -- meaning that Action being left in the tree at all was
# itself blocking the mission, not just noise). FOUND LIVE 2026-09-03 (GPT
# review): fixed properly now as a real plan-rewrite (canonicalization)
# instead of a goal-spec-only workaround -- the redundant action is
# actually removed from the tree that gets executed, not merely skipped
# when deriving the goal.
#
# FOUND LIVE 2026-09-03 (GPT review, D0): look_at was REMOVED from this set
# again, one round later. The live trace for exactly this pattern --
# "There is a person right in front of you. Remember them as 33." ->
# look_at(direction="front") + remember_person(target="person", name="33")
# -- showed remember_person's own target is the bare class name "person";
# the ONLY place "front" (the one piece of information that could actually
# disambiguate WHICH person) was ever recorded was look_at's own args, which
# this canonicalizer was silently deleting. look_at's args_schema is
# direction-only with no entity reference, which is exactly why it was
# treated as always-safe to drop -- but "direction" can itself BE a
# reference (a bearing pointing at a specific entity), so "carries no
# entity reference" does not mean "carries no reference semantics". Until a
# real reference-resolution stage (see the C.5/E.1 architecture
# consolidation work) exists to consume that bearing, dropping look_at is a
# real information-loss bug, not a convenience. search_for_entity/
# locate_entity are unaffected -- they carry an explicit target entity
# reference of their own, independently compared against the terminal
# action's target below, so keeping them redundant-eligible loses nothing.
_REDUNDANT_LOCATE_SKILLS = frozenset({"search_for_entity", "locate_entity"})
_SELF_LOCATING_TERMINAL_SKILLS = frozenset({"remember_person", "remember_entity"})


def _redundant_locate_matches_terminal(action: dict[str, Any], terminal_target: str) -> bool:
    """True if `action` (a search_for_entity/locate_entity Action) is safe
    to drop as a redundant duplicate of the terminal remember_person/
    remember_entity action's own internal locate step -- only when its own
    target normalizes to the SAME target the terminal action itself names.
    A search/locate for something else entirely (e.g. search_for_entity
    target="chair" ahead of remember_person target="the person") must never
    be silently dropped -- it may be there for an unrelated reason."""
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    action_target = _normalize_alias(str(args.get("target") or ""))
    return bool(terminal_target) and action_target == terminal_target


def _terminal_self_locating_action(
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """(terminal, redundant_prefix_actions) for a plan whose only
    non-redundant physical action is remember_person/remember_entity, where
    every other physical action is a locate-type skill that ALSO targets
    the same entity as the terminal action (see
    _redundant_locate_matches_terminal) -- None if the pattern does not
    hold (zero or 2+ remember-type actions, a non-locate action alongside
    one, or a locate action that targets something else)."""
    terminal = [a for a in actions if str(a.get("skill") or "") in _SELF_LOCATING_TERMINAL_SKILLS]
    if len(terminal) != 1:
        return None
    term = terminal[0]
    term_args = term.get("args") if isinstance(term.get("args"), dict) else {}
    term_target = _normalize_alias(str(term_args.get("target") or ""))
    others = [a for a in actions if a is not term]
    for a in others:
        if str(a.get("skill") or "") not in _REDUNDANT_LOCATE_SKILLS:
            return None
        if not _redundant_locate_matches_terminal(a, term_target):
            return None
    return term, others


def _without_redundant_locate_actions(node: Any, redundant_ids: set[int]) -> Any | None:
    """A copy of `node` with any Action node whose id() is in redundant_ids
    removed -- dropped from a Sequence/Fallback/Parallel's children list
    directly, or by dropping the whole Retry/Timeout wrapper when its sole
    child is one of them. Returns None when `node` itself was removed."""
    if not isinstance(node, dict):
        return node
    node_type = node.get("type")
    if node_type == "Action":
        return None if id(node) in redundant_ids else node
    if node_type in {"Sequence", "Fallback", "Parallel"}:
        new_children = []
        for child in node.get("children", []) or []:
            rewritten = _without_redundant_locate_actions(child, redundant_ids)
            if rewritten is not None:
                new_children.append(rewritten)
        new_node = dict(node)
        new_node["children"] = new_children
        return new_node
    if node_type in {"Retry", "Timeout"}:
        rewritten_child = _without_redundant_locate_actions(node.get("child"), redundant_ids)
        if rewritten_child is None:
            return None
        new_node = dict(node)
        new_node["child"] = rewritten_child
        return new_node
    return node


def _canonicalize_redundant_locate_prefix(plan: dict[str, Any]) -> None:
    """Actually remove a redundant look_at/search_for_entity/locate_entity
    Action from the executable BT tree when it immediately duplicates work
    remember_person/remember_entity already does internally for the SAME
    target (see _terminal_self_locating_action) -- not just skip it when
    deriving goal_spec (that alone left it in the tree to actually run,
    which is what a real live E.1 attempt showed blocking the mission).
    Mutates plan["root"] in place. A no-op when the pattern does not match
    (including a locate action that targets something else -- never
    touched)."""
    match = _terminal_self_locating_action(_physical_actions_in(plan.get("root")))
    if match is None:
        return
    _terminal, redundant = match
    if not redundant:
        return
    redundant_ids = {id(a) for a in redundant}
    plan["root"] = _without_redundant_locate_actions(plan.get("root"), redundant_ids)


def _apply_deterministic_goal_spec(plan: dict[str, Any]) -> None:
    """E.1 follow-up (2026-09-02, GPT spec): the LLM/bootstrap planner does
    not always produce a structured goal_spec for a mission containing a
    physical skill, and PolicyGuard correctly rejects "implicit human
    verification + a physical skill" outright (see its own
    _check_goal_alignment comment) rather than silently accepting a
    meaningless success criterion -- but that means a real command like "go
    check on 33" could never reach execution at all, purely because the
    planner didn't happen to also emit a matching goal_spec. This closes the
    gap for the one case that can be filled in WITHOUT guessing: exactly one
    physical Action in the whole plan, whose skill declares EXACTLY one
    result_predicate (skill_registry.py) -- there is only one possible
    correct predicate, so filling it in is not a guess, it's the only
    consistent reading. The Action's own args are copied verbatim into the
    goal_spec's args (goal_check.py's existing predicate handlers already
    read the same argument names a skill's own args_schema uses -- e.g.
    robot_at_place reads args.name, exactly what go_to_place's own args
    already carry). Anything less clean -- zero or 2+ physical actions, or a
    skill whose result_predicates has 0 or 2+ entries -- is left alone,
    falling straight through to PolicyGuard's existing rejection, exactly as
    before this fix: replanning/rejecting beats guessing wrong. (2026-09-03:
    the "N redundant locate actions + 1 self-locating remember" case used to
    be handled here too, as a goal-spec-only special case; it is now
    _canonicalize_redundant_locate_prefix, called earlier in plan(), which
    actually strips those actions from the tree -- so by the time this
    function runs, a matching plan already has exactly one physical Action
    and needs no special case here at all.)"""
    goal_spec = plan.get("goal_spec")
    if not isinstance(goal_spec, dict):
        return
    verification = goal_spec.get("verification") if isinstance(goal_spec.get("verification"), dict) else {}
    is_implicit = goal_spec.get("type") == "human" or str(verification.get("mode") or "") == "implicit_conversation"
    if not is_implicit:
        return

    actions = _physical_actions_in(plan.get("root"))
    if len(actions) != 1:
        return
    action = actions[0]
    skill = str(action.get("skill") or "")
    spec = DEFAULT_SKILLS.get(skill)
    if spec is None or len(spec.result_predicates) != 1:
        return
    predicate = spec.result_predicates[0]
    action_args = action.get("args") if isinstance(action.get("args"), dict) else {}

    plan["goal_spec"] = {
        "type": "structured",
        "predicate": predicate,
        "args": dict(action_args),
        "verification": {"mode": "world_state"},
    }


class PlanningPipeline:
    def __init__(
        self,
        *,
        planner: Planner,
        context_builder: ContextBuilder,
        validator: PlanValidator,
        policy_guard: PolicyGuard,
    ) -> None:
        self._planner = planner
        self._context_builder = context_builder
        self._validator = validator
        self._policy_guard = policy_guard

    def plan(
        self,
        mission: Mission,
        missions: tuple[Mission, ...],
    ) -> PlanningResult:
        context_json = self._context_builder.build_json(mission, missions)
        try:
            plan_json = self._planner.plan(mission.intent_text, context_json)
        except Exception as exc:  # noqa: BLE001
            return PlanningResult(
                False,
                "planner",
                f"planner failed: {type(exc).__name__}: {exc}",
                context_json=context_json,
            )

        validation = self._validator.validate_json(plan_json)
        if not validation.ok:
            return PlanningResult(
                False,
                "validator",
                "; ".join(validation.errors),
                context_json=context_json,
                plan_json=plan_json,
            )

        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return PlanningResult(
                False,
                "validator",
                f"invalid json after validation: {exc}",
                context_json=context_json,
                plan_json=plan_json,
            )

        _canonicalize_redundant_locate_prefix(plan)

        grounding_error = _apply_grounding_normalizer(plan, context_json)
        if grounding_error:
            return PlanningResult(
                False,
                "grounding",
                grounding_error,
                context_json=context_json,
                plan_json=plan_json,
            )

        _apply_deterministic_goal_spec(plan)

        policy = self._policy_guard.check(plan)
        if not policy.ok:
            return PlanningResult(
                False,
                "policy",
                "; ".join(policy.errors),
                context_json=context_json,
                plan_json=plan_json,
            )

        return PlanningResult(
            True,
            "done",
            "planned",
            context_json=context_json,
            plan_json=plan_json,
            bt_json=json.dumps(plan["root"], sort_keys=True, separators=(",", ":")),
            goal_spec_json=json.dumps(
                plan["goal_spec"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
