from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .context_builder import ContextBuilder
from .goal_check import _normalize_alias
from .mission import Mission
from .planner import Planner
from .policy_guard import PHYSICAL_SKILLS, PolicyGuard
from .reference_extraction import has_explicit_look_instruction
from .skill_registry import DEFAULT_SKILLS, internally_resolved_predicates
from .validator import KNOWN_PREDICATE_NAMES, PlanValidator


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


_SEARCH_THEN_APPROACH_LOCATE_SKILLS = ("search_for_entity", "locate_entity")


def _sequential_physical_actions_in(node: Any) -> list[dict[str, Any]]:
    """Like _physical_actions_in, but does NOT recurse into Fallback/Parallel
    children -- those represent alternative or concurrent branches, not an
    unconditional "this happens, then that happens" sequence. Used only by
    _fill_search_then_approach_goal_spec below to confirm its two physical
    actions are genuinely, unconditionally sequenced before treating the
    second action's success as depending on the first having just run --
    if either action turns out to live inside a Fallback/Parallel instead,
    this collector simply will not find it, and the fill is correctly
    skipped (falls through to PolicyGuard's existing rejection)."""
    found: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return found
    node_type = node.get("type")
    if node_type == "Sequence":
        for child in node.get("children", []) or []:
            found.extend(_sequential_physical_actions_in(child))
        return found
    if node_type in {"Retry", "Timeout"}:
        return _sequential_physical_actions_in(node.get("child"))
    if node_type == "Action":
        skill = str(node.get("skill") or "")
        if skill in PHYSICAL_SKILLS:
            found.append(node)
    return found


def _fill_search_then_approach_goal_spec(plan: dict[str, Any]) -> bool:
    """FOUND LIVE 2026-09-09: full-session log analysis of "find X"/"go to X"
    missions for an object or person with no bound alias -- exactly the
    search_for_entity/locate_entity + approach_entity shape
    _apply_deterministic_goal_spec's own docstring already names as one it
    deliberately leaves alone (2 physical actions) -- showed the planner
    consistently failed to supply a structured goal_spec on its own, and a
    prompt-only fix (teaching the model the correct structured goal_spec to
    write) was NOT reliably followed live. This is the SAME "instruction
    compliance is not a guarantee" lesson _apply_grounding_normalizer's own
    docstring already drew from an analogous case -- the fix is a second
    deterministic enforcement/fill layer, not a stronger prompt.

    Unlike the general 2+-physical-actions case, THIS specific shape is not
    a guess: when the plan's only two physical actions are (1) a locate-type
    search for a target and (2) approach_entity for the SAME target (by
    exact normalized label match) with no entity_id of its own, the mission
    succeeds if and only if that approach actually happened -- there is
    exactly one correct predicate, entity_approached, and its own
    execution-time check (decision_broker.py's _check_entity_approached)
    already works from `target` alone, no entity_id required, resolving
    against whatever was most recently perceived matching that label --
    precisely the just-completed search's own result. Declines (returns
    False, changes nothing) for anything less clean: not exactly two
    physical actions, either skill name wrong, targets missing or not an
    exact match, an entity_id already present on the approach (that path is
    _apply_grounding_normalizer's job, not this one's), or the two actions
    not genuinely, unconditionally sequenced (see
    _sequential_physical_actions_in above) -- every one of those falls
    through to PolicyGuard's existing rejection, exactly as before this
    function existed."""
    sequential = _sequential_physical_actions_in(plan.get("root"))
    if len(sequential) != 2:
        return False
    first, second = sequential
    if str(first.get("skill") or "") not in _SEARCH_THEN_APPROACH_LOCATE_SKILLS:
        return False
    if str(second.get("skill") or "") != "approach_entity":
        return False
    second_args = second.get("args") if isinstance(second.get("args"), dict) else {}
    if str(second_args.get("entity_id") or "").strip():
        return False
    first_args = first.get("args") if isinstance(first.get("args"), dict) else {}
    first_target = str(first_args.get("target") or "").strip()
    second_target = str(second_args.get("target") or "").strip()
    if not first_target or not second_target:
        return False
    if _normalize_alias(first_target) != _normalize_alias(second_target):
        return False

    plan["goal_spec"] = {
        "type": "structured",
        "predicate": "entity_approached",
        "args": {"target": second_target},
        "verification": {"mode": "world_state"},
    }
    return True


# 2026-09-13 (see z——doc/FINAL_100_PERCENT_DELIVERY_2026-09-12/
# 62_CHANGE_APPROVAL_PLANNER_GOALSPEC_ENABLING_SEQUENCE.md): every skill safe to
# treat as a purely preparatory step ahead of a plan's real terminal action --
# orientation/gaze or perception acquisition, neither of which has any
# independent "mission accomplished" meaning of its own once something else
# follows. Deliberately excludes any skill with its own substantive,
# independently-plausible-as-the-whole-mission completion meaning (reach_to,
# retract, hold_pose, touch_entity, oscillate, remember_*, scan_room,
# simple_move, point_at, play_animation, ...) -- those are never safe to
# demote to "merely enabling" just because something happens to follow them in
# the same plan. `go_to_place` is deliberately NOT here despite one real
# historical "go there, then scan the room" case this would have covered:
# unlike approach_entity (only ever a means to reach an entity the mission
# already names), a bare go_to_place is itself frequently the ENTIRE point of
# a real mission ("go to the kitchen") -- two pre-existing tests
# (test_planning_pipeline_does_not_guess_with_two_physical_actions,
# test_planning_pipeline_does_not_guess_when_remember_entity_follows_a_non_locate_action)
# already encode this exact design decision deliberately, predating this
# change; respecting it here rather than quietly overriding it for one
# historical occurrence. A sequence containing go_to_place (or any other
# excluded skill) anywhere but last still falls through to PolicyGuard's
# existing rejection, unchanged.
_ENABLING_SEQUENCE_SKILLS = frozenset({
    "look_at", "look_at_static", "track_with_gaze",
    "face_entity", "track_entity", "track_frame",
    "search_for_entity", "locate_entity", "get_pose",
    "approach_entity",
})


def _fill_enabling_sequence_goal_spec(plan: dict[str, Any]) -> bool:
    """Real, live-confirmed gap beyond _fill_search_then_approach_goal_spec's
    own narrow shape (see this doc's own CHANGE APPROVAL for the historical
    evidence, drawn directly from context/ai_bt_missions.jsonl's real rejected-
    mission journal, not invented cases): a real "look_at" then "reach_to" plan
    ("Reach your hand toward the person in front of you") and a real "look_at"
    then "remember_person" plan ("Find the person in front of you and remember
    them as 33") both got the identical implicit-verification rejection this
    whole deterministic-fill mechanism exists to close, and neither is the
    2-action approach_entity-specific shape the function above covers.

    Generalizes the same "not a guess" reasoning to an UNCONDITIONAL sequence
    of any length whose LAST physical action has exactly one result_predicate
    (the ORIGINAL single-action fill's own criterion, just no longer requiring
    every OTHER action to be absent) and every action BEFORE it is drawn from
    _ENABLING_SEQUENCE_SKILLS above -- skills with no competing claim to being
    the mission's real point. The terminal action's own predicate is the one
    unambiguous success criterion in exactly the same sense the single-action
    fill already established: there is only one physical action whose result
    the mission could sensibly be judged by, the others are there to put the
    robot/camera in the right place/orientation first.

    Same-target safety check as the function above, generalized: for each
    enabling action that itself declares a non-empty `args.target`, it must
    normalized-match the terminal action's own `args.target` (empty-target
    enabling actions, e.g. go_to_place's `place`-only args or look_at's
    `direction`-only args, have nothing to compare and are unaffected). This
    is what keeps a genuinely suspicious plan -- e.g. searching for one thing
    and then approaching an unrelated other one -- falling through to
    PolicyGuard's existing rejection exactly as before, not silently guessing
    which target the mission actually meant."""
    sequential = _sequential_physical_actions_in(plan.get("root"))
    if len(sequential) < 2:
        return False
    enabling, terminal = sequential[:-1], sequential[-1]
    for action in enabling:
        if str(action.get("skill") or "") not in _ENABLING_SEQUENCE_SKILLS:
            return False

    terminal_skill = str(terminal.get("skill") or "")
    spec = DEFAULT_SKILLS.get(terminal_skill)
    if spec is None or len(spec.result_predicates) != 1:
        return False

    terminal_args = terminal.get("args") if isinstance(terminal.get("args"), dict) else {}
    terminal_target = str(terminal_args.get("target") or "").strip()
    for action in enabling:
        action_args = action.get("args") if isinstance(action.get("args"), dict) else {}
        enabling_target = str(action_args.get("target") or "").strip()
        if enabling_target and _normalize_alias(enabling_target) != _normalize_alias(terminal_target):
            return False

    plan["goal_spec"] = {
        "type": "structured",
        "predicate": spec.result_predicates[0],
        "args": dict(terminal_args),
        "verification": {"mode": "world_state"},
    }
    return True


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


def _grounded_entity_ids(context_json: str) -> dict[str, dict[str, str]]:
    """normalized alias -> {"entity_id", "grounding_state", "semantic_entity_id"},
    from context_json.caller_context.grounded_entities (E.1,
    mc_voice_pipeline_legacy's RobotGatewayBridge -- a real, Bridge-
    constructed, never-Omega-authored identity source, see that repo's own
    _grounded_entities_for_intent). Empty dict if absent or malformed --
    never raises.

    FOUND LIVE 2026-09-03 (GPT review, Gate-1): this used to require a
    non-empty entity_id just to include an alias at all -- which silently
    dropped every UNRESOLVED alias (known, but its underlying live track
    has gone STALE -- see mc_world_state/entity_identity.py's grounding_state)
    from the map entirely. That made _apply_grounding_normalizer below fall
    through to its flat-membership fallback for an UNRESOLVED alias's
    target, which could then accept a cross-wired entity_id that happens to
    belong to some OTHER, unrelated grounded alias -- the exact mistake
    this stage exists to catch, reopened in a new form. Every bound alias
    is included now regardless of grounding_state; entity_id is legitimately
    "" for an UNRESOLVED one."""
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
    result: dict[str, dict[str, str]] = {}
    for entry in grounded:
        if not isinstance(entry, dict):
            continue
        alias = str(entry.get("alias") or "")
        if not alias:
            continue
        result[_normalize_alias(alias)] = {
            "entity_id": str(entry.get("entity_id") or ""),
            "grounding_state": str(entry.get("grounding_state") or "UNRESOLVED"),
            "semantic_entity_id": str(entry.get("semantic_entity_id") or ""),
        }
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

    FOUND LIVE 2026-09-03 (GPT review, round 1): the check below used to be
    flat set-membership against ALL of this mission's grounded entity_ids
    (`entity_id not in grounded.values()`) -- correct with only one alias
    grounded, wrong the moment two or more are: with both "33"->A and
    "22"->B grounded, target="33"+entity_id=B used to PASS, because B is a
    real grounded id, just for the WRONG alias -- exactly the cross-wired
    mistake this stage exists to catch. Fixed: when `target` itself exactly
    matches a grounded alias, entity_id (if present) must equal THAT
    alias's own entity_id, not merely be somewhere in the grounded set.

    FOUND LIVE 2026-09-03 (GPT review, Gate-1): that first fix was itself
    reopened by _grounded_entity_ids no longer dropping UNRESOLVED aliases
    -- a target matching an UNRESOLVED alias must reject ANY entity_id
    outright (there is no live entity to reference, so entity_id cannot be
    legitimately anything), not silently fall through to the flat
    membership fallback, which would again accept a cross-wired id
    belonging to some OTHER, RESOLVED alias.

    Mutates `plan` in place for the injection case. Returns None when the
    plan needs no rejection (whether or not anything was injected).
    """
    grounded = _grounded_entity_ids(context_json)
    resolved_ids = {info["entity_id"] for info in grounded.values() if info["entity_id"]}
    for action in _all_actions_in(plan.get("root")):
        args = action.get("args") if isinstance(action.get("args"), dict) else None
        if args is None:
            continue
        entity_id = str(args.get("entity_id") or "").strip()
        target = str(args.get("target") or "").strip()
        matched = grounded.get(_normalize_alias(target)) if target else None

        if entity_id:
            skill = str(action.get("skill") or "")
            if matched is not None:
                if matched["grounding_state"] != "RESOLVED":
                    return (
                        f"plan uses entity_id {entity_id!r} on skill {skill!r} whose target "
                        f"{target!r} is a known alias but currently UNRESOLVED (no live entity "
                        "to reference right now) -- entity_id must never be supplied for an "
                        "unresolved alias"
                    )
                if entity_id != matched["entity_id"]:
                    return (
                        f"plan uses entity_id {entity_id!r} on skill {skill!r} whose target "
                        f"{target!r} is grounded to a DIFFERENT entity_id ({matched['entity_id']!r}) "
                        "-- entity_id must match the specific alias actually mentioned, never a "
                        "different one from this mission's grounded_entities"
                    )
            elif entity_id not in resolved_ids:
                return (
                    f"plan uses entity_id {entity_id!r} on skill {skill!r} that does not match "
                    "any of this mission's grounded_entities -- entity_id must come from "
                    "context_json.caller_context.grounded_entities, never invented"
                )
            continue  # already has a validated entity_id -- nothing to inject
        if matched is not None and matched["grounding_state"] == "RESOLVED" and matched["entity_id"]:
            args["entity_id"] = matched["entity_id"]
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
#
# D1 (2026-09-03, GPT review): these used to be hand-written literal sets --
# exactly the "one more place to forget" pattern policy_guard.py's own
# PHYSICAL_SKILLS/PREDICATE_TO_SKILLS derivation comment already warns
# about (that file was hand-copying skill-name sets independently until a
# real production bug traced back to one drifting out of sync). Computed
# from skill_registry.py's own subsumes_locate_skills field instead: adding
# a new self-locating skill now means declaring it there, once -- these two
# sets recompute automatically, the same pattern PHYSICAL_SKILLS uses.
_SELF_LOCATING_TERMINAL_SKILLS = frozenset(
    name for name, spec in DEFAULT_SKILLS.items() if spec.subsumes_locate_skills
)
_REDUNDANT_LOCATE_SKILLS = frozenset(
    skill
    for spec in DEFAULT_SKILLS.values()
    for skill in spec.subsumes_locate_skills
)


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


# Identity/Grounding foundation finalization (2026-09-03, GPT re-review):
# Gate-1.1's structured contract (a plan-level reference_constraints array
# the PLANNER itself authored, deterministically checked) was itself found
# to still be trust-boundary-open, by direct counter-example: a planner
# that mis-attributes "on your left" to a person when it actually describes
# a chair would write internally-consistent fields (entity_class="person",
# reference_frame="robot", source_span="on your left" -- each individually
# a TRUE statement about the words) that pass every check this stage could
# run, because the checks were verifying the model's OWN claims against
# each other and against the raw utterance, never against an INDEPENDENT
# judgment of what those words actually modify. Same failure mode as
# GroundingNormalizer would have if the LLM were allowed to both invent AND
# self-certify an entity_id.
#
# The fix: extraction moves entirely out of the planner and into
# reference_extraction.py's deterministic recognizer, run BEFORE the
# planner (see context_builder.py). The planner may only REFERENCE a
# constraint that module already produced (by constraint_id, via
# context_json.reference_constraints) -- it is never handed the pen to
# author fields itself. This stage's job shrinks accordingly: reject a
# directly-set `relation` outright (the planner must never write one), and
# look up + apply a referenced constraint FROM CONTEXT -- plan.
# reference_constraints (if a confused planner still emits one) is simply
# never consulted at all.
_VALID_RELATIONS = frozenset({"front", "left", "right", "nearest"})


def _trusted_reference_constraints(context_json: str) -> dict[str, dict[str, Any]]:
    """context_json.reference_constraints, keyed by constraint_id --
    reference_extraction.py's own deterministic output, computed from
    intent_text BEFORE the planner ever ran (see context_builder.py). This
    is the ONLY source _apply_reference_constraint_guard trusts; a
    same-shaped array the planner might put in its OWN plan.
    reference_constraints output is never read. Malformed entries (not a
    dict, missing constraint_id) are silently dropped -- never raises."""
    try:
        context = json.loads(context_json) if context_json else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    constraints = context.get("reference_constraints") if isinstance(context, dict) else None
    if not isinstance(constraints, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in constraints:
        if not isinstance(entry, dict):
            continue
        constraint_id = str(entry.get("constraint_id") or "").strip()
        if not constraint_id:
            continue
        out[constraint_id] = entry
    return out


def _validate_reference_constraint(
    constraint: dict[str, Any], *, intent_text: str,
) -> tuple[str, str]:
    """(relation, error) -- relation is "" and error is non-empty when the
    constraint fails any check; otherwise relation is one of
    front/left/right/nearest and error is "".

    These constraints come from a TRUSTED, deterministic extractor now
    (reference_extraction.py), not the planner -- so this is defense in
    depth, not the primary trust boundary (that moved to the extractor's
    own narrow-grammar design). Still checked, in case context_json ever
    carries a stale or hand-crafted entry: reference_frame must be "robot"
    (ResolveEntityReference's geometry has only ever understood
    robot-relative bearings), entity_class must be non-empty, and
    source_span must be a real, near-verbatim substring of this mission's
    own intent_text.

    CHANGE APPROVAL 1 (this run): entity_class used to be required to be
    exactly "person" -- the only class reference_extraction.py produced at
    the time, per an explicit prior decision that there was "no legitimate
    reason" for a non-person constraint. That round's own real, live
    evidence (two simultaneously visible potted plants, a real "the
    nearest plant" utterance remember_entity correctly refused to guess
    at) directly contradicted that rationale, so reference_extraction.py
    now also produces a bounded, tight-adjacency-matched OBJECT noun as
    entity_class (see its own _OBJECT_RELATION_PATTERNS). Accepting any
    non-empty entity_class here is still safe: this field is validate-time
    only (never propagated into the actual bind -- remember_entity/
    remember_person each independently re-derive entity_class from a real,
    freshly-observed perception fact before ever calling
    ResolveEntityReference), and source_span below still independently
    re-verifies the words came from the user's own real intent_text either
    way.
    """
    relation = str(constraint.get("relation") or "").strip().lower()
    if relation not in _VALID_RELATIONS:
        return "", f"relation {relation!r} is not one of front/left/right/nearest"
    reference_frame = str(constraint.get("reference_frame") or "").strip().lower()
    if reference_frame != "robot":
        return "", (
            f"reference_frame {reference_frame!r} is not \"robot\" -- ResolveEntityReference "
            "only understands robot-relative bearings, never a relation to some other entity"
        )
    entity_class = str(constraint.get("entity_class") or "").strip().lower()
    if not entity_class or len(entity_class) > 64:
        return "", f"entity_class {entity_class!r} must be non-empty and <= 64 chars"
    source_span = str(constraint.get("source_span") or "").strip()
    if not source_span or _normalize_alias(source_span) not in _normalize_alias(intent_text):
        return "", (
            f"source_span {source_span!r} was not found in this mission's own intent_text -- "
            "a reference_constraint must trace to the user's real words, never invented"
        )
    return relation, ""


def _action_alias_or_name(args: dict[str, Any]) -> str:
    """remember_entity uses `alias`, remember_person uses `name` -- the
    same "what to call them" argument under two different key names (see
    skill_registry.py's own args_schema for both)."""
    return str(args.get("alias") or args.get("name") or "")


def _append_missing_required_producers(
    plan: dict[str, Any], *, context_json: str, intent_text: str,
) -> None:
    """CHANGE APPROVAL A (2026-09-05, GPT-approved): a trusted, pre-planner
    reference_constraint with a non-empty bind_alias already tells the
    machine -- deterministically, before Stage-2 ever ran -- that this
    mission requires a specific alias bound to a person. If the candidate
    plan's own physical actions contain no producer of that binding at all
    (Stage-2 can and does sometimes omit it entirely; see the E1 root-cause
    audit trail), this appends one, built only from data the constraint
    itself already carries: bind_alias, entity_class, and the constraint's
    own id. This is the same "attach what the machine already owns,
    deterministically, rather than argue with the model again" reasoning
    _inject_owned_reference_constraint below already uses for a single
    field -- generalized one level, from completing an existing action to
    supplying a missing one.

    ABSOLUTE SAFETY RULE, load-bearing: this function only ever APPENDS. It
    never deletes, prunes, reorders, or rewrites any pre-existing Action --
    not even ones that look like an obviously-redundant misfire (e.g. a
    search_for_entity/approach_entity pair in front of no terminal at all).
    An earlier candidate design also removed existing actions that shared
    the missing producer's own `target` field, on the theory that they were
    the model's wrong substitute for the same request; adversarial review
    (see the closure audit) proved that theory unfalsifiable with this
    registry's metadata -- SkillSpec has no precondition/delete-effect
    model, only result_predicates (an add-list), so nothing here can
    formally distinguish "the model's wrong guess" from "a second, genuine
    request that happens to share a target class" -- and found concrete
    cases where deleting silently discarded a real user request. That
    design was rejected. This function does not attempt it: every
    pre-existing Action, redundant-looking or not, survives untouched, and
    it is left entirely to the pre-existing, unmodified downstream stages
    (the canonicalizers below, PolicyGuard) to accept or reject the
    resulting plan exactly as they would any planner-authored one.

    Every still-missing, still-valid bind_alias gets its own appended
    producer -- not just the first one a constraint dict happens to
    iterate to. Stopping at the first match was found, in review, to
    silently leave a second genuinely-distinct alias with no producer at
    all while still reporting the mission as planned.

    Producer selection is registry-derived, not invented. Today
    reference_extraction.py's own extractor hardcodes entity_class to
    "person" (its only literal value, see that module), and
    _validate_reference_constraint below rejects any constraint whose
    entity_class is not "person" -- so "person" is the only entity_class a
    trusted constraint can ever carry. remember_person is the one skill in
    DEFAULT_SKILLS whose own description names exactly that class ("Bind a
    name to a person the robot can currently see..."). For any other
    entity_class -- unreachable today, since the check above already
    excludes it -- this function does nothing: remember_entity ALSO
    declares subsumes_locate_skills and could plausibly apply, and nothing
    in the registry disambiguates the two for a class neither extractor nor
    validator can ever actually produce, so this deliberately does not
    guess rather than invent a preference the registry does not state.

    Mutates `plan` in place for the append case; a no-op otherwise. Runs
    before every other stage below so the rest of the pipeline (redundant-
    locate canonicalization, reference-constraint injection/guard,
    grounding normalization, deterministic goal_spec fill, PolicyGuard)
    treats an appended producer exactly as it would treat one the model
    wrote itself."""
    constraints = _trusted_reference_constraints(context_json)
    if not constraints:
        return

    present_aliases: set[str] = set()
    for action in _all_actions_in(plan.get("root")):
        if str(action.get("skill") or "") not in _SELF_LOCATING_TERMINAL_SKILLS:
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        alias = _normalize_alias(_action_alias_or_name(args))
        if alias:
            present_aliases.add(alias)

    to_append: list[dict[str, Any]] = []
    handled_aliases: set[str] = set()
    for constraint_id, constraint in constraints.items():
        bind_alias = str(constraint.get("bind_alias") or "").strip()
        if not bind_alias:
            continue
        normalized_alias = _normalize_alias(bind_alias)
        if normalized_alias in present_aliases or normalized_alias in handled_aliases:
            continue
        _, error = _validate_reference_constraint(constraint, intent_text=intent_text)
        if error:
            continue
        entity_class = str(constraint.get("entity_class") or "").strip().lower()
        if entity_class != "person":
            continue  # no registry-unambiguous producer for this class; see docstring
        handled_aliases.add(normalized_alias)
        to_append.append({
            "type": "Action",
            "skill": "remember_person",
            "args": {
                "name": bind_alias,
                "target": entity_class,
                "reference_constraint_id": constraint_id,
            },
        })

    if not to_append:
        return
    root = plan.get("root")
    if not isinstance(root, dict):
        return
    if root.get("type") == "Sequence":
        children = root.get("children")
        if not isinstance(children, list):
            children = []
            root["children"] = children
        children.extend(to_append)
    else:
        plan["root"] = {"type": "Sequence", "children": [root, *to_append]}


def _inject_owned_reference_constraint(
    plan: dict[str, Any], *, context_json: str, intent_text: str,
) -> None:
    """FOUND LIVE 2026-09-05 (E1 TURN1, sixth real attempt): with everything
    else fixed the plan finally reached the robot, as a single clean action --
    and was refused by the skill itself:

        remember_person blocked/failed: remember_entity: 2 person candidates
        currently active

    Correctly refused: two people really were visible, and the plan carried no
    way to say which. But the user HAD said which -- "离你最近的人", the nearest
    one -- and reference_extraction.py had already turned that into a real
    constraint sitting in context_json. The planner simply never referenced it,
    so the one piece of disambiguating information the user actually gave was
    dropped between extraction and execution.

    It was described to the model only conditionally ("If one of its entries
    genuinely fits", "only when 2+ same-class candidates could otherwise be
    meant"), while every wrong path was described plainly -- and no prompt text
    ever said that "nearest" is itself a qualifying spatial relation. Rather
    than argue with the model again, this attaches the constraint the same way
    _apply_grounding_normalizer already attaches an entity_id the model was
    told to copy and did not: deterministically, from data the machine already
    owns.

    Safe precisely because it reuses the ownership test the guard below already
    enforces in the other direction. The guard says a constraint may only be
    USED for the alias its own sentence names; this says that when exactly one
    extracted constraint names THIS action's alias, and it validates, it is the
    one that belongs here. Nothing is guessed: a constraint with no bind_alias,
    an alias with two competing constraints, or an action that already carries
    a reference_constraint_id or relation is left untouched, and the guard then
    has the final word either way."""
    constraints = _trusted_reference_constraints(context_json)
    if not constraints:
        return
    for action in _all_actions_in(plan.get("root")):
        if str(action.get("skill") or "") not in _SELF_LOCATING_TERMINAL_SKILLS:
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else None
        if not args:
            continue
        if str(args.get("reference_constraint_id") or "").strip():
            continue
        if str(args.get("relation") or "").strip():
            continue  # the guard rejects this outright; never paper over it here
        action_alias = _normalize_alias(_action_alias_or_name(args))
        if not action_alias:
            continue
        owned = [
            constraint_id
            for constraint_id, constraint in constraints.items()
            if _normalize_alias(str(constraint.get("bind_alias") or "").strip()) == action_alias
            and not _validate_reference_constraint(constraint, intent_text=intent_text)[1]
        ]
        if len(owned) == 1:
            args["reference_constraint_id"] = owned[0]


def _apply_reference_constraint_guard(
    plan: dict[str, Any], *, context_json: str, intent_text: str,
) -> str | None:
    """Identity/Grounding foundation finalization (2026-09-03, GPT
    re-review): the deterministic enforcement layer for the
    reference_constraints contract. Three rules, mirroring
    _apply_grounding_normalizer's own shape for entity_id:

    - `relation` must NEVER be set directly on a remember_person/
      remember_entity Action's own args -- the planner may only reference
      an existing, EXTRACTOR-produced constraint via
      `reference_constraint_id`. There is no path by which the planner's
      own say-so about `relation` is ever trusted, under any circumstance.
    - A referenced constraint must exist in the TRUSTED extractor output
      (context_json.reference_constraints, never plan.reference_constraints)
      and pass every check in _validate_reference_constraint.
    - Identity/Grounding foundation finalization v3 (2026-09-03, GPT
      re-review): a constraint's own relation/frame/class being real and
      verified is NOT enough on its own -- a confused (not malicious)
      planner could still borrow a genuinely real constraint that belongs
      to a DIFFERENT mention for THIS action's alias/name (GPT's own worked
      example: "The person on your left is waving. Remember the person on
      your right as 44." -- referencing the LEFT constraint for alias "44"
      would pass every per-constraint check while binding the wrong
      person). Whenever the referenced constraint's own `bind_alias` (set
      by the extractor ONLY when a sentence unambiguously names who it is
      about -- see reference_extraction.py's own comment) is non-empty, it
      must exactly match this action's alias/name.
    - Identity/Grounding foundation finalization v4 (2026-09-04, GPT
      re-review): v3's own fix still had one exception left -- an EMPTY
      bind_alias was accepted without a match whenever only ONE constraint
      existed in the mission (`elif len(constraints) > 1`), reasoning a
      lone constraint could never be ambiguous about which action it
      belongs to. GPT's re-review found this exception itself still lets
      an unowned real spatial constraint -- relation/frame/class all
      genuinely verified, but the sentence never actually names WHO it is
      for -- be used for identity binding under ANY alias the planner
      happens to write, as long as it is the only constraint present in
      that mission. Fixed: an empty bind_alias is now rejected
      unconditionally, with no exception for constraint count. A
      reference_constraint is usable for identity binding only when its
      own sentence unambiguously names the specific alias/name being
      bound -- full stop, regardless of how many other constraints exist.
      The original E.1 phrasing ("There is a person right in front of you.
      Remember them as 44.") was never relying on this exception to begin
      with -- v3's own cross-sentence pattern already captures
      bind_alias="44" for it directly -- so it keeps working unchanged.

    On success, `relation` is filled into the action's own args (so
    seattle_lab's skill execution code, which already just reads
    args.get("relation"), needs no changes at all). Mutates `plan` in
    place for the fill-in case, exactly like _apply_grounding_normalizer.
    Returns None when the plan needs no rejection.
    """
    constraints = _trusted_reference_constraints(context_json)
    for action in _all_actions_in(plan.get("root")):
        skill = str(action.get("skill") or "")
        if skill not in _SELF_LOCATING_TERMINAL_SKILLS:
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else None
        if not args:
            continue
        if str(args.get("relation") or "").strip():
            return (
                f"plan sets relation directly on skill {skill!r} -- relation must never be set "
                "directly by the planner; reference an entry in context_json."
                "reference_constraints via reference_constraint_id instead"
            )
        constraint_id = str(args.get("reference_constraint_id") or "").strip()
        if not constraint_id:
            continue
        constraint = constraints.get(constraint_id)
        if constraint is None:
            return (
                f"plan uses reference_constraint_id {constraint_id!r} on skill {skill!r} that "
                "does not match any entry in context_json.reference_constraints"
            )
        relation, error = _validate_reference_constraint(constraint, intent_text=intent_text)
        if error:
            return f"plan's reference_constraint {constraint_id!r} on skill {skill!r} is invalid: {error}"
        bind_alias = str(constraint.get("bind_alias") or "").strip()
        action_alias = _action_alias_or_name(args)
        if bind_alias:
            if _normalize_alias(bind_alias) != _normalize_alias(action_alias):
                return (
                    f"plan's reference_constraint {constraint_id!r} on skill {skill!r} is bound "
                    f"to alias/name {bind_alias!r} in this mission's own words, but this action "
                    f"uses {action_alias!r} -- a reference_constraint can only be used for the "
                    "specific binding its own sentence actually names"
                )
        else:
            return (
                f"plan's reference_constraint {constraint_id!r} on skill {skill!r} is not "
                f"deterministically associated with alias/name {action_alias!r} -- this "
                "constraint's own sentence never unambiguously names who it is for "
                "(context_json.reference_constraints[...].bind_alias is empty), so it cannot "
                "be confirmed to belong to this specific binding, regardless of how many other "
                "reference_constraints exist in this mission"
            )
        args["relation"] = relation
    return None


# FOUND LIVE 2026-09-03 (GPT review, Gate-1): _terminal_self_locating_action
# used to operate on a flat, position-blind list of every physical Action
# anywhere in the tree (_physical_actions_in's own contract, by design, for
# its OTHER callers like PolicyGuard-style counting). For a plan-REWRITE
# decision that is not safe -- a flat list cannot tell a genuine
# `Sequence: [look_at(...), remember_entity(...)]` from
# `Fallback: [branch A: look_at(...), branch B: remember_entity(...)]`
# (mutually exclusive alternatives, not a prefix at all) or from
# `Sequence: [remember_entity(...), look_at(...)]` (look_at comes AFTER,
# order was never actually checked). _physical_actions_with_sequence_position
# below pairs each action with the id() of the Sequence node that DIRECTLY
# contains it (None if it is not a direct Sequence child at all -- inside a
# Fallback/Parallel/Retry/Timeout instead) and its index in that Sequence's
# own children list, so _terminal_self_locating_action can require every
# "redundant prefix" candidate to be a genuine sibling of the terminal
# action, in the SAME Sequence, at a strictly smaller index -- never across
# a branch, never out of order.

def _physical_actions_with_sequence_position(
    node: Any, *, parent_sequence_id: int | None = None, index_in_parent: int | None = None,
) -> list[tuple[dict[str, Any], int | None, int | None]]:
    """Every physical Action node, each paired with (parent_sequence_id,
    index_in_parent) -- both None when the action is not a direct child of
    a Sequence node (inside a Fallback/Parallel, or the child of a Retry/
    Timeout, all of which reset the "same Sequence prefix" relationship
    entirely, on purpose)."""
    found: list[tuple[dict[str, Any], int | None, int | None]] = []
    if not isinstance(node, dict):
        return found
    node_type = node.get("type")
    if node_type == "Sequence":
        seq_id = id(node)
        for i, child in enumerate(node.get("children", []) or []):
            found.extend(_physical_actions_with_sequence_position(
                child, parent_sequence_id=seq_id, index_in_parent=i))
        return found
    if node_type in {"Fallback", "Parallel"}:
        for child in node.get("children", []) or []:
            found.extend(_physical_actions_with_sequence_position(child))
        return found
    if node_type in {"Retry", "Timeout"}:
        return _physical_actions_with_sequence_position(node.get("child"))
    if node_type == "Action":
        skill = str(node.get("skill") or "")
        if skill in PHYSICAL_SKILLS:
            found.append((node, parent_sequence_id, index_in_parent))
    return found


def _terminal_self_locating_action(
    positioned_actions: list[tuple[dict[str, Any], int | None, int | None]],
    *,
    intent_text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """(terminal, redundant_prefix_actions) for a plan whose only
    non-redundant physical action is remember_person/remember_entity, where
    every OTHER physical action anywhere in the tree is a genuine sibling of
    it -- same Sequence, strictly earlier position -- and is either a
    locate-type skill that ALSO targets the same entity as the terminal
    action (see _redundant_locate_matches_terminal), or a look_at with no
    genuine textual support of its own (see below), dropped as pure
    redundant motion (remember_person/remember_entity locate internally
    regardless). None if the pattern does not hold (zero or 2+
    remember-type actions, the terminal itself not a direct Sequence
    child, a physical action anywhere else in the tree that is not a
    genuine same-Sequence prefix of the terminal, or a locate action that
    targets something else). redundant_prefix_actions can be empty even
    when the pattern otherwise matches (see the look_at case below) -- the
    caller already treats an empty list as "nothing to rewrite".

    Gate-1.1 architecture round (2026-09-03, GPT re-review): this used to
    ALSO copy a mappable look_at direction into the terminal's own
    `relation` arg before dropping it -- GPT's re-review found this treated
    a look_at's direction as reference evidence unconditionally, with no
    check that the user's words ever described a PERSON's position rather
    than a literal turn instruction. look_at's direction is never harvested
    into `relation` here under any circumstance; the terminal's own
    `relation` (if any) comes ONLY from a validated reference_constraint
    (_apply_reference_constraint_guard, run later in plan()).

    Identity/Grounding foundation finalization (2026-09-03, GPT re-review):
    a SEPARATE, later finding on the same mechanism -- dropping a
    preceding look_at as "redundant" is only safe when it genuinely IS
    redundant. "Look to your left, then remember this person as 44" is two
    real, distinct user intentions (turn left; then remember whoever you
    then see) -- the look_at is an EXPLICIT physical action the user asked
    for, not planner filler, and silently deleting it means the robot
    never performs the turn the user explicitly requested (worse: with
    only one person currently in view, it could bind that person's
    identity without ever having looked where the user pointed). A look_at
    is now excluded from the redundant set whenever intent_text contains a
    genuine look/turn instruction toward its own direction
    (has_explicit_look_instruction) -- it remains in the tree and actually
    executes. This does not create a NEW relation-provenance risk: the
    preserved look_at's direction still never feeds `relation` (that path
    stays closed, per the paragraph above) -- it only changes whether the
    physical action itself is real or discarded.
    """
    terminal_entries = [
        entry for entry in positioned_actions
        if str(entry[0].get("skill") or "") in _SELF_LOCATING_TERMINAL_SKILLS
    ]
    if len(terminal_entries) != 1:
        return None
    term, term_seq_id, term_index = terminal_entries[0]
    if term_seq_id is None or term_index is None:
        return None  # terminal itself is not a direct Sequence child -- too structurally unclear to rewrite
    term_args = term.get("args") if isinstance(term.get("args"), dict) else {}
    term_target = _normalize_alias(str(term_args.get("target") or ""))

    all_others = [entry[0] for entry in positioned_actions if entry[0] is not term]
    prefix_others = [
        entry[0] for entry in positioned_actions
        if entry[0] is not term and entry[1] == term_seq_id and entry[2] is not None and entry[2] < term_index
    ]
    if len(prefix_others) != len(all_others):
        return None  # some other physical action exists OUTSIDE this safe same-Sequence-prefix relationship

    look_at_actions = [a for a in prefix_others if str(a.get("skill") or "") == "look_at"]
    other_locates = [a for a in prefix_others if str(a.get("skill") or "") != "look_at"]

    for a in other_locates:
        if str(a.get("skill") or "") not in _REDUNDANT_LOCATE_SKILLS:
            return None
        if not _redundant_locate_matches_terminal(a, term_target):
            return None

    if len(look_at_actions) > 1:
        return None  # more than one bearing -- not a pattern worth guessing at
    if look_at_actions and term_args.get("entity_id"):
        return None  # terminal already has its own, fully-specified identity

    redundant = list(other_locates)
    if look_at_actions:
        look_at = look_at_actions[0]
        look_at_args = look_at.get("args") if isinstance(look_at.get("args"), dict) else {}
        direction = str(look_at_args.get("direction") or "")
        if not has_explicit_look_instruction(intent_text, direction):
            redundant.append(look_at)
        # else: a genuine, user-requested physical action -- preserved,
        # left in the tree, never dropped.

    return term, redundant


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


_TARGET_ACQUIRING_TERMINALS = frozenset(
    name for name, spec in DEFAULT_SKILLS.items() if spec.resolves_own_target_acquisition
)


def _predicate_producers() -> dict[str, set[str]]:
    producers: dict[str, set[str]] = {}
    for name, spec in DEFAULT_SKILLS.items():
        for predicate in spec.result_predicates:
            producers.setdefault(predicate, set()).add(name)
    return producers


def _canonicalize_gate_the_terminal_resolves_itself(plan: dict[str, Any]) -> None:
    """FOUND LIVE 2026-09-05 (E1 TURN1, fourth and fifth attempts): the planner
    gated a target-acquiring terminal on the very thing that terminal does --

        Sequence[ Condition(entity_located), remember_person(name=33) ]

    -- and the mission stalled on "condition UNKNOWN: no verification evidence
    for entity_located" every time, because nothing had run to establish it.
    Two prompt revisions did not stop it, so it is removed here instead.

    NARROWED 2026-09-05 after review, and this is the important part. The first
    version of this function decided a gate was unsatisfiable whenever no
    EARLIER ACTION in the plan produced its predicate. That inference is simply
    false: a Condition may perfectly well check a fact that already holds in
    WorldState -- person_visible has no producing skill at all, and gating on it
    is entirely legitimate. Written that way this function would have silently
    deleted real pre-execution checks, weakening exactly the execution-time
    verification the D/WorldState layers exist to provide -- a much worse defect
    than the stall it was fixing.

    So the test is now contract-driven, not inference-driven: a gate is removed
    only when its predicate is one the terminal itself declares it resolves
    internally, derived from subsumes_locate_skills via
    internally_resolved_predicates(). For remember_person/remember_entity that
    is exactly {entity_located, search_for_entity_completed}. Every other
    predicate -- person_visible, robot_at_place, person_named, anything a policy
    or the world already answers -- is left alone, whether or not any Action
    produces it.

    Still additionally producer-aware: if an earlier Action really did produce
    the predicate, the pair is a genuine "locate, then confirm that locate
    worked" and is kept untouched. Mutates plan["root"] in place."""
    producers = _predicate_producers()

    def rewrite(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node_type = node.get("type")
        if node_type in {"Sequence", "Fallback", "Parallel"}:
            children = [c for c in (node.get("children") or []) if isinstance(c, dict)]
            drop: set[int] = set()
            if node_type == "Sequence":
                for i, child in enumerate(children):
                    if child.get("type") not in {"Condition", "GoalCheck"}:
                        continue
                    predicate = str(child.get("predicate") or "")
                    if not predicate:
                        continue
                    resolved_by_a_later_terminal = any(
                        later.get("type") == "Action"
                        and predicate in internally_resolved_predicates(
                            str(later.get("skill") or ""))
                        for later in children[i + 1:]
                    )
                    if not resolved_by_a_later_terminal:
                        continue
                    produced_earlier = any(
                        earlier.get("type") == "Action"
                        and str(earlier.get("skill") or "") in producers.get(predicate, set())
                        for earlier in children[:i]
                    )
                    if not produced_earlier:
                        drop.add(i)
            new_node = dict(node)
            new_node["children"] = [
                rewrite(c) for i, c in enumerate(children) if i not in drop
            ]
            return new_node
        if node_type in {"Retry", "Timeout"}:
            new_node = dict(node)
            new_node["child"] = rewrite(node.get("child"))
            return new_node
        return node

    plan["root"] = rewrite(plan.get("root"))


def _canonicalize_redundant_locate_prefix(plan: dict[str, Any], *, intent_text: str) -> None:
    """Actually remove a redundant look_at/search_for_entity/locate_entity
    Action from the executable BT tree when it immediately duplicates work
    remember_person/remember_entity already does internally for the SAME
    target (see _terminal_self_locating_action) -- not just skip it when
    deriving goal_spec (that alone left it in the tree to actually run,
    which is what a real live E.1 attempt showed blocking the mission).
    Mutates plan["root"] in place. A no-op when the pattern does not match
    (including a locate action that targets something else, or a look_at
    with genuine textual support of its own -- never touched)."""
    match = _terminal_self_locating_action(
        _physical_actions_with_sequence_position(plan.get("root")), intent_text=intent_text)
    if match is None:
        return
    _terminal, redundant = match
    if not redundant:
        return
    redundant_ids = {id(a) for a in redundant}
    plan["root"] = _without_redundant_locate_actions(plan.get("root"), redundant_ids)


_MISSION_GOAL_CONTRACT_SCHEMA = "mc_ai_bt.mission_goal_contract.v1"
_VALID_TARGET_KINDS = frozenset({"spatial_reference", "alias_reference"})
# Machine-owned execution metadata, never a planner-facing semantic Action
# input -- deliberately absent from every SkillSpec.args_schema (those are
# the LLM-prompt catalog). It only ever exists because THIS module's own
# _synthesize_mission_goal constructs it from scratch; the LLM planner is
# never consulted for a V1 mission-goal plan's action list at all (see
# PlanningPipeline.plan's own comment), so there is no untrusted candidate
# to strip a forged copy of this key from in the first place.
_INTERNAL_GROUNDING_FIELD = "_planning_grounding_ref"


def _strip_internal_grounding_field(plan: dict[str, Any]) -> None:
    """Trust-boundary enforcement for the LEGACY (no MissionGoalContract)
    path only: _synthesize_mission_goal above never reads plan_json at all,
    so it has nothing to strip a forged copy of _INTERNAL_GROUNDING_FIELD
    from. THIS path is different -- plan_json here is the LLM planner's own
    untrusted output, and although it is never told this key exists (it is
    absent from every SkillSpec.args_schema), a planner that guessed or
    hallucinated it must not be able to smuggle a fake -- or a real but
    unrelated -- grounding_ref past seattle_lab's explicit-entity_id
    evidence_ref propagation (GPT-approved trust boundary). Unconditional:
    it does not matter whether a present value looks plausible: this key
    is machine-owned in every code path, so any externally-authored
    occurrence of it is stripped before any other stage ever sees it,
    the same "strip first, only a trusted stage may inject" shape
    _apply_grounding_normalizer already uses for entity_id."""
    for action in _all_actions_in(plan.get("root")):
        args = action.get("args")
        if isinstance(args, dict):
            args.pop(_INTERNAL_GROUNDING_FIELD, None)


@dataclass(frozen=True)
class MissionGoal:
    goal_id: str
    predicate: str
    target_id: str
    alias: str = ""


def _parse_mission_goal_contract(
    context_json: str,
) -> tuple[str, tuple[MissionGoal, ...], dict[str, dict[str, str]]]:
    """B (MissionGoalContract V1, GPT-approved architecture amendment to
    FROZEN #14/#15): reads context_json.mission_goal_contract -- a
    machine-materialized structure Bridge produces from Omega's own
    terminal trailer syntax on submit_mission's single argument, never
    free-form text this module itself interprets (Bridge already stripped
    and parsed it; this function never sees intent_text).

    Returns (state, goals, targets):
      "ABSENT"              -- no contract this mission; legacy pipeline
                                (Change A and everything before it) runs
                                completely unchanged, see plan().
      "VALID"               -- well-formed: every goal's target_id resolves
                                to a declared target, every predicate is a
                                real registry predicate, every relation/
                                entity_class/reference_frame is in the same
                                closed vocabulary _validate_reference_constraint
                                already enforces above.
      "PRESENT_BUT_INVALID" -- Omega/Bridge intended to provide one but it
                                fails validation -- plan() must REJECT this
                                mission outright, never silently fall back
                                to legacy execution (an invalid semantic
                                authorization is not the same epistemic
                                state as no authorization being offered).

    Defense in depth, same posture as _validate_reference_constraint: this
    does not blindly trust Bridge's own JSON shape either, even though
    Bridge is the machine-owned materializer -- a stale/hand-crafted
    context_json must fail exactly like a hallucinated one would."""
    try:
        context = json.loads(context_json) if context_json else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return "ABSENT", (), {}
    if not isinstance(context, dict):
        return "ABSENT", (), {}
    # Same location grounded_entities already lives at (_grounded_entity_ids
    # above reads caller_context too) -- Bridge owns and constructs
    # mission.context_json, which ContextBuilder.build_json wraps unchanged
    # into context.caller_context; this is not a second, competing
    # context-building authority, just the same existing field.
    caller_context = context.get("caller_context")
    if not isinstance(caller_context, dict):
        return "ABSENT", (), {}
    contract = caller_context.get("mission_goal_contract")
    if contract is None:
        return "ABSENT", (), {}
    if not isinstance(contract, dict) or contract.get("schema") != _MISSION_GOAL_CONTRACT_SCHEMA:
        return "PRESENT_BUT_INVALID", (), {}

    raw_targets = contract.get("targets")
    raw_goals = contract.get("goals")
    if not isinstance(raw_targets, dict) or not raw_targets:
        return "PRESENT_BUT_INVALID", (), {}
    if not isinstance(raw_goals, list) or not raw_goals:
        return "PRESENT_BUT_INVALID", (), {}

    targets: dict[str, dict[str, str]] = {}
    for target_id, raw_target in raw_targets.items():
        if not isinstance(target_id, str) or not target_id or not isinstance(raw_target, dict):
            return "PRESENT_BUT_INVALID", (), {}
        kind = str(raw_target.get("kind") or "")
        if kind not in _VALID_TARGET_KINDS:
            return "PRESENT_BUT_INVALID", (), {}
        if kind == "spatial_reference":
            entity_class = str(raw_target.get("entity_class") or "").strip().lower()
            relation = str(raw_target.get("relation") or "").strip().lower()
            reference_frame = str(raw_target.get("reference_frame") or "").strip().lower()
            if entity_class != "person" or relation not in _VALID_RELATIONS or reference_frame != "robot":
                return "PRESENT_BUT_INVALID", (), {}
            targets[target_id] = {
                "kind": kind, "entity_class": entity_class,
                "relation": relation, "reference_frame": reference_frame,
            }
        else:  # alias_reference
            alias = str(raw_target.get("alias") or "").strip()
            if not alias:
                return "PRESENT_BUT_INVALID", (), {}
            targets[target_id] = {"kind": kind, "alias": alias}

    goals: list[MissionGoal] = []
    seen_goal_ids: set[str] = set()
    for raw_goal in raw_goals:
        if not isinstance(raw_goal, dict):
            return "PRESENT_BUT_INVALID", (), {}
        goal_id = str(raw_goal.get("goal_id") or "").strip()
        predicate = str(raw_goal.get("predicate") or "").strip()
        target_id = str(raw_goal.get("target_id") or "").strip()
        alias = str(raw_goal.get("alias") or "").strip()
        if (
            not goal_id or goal_id in seen_goal_ids
            or predicate not in KNOWN_PREDICATE_NAMES
            or target_id not in targets
        ):
            return "PRESENT_BUT_INVALID", (), {}
        seen_goal_ids.add(goal_id)
        goals.append(MissionGoal(goal_id=goal_id, predicate=predicate, target_id=target_id, alias=alias))

    if not goals:
        return "PRESENT_BUT_INVALID", (), {}
    return "VALID", tuple(goals), targets


def _resolve_mission_target(
    target: dict[str, str], *, context_json: str, reference_resolver: Any,
) -> tuple[str, str, str]:
    """(state, entity_id, grounding_ref). state is one of
    RESOLVED/AMBIGUOUS/NOT_FOUND/UNKNOWN.

    alias_reference targets reuse the EXISTING, Bridge-materialized
    grounded_entities lookup (_grounded_entity_ids above) -- the identical
    data _apply_grounding_normalizer already trusts for navigating to an
    already-bound alias (E1 TURN2's own path); no grounding_ref concept
    applies here (no NEW identity decision is being made, only navigation
    to one already on record), and an UNRESOLVED alias fails closed with
    no nearest-same-class fallback, exactly like the existing negative-test
    contract.

    spatial_reference targets go through reference_resolver -- the
    planning-time CALLER of the existing, unmodified
    /mc_world_state/resolve_entity_reference service (the SAME authoritative
    service remember_person/remember_entity's own execution already calls;
    this is one more caller of it, not a new grounding engine). Returns
    UNKNOWN (never fabricates RESOLVED) if no resolver was injected or the
    service is not reachable."""
    if target["kind"] == "alias_reference":
        grounded = _grounded_entity_ids(context_json)
        entry = grounded.get(_normalize_alias(target["alias"]))
        if entry is None or entry.get("grounding_state") != "RESOLVED" or not entry.get("entity_id"):
            return "NOT_FOUND", "", ""
        return "RESOLVED", entry["entity_id"], ""
    if reference_resolver is None:
        return "UNKNOWN", "", ""
    return reference_resolver.resolve(
        entity_class=target["entity_class"],
        relation=target["relation"],
        reference_frame=target["reference_frame"],
    )


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
    and needs no special case here at all.)

    FOUND LIVE 2026-09-09: "2+ physical actions -> always left alone" above
    turned out to have exactly one more genuinely unambiguous exception:
    search_for_entity/locate_entity immediately followed by approach_entity
    for the same target (the ordinary "find X"/"go to X" shape for an
    object/person with no bound alias) -- see
    _fill_search_then_approach_goal_spec's own docstring for why this
    specific 2-action case is not a guess either. Tried first, since it only
    ever fires on a real 2-physical-action plan the code below would
    otherwise leave untouched.

    FOUND LIVE 2026-09-13 (real mission-journal evidence, not a synthetic
    case): the above still left a large real class of common, unambiguous
    multi-action plans falling through to rejection -- "look_at, reach_to"
    ("Reach your hand toward the person in front of you"), "look_at,
    remember_person" ("Find the person in front of you and remember them as
    33"), and reordered/longer chains of the same shape the narrow fill above
    does not cover (any order/count of pure target-acquisition/orientation
    steps ahead of one terminal action). _fill_enabling_sequence_goal_spec
    generalizes the same "exactly one action's result could sensibly be the
    mission's success criterion" reasoning to any length of leading
    ENABLING-only steps -- see its own docstring for the exact allowlist and
    the same-target safety check that keeps a genuinely suspicious plan (e.g.
    searching for one thing then approaching an unrelated other one) falling
    through unchanged. Tried after the narrower, longer-established fill
    above (never overrides it), before the single-action case below."""
    goal_spec = plan.get("goal_spec")
    if not isinstance(goal_spec, dict):
        return
    verification = goal_spec.get("verification") if isinstance(goal_spec.get("verification"), dict) else {}
    is_implicit = goal_spec.get("type") == "human" or str(verification.get("mode") or "") == "implicit_conversation"
    if not is_implicit:
        return

    if _fill_search_then_approach_goal_spec(plan):
        return
    if _fill_enabling_sequence_goal_spec(plan):
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
        reference_resolver: Any = None,
    ) -> None:
        self._planner = planner
        self._context_builder = context_builder
        self._validator = validator
        self._policy_guard = policy_guard
        # B (MissionGoalContract V1, GPT-approved): optional -- every
        # existing caller/test that omits it gets exactly today's behavior,
        # since it is only ever consulted inside _synthesize_mission_goal,
        # itself only reached when a VALID V1 contract is present (see
        # plan() below). None here plus no contract in context_json is a
        # complete no-op, not a degraded mode.
        self._reference_resolver = reference_resolver

    def _synthesize_mission_goal(
        self, goal: "MissionGoal", targets: dict[str, dict[str, str]], *, context_json: str,
    ) -> PlanningResult:
        """B (MissionGoalContract V1, GPT-approved): the LLM candidate
        planner is never consulted here -- plan() does not even call it for
        a mission carrying a VALID single-goal contract (see plan()'s own
        comment on why that makes the usual "strip a planner-forged
        internal field" defense vacuous rather than skipped: there is no
        externally-sourced args dict for this method to strip a forged
        value FROM in the first place, every key below is constructed by
        this function alone). Deterministic, registry-derived producer
        selection -- 0 or 2+ producers is a fail-closed reject, not a guess
        (same posture as every other producer-selection rule in this
        module)."""
        producers = [
            name for name, spec in DEFAULT_SKILLS.items()
            if goal.predicate in spec.result_predicates
        ]
        if len(producers) != 1:
            return PlanningResult(
                False, "mission_goal_contract",
                f"predicate {goal.predicate!r} has {len(producers)} registry producer(s) (need exactly 1)",
                context_json=context_json,
            )
        skill = producers[0]

        target = targets.get(goal.target_id)
        if target is None:
            return PlanningResult(
                False, "mission_goal_contract",
                f"goal {goal.goal_id!r} references unknown target_id {goal.target_id!r}",
                context_json=context_json,
            )

        state, entity_id, grounding_ref = _resolve_mission_target(
            target, context_json=context_json, reference_resolver=self._reference_resolver)
        if state != "RESOLVED":
            return PlanningResult(
                False, "target_resolution",
                f"target {goal.target_id!r} resolution state is {state} -- no physical BT authorized "
                "(no nearest/same-class fallback)",
                context_json=context_json,
            )

        if target["kind"] == "spatial_reference":
            label = f"{target['relation']} {target['entity_class']}"
        else:
            label = target["alias"]
        args: dict[str, Any] = {"target": label, "entity_id": entity_id}
        goal_spec_args: dict[str, Any] = {"target": label, "entity_id": entity_id}

        if skill in _SELF_LOCATING_TERMINAL_SKILLS:
            if not goal.alias:
                return PlanningResult(
                    False, "mission_goal_contract",
                    f"goal {goal.goal_id!r} (predicate {goal.predicate!r}) requires an alias, none provided",
                    context_json=context_json,
                )
            name_key = "name" if skill == "remember_person" else "alias"
            args[name_key] = goal.alias
            goal_spec_args[name_key] = goal.alias
            # Machine-owned evidence propagation (GPT-approved trust
            # boundary): only the explicit-entity_id naming path, which
            # today has no upstream decision evidence of its own to point
            # to (seattle_lab/mc_embodied_skills/node.py's own
            # _remember_entity_by_id, evidence_ref=""), consumes this.
            # The normal relation/reference_constraint_id path is untouched
            # and keeps producing its own fresher, execution-time evidence.
            if target["kind"] == "spatial_reference" and grounding_ref:
                args[_INTERNAL_GROUNDING_FIELD] = grounding_ref

        plan = {
            "root": {"type": "Action", "skill": skill, "args": args},
            "goal_spec": {
                "type": "structured", "predicate": goal.predicate,
                "args": goal_spec_args, "verification": {"mode": "world_state"},
            },
        }
        policy = self._policy_guard.check(plan)
        if not policy.ok:
            return PlanningResult(
                False, "policy", "; ".join(policy.errors), context_json=context_json,
            )
        return PlanningResult(
            True, "done", "planned",
            context_json=context_json,
            bt_json=json.dumps(plan["root"], sort_keys=True, separators=(",", ":")),
            goal_spec_json=json.dumps(plan["goal_spec"], sort_keys=True, separators=(",", ":")),
        )

    def plan(
        self,
        mission: Mission,
        missions: tuple[Mission, ...],
    ) -> PlanningResult:
        context_json = self._context_builder.build_json(mission, missions)

        # B (MissionGoalContract V1, GPT-approved architecture amendment):
        # checked BEFORE ever calling the LLM planner. A VALID single-goal
        # contract is deterministically synthesized end to end without the
        # LLM's own candidate action list ever being consulted for
        # authorization -- see _synthesize_mission_goal's own comment on
        # why that is the actual trust boundary, not merely a stripping
        # step applied after the fact. ABSENT falls straight through to
        # every existing stage below (Change A and everything before it),
        # completely unchanged. PRESENT_BUT_INVALID and >1 goals (V1 scope:
        # GoalCheck has no conjunction semantics yet) both REJECT outright
        # -- never a silent downgrade to legacy execution, per the
        # explicit fail-closed distinction GPT's review required.
        goal_state, goals, targets = _parse_mission_goal_contract(context_json)
        if goal_state == "PRESENT_BUT_INVALID":
            return PlanningResult(
                False, "mission_goal_contract",
                "mission_goal_contract is present but invalid (malformed shape, unknown "
                "predicate/relation/entity_class, or a goal references an undeclared target_id) "
                "-- rejecting outright, not falling back to legacy candidate-plan execution",
                context_json=context_json,
            )
        if goal_state == "VALID":
            if len(goals) > 1:
                return PlanningResult(
                    False, "mission_goal_contract",
                    f"UNSUPPORTED_V1: contract declares {len(goals)} required goals; "
                    "GoalCheck does not yet support conjunction semantics -- fail closed rather "
                    "than execute a partial mission or silently drop a goal",
                    context_json=context_json,
                )
            return self._synthesize_mission_goal(goals[0], targets, context_json=context_json)

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

        _strip_internal_grounding_field(plan)

        _append_missing_required_producers(
            plan, context_json=context_json, intent_text=mission.intent_text)

        _canonicalize_redundant_locate_prefix(plan, intent_text=mission.intent_text)
        _canonicalize_gate_the_terminal_resolves_itself(plan)

        _inject_owned_reference_constraint(
            plan, context_json=context_json, intent_text=mission.intent_text)
        reference_error = _apply_reference_constraint_guard(
            plan, context_json=context_json, intent_text=mission.intent_text)
        if reference_error:
            return PlanningResult(
                False,
                "reference_constraint",
                reference_error,
                context_json=context_json,
                plan_json=plan_json,
            )

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
