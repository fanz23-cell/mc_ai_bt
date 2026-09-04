from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillSpec:
    name: str
    resources: tuple[str, ...]
    description: str
    args_schema: dict[str, object] = field(default_factory=dict)
    result_predicates: tuple[str, ...] = ()
    realtime: bool = False
    # "embodied" (default): routes through the generic /mc_embodied_skills/execute action
    #   (skill_adapters.py's EMBODIED_SKILLS -- computed from this field, see below).
    # "dedicated": has its own ActionClient/ServiceClient in skill_adapters.py
    #   (go_to_place/come_to_me/simple_move/play_animation/look_at/point_at/
    #   request_human_confirmation).
    # "voice": handled by mc_ai_bt/voice directly, dispatches to no robot action at all (say).
    # This is the single source every other skill-name set in this file and in
    # policy_guard.py/skill_adapters.py is computed from -- see
    # OMEGACLAW_AI_BT_INTEGRATION.md §9 for why hand-copying these sets independently caused
    # three separate live production failures in one afternoon.
    dispatch: str = "embodied"
    # FOUND LIVE 2026-08-31: align_axis/move_along_axis/maintain_distance/wait_for_contact/
    # detect_contact are registered here exactly like any working skill (same dataclass
    # shape, same fields) but seattle_lab/mc_embodied_skills/node.py unconditionally
    # returns STATUS_BLOCKED for every one of them -- "requires new IK/pose-tracking work
    # not yet built" / "requires a dedicated closed-loop controller". Nothing before this
    # field told the LLM planner that; the exact same unfiltered catalog (planner.py's
    # build_planner_messages) was serialized into its prompt whether a skill actually
    # worked or not, so the planner had no way to distinguish "really works" from
    # "registered but not implemented" -- it could only find out by trying and getting
    # STATUS_BLOCKED back. "available" (every other skill): real today, safe to plan
    # with. "blocked": registered as a real capability but not implemented yet -- must
    # never reach the planner's prompt (see context_builder.py/planner.py's catalog
    # filtering). "experimental": implemented but not yet trusted enough to plan with by
    # default -- reserved for future use, currently unused.
    status: str = "available"
    # 2026-09-03 (GPT review, D1): which OTHER skill names this skill already
    # performs internally, making a separate planned Action for them purely
    # redundant -- e.g. remember_entity/remember_person locate the target
    # themselves (turning to look if needed), so a planner-authored
    # search_for_entity/locate_entity Action immediately before one of them
    # is dead weight. planning_pipeline.py's canonicalizer derives its
    # redundant-skill set from this field instead of a hand-maintained
    # literal set -- the exact "one more place to forget" pattern
    # `dispatch`'s own comment above already warns about. Deliberately does
    # NOT include look_at: its direction arg can itself carry reference
    # semantics (see planning_pipeline.py's own D0 comment for the real,
    # live information-loss bug that caused), so it is never safe to treat
    # as unconditionally redundant.
    subsumes_locate_skills: tuple[str, ...] = ()


# Real clip names from the animation library (mc_one_codey/*/context/animations/clips/),
# grouped for the play_animation description below. This is the single source both that
# description AND planner.py's BootstrapPlanner fallback are built from -- see
# tests/test_animation_clip_catalog_consistency.py, which checks every name here actually
# has a clip file on disk. Found live 2026-08-31: play_animation's description used to name
# no clips at all, so the LLM planner invented plausible-sounding ones ("wave", "nod") that
# don't exist and silently fail to play; BootstrapPlanner's own regex fallback had the exact
# same bug hardcoded as a literal string. Curated, not exhaustive -- add to it as new clips
# earn a place in the planner's vocabulary, but never let this drift from the real catalog.
# The real wave/greeting clip -- named separately so planner.py's regex
# fallback can import a real symbol instead of hardcoding the string again.
WAVE_CLIP = "wave_and_jaw"

KNOWN_ANIMATION_CLIPS: dict[str, tuple[str, ...]] = {
    "greeting/gesture": (
        WAVE_CLIP, "fist_bump", "both_beckon", "left_beckon", "right_beckon",
        "both_present", "left_present", "right_present",
    ),
    "agreement": ("yes_once", "yes_eager", "no_subtle", "not_at_all", "exactly", "different"),
    "expression": (
        "smile", "smirk", "surprised", "confused", "frustrated", "disgusted", "angry",
        "disappointment",
    ),
    "other": ("interjection", "thrilled_all", "excited_bouncing", "square_up", "stage_walk"),
}


def _play_animation_description() -> str:
    groups = "; ".join(
        f"{category} — {', '.join(names)}" for category, names in KNOWN_ANIMATION_CLIPS.items()
    )
    return (
        "Play a named animator clip or generator. `animation` must be a REAL "
        "clip name from the animation library — never invent a plausible-"
        "sounding name (there is no 'wave' or 'nod' clip; an invented name "
        f"silently fails to play). Known clips include: {groups}. If the "
        "desired gesture isn't one of these, prefer a skill above instead of "
        "guessing."
    )


DEFAULT_SKILLS: dict[str, SkillSpec] = {
    # FOUND LIVE 2026-09-01 (3rd GPT review): locate_entity/get_pose/search_for_entity all
    # dispatch to mc_embodied_skills' _locate_with_scan, which physically turns the robot
    # base (SimpleMove) up to _SCAN_MAX_TURNS times when the target isn't in the current
    # frame -- confirmed live runtime behavior, not hypothetical. "base" must be declared
    # here so PolicyGuard's max_base_actions budget and validator.py's Parallel
    # resource-conflict check both see this real motion; before this fix all three could
    # be scheduled in "Parallel" alongside a dedicated base skill (go_to_place/simple_move)
    # with zero declared conflict, while at runtime mc_embodied_skills acquired no "base"
    # lease at all for its own SimpleMove goals -- an un-arbitrated race, not just budget
    # undercounting.
    "locate_entity": SkillSpec(
        "locate_entity",
        ("base",),
        "Resolve an arbitrary entity from World State/perception without exposing hidden simulator truth.",
        {"target": "entity|entity_id|object|person"},
        ("entity_located",),
    ),
    "track_entity": SkillSpec(
        "track_entity",
        ("gaze",),
        "One-shot: look toward an entity's current perceived position, then stop. Does NOT "
        "continuously follow a moving target -- call again to re-aim if it moves.",
        {"target": "entity|entity_id|object|person"},
        ("track_entity_completed",),
    ),
    "track_frame": SkillSpec(
        "track_frame",
        ("gaze",),
        "One-shot: look toward a ROS TF frame's current position, then stop. Does NOT "
        "continuously track a moving frame -- call again to re-aim if it moves.",
        {"frame": "frame|target_frame"},
        ("track_completed",),
    ),
    "get_pose": SkillSpec(
        "get_pose",
        ("base",),  # see locate_entity's comment above -- same _locate_with_scan dispatch
        "Read the latest fresh pose for an entity/frame.",
        {"target": "entity|entity_id|target|frame"},
        ("pose_available",),
    ),
    "check_relation": SkillSpec(
        "check_relation",
        (),
        "Check a registered relation predicate against World State.",
        {"relation": "relation"},
        ("relation_checked",),
    ),
    "search_for_entity": SkillSpec(
        "search_for_entity",
        ("base", "gaze"),  # see locate_entity's comment above -- same _locate_with_scan dispatch
        "Search perception space for an arbitrary entity.",
        {"target": "entity|entity_id|object|person"},
        ("search_for_entity_completed",),
        realtime=True,
    ),
    "look_at_static": SkillSpec(
        "look_at_static",
        ("gaze",),
        "Look at a static direction, entity, frame or xyz target.",
        {"target": "direction|entity|frame|x/y/z"},
        ("look_at_static_completed",),
    ),
    "track_with_gaze": SkillSpec(
        "track_with_gaze",
        ("gaze",),
        "One-shot despite the name: looks toward a target (entity or frame) once, then "
        "stops -- does NOT continuously follow it. Call again to re-aim if it moves.",
        {"target": "entity|frame"},
        ("track_with_gaze_completed",),
    ),
    "reach_to": SkillSpec(
        "reach_to",
        ("right_arm",),
        "Closed-loop arm reach to an arbitrary fresh target; may seed from static contact_pose.",
        {"target": "entity|object|person|frame|x/y/z", "arm": "left|right"},
        ("reach_completed",),
        realtime=True,
    ),
    "align_axis": SkillSpec(
        "align_axis",
        ("right_arm",),
        "Align a tool/hand axis with a target axis under controller feedback.",
        {"axis": "axis", "target": "entity|frame|x/y/z"},
        ("axis_aligned",),
        realtime=True,
        status="blocked",  # requires new IK/pose-tracking work not yet built
    ),
    "move_along_axis": SkillSpec(
        "move_along_axis",
        ("right_arm",),
        "Move a controlled frame along an axis by a bounded distance.",
        {"axis": "axis", "distance_m": "number"},
        ("axis_motion_completed",),
        realtime=True,
        status="blocked",  # requires new IK/pose-tracking work not yet built
    ),
    "maintain_distance": SkillSpec(
        "maintain_distance",
        ("base",),
        "Maintain a bounded distance to a realtime entity.",
        {"target": "entity|person|object", "distance_m": "number"},
        ("distance_maintained",),
        realtime=True,
        status="blocked",  # requires a dedicated closed-loop controller
    ),
    "hold_pose": SkillSpec(
        "hold_pose",
        ("right_arm",),
        "Hold an arm/tool pose for a bounded duration.",
        {"target": "entity|frame|x/y/z", "duration_sec": "number"},
        ("pose_held",),
        realtime=True,
    ),
    "wait_for_contact": SkillSpec(
        "wait_for_contact",
        ("right_arm",),
        "Wait for generic contact evidence from sensors/verifiers.",
        {"target": "entity|object|person", "min_duration_sec": "number"},
        ("contact_detected",),
        realtime=True,
        status="blocked",  # requires a dedicated closed-loop controller
    ),
    "detect_contact": SkillSpec(
        "detect_contact",
        ("right_arm",),
        "Read generic contact evidence without interpreting task success.",
        {"target": "entity|object|person"},
        ("contact_detected",),
        realtime=True,
        status="blocked",  # requires a dedicated closed-loop controller
    ),
    "oscillate": SkillSpec(
        "oscillate",
        ("right_arm",),
        "Apply bounded local oscillation around a target under safety limits.",
        {"target": "entity|object|person|frame|x/y/z", "axis": "axis", "extent_m": "number", "cycles": "integer"},
        ("oscillation_completed",),
        realtime=True,
    ),
    "retract": SkillSpec(
        "retract",
        ("right_arm",),
        "Safely retract an arm/tool from the current interaction.",
        {"arm": "left|right"},
        ("retracted",),
        realtime=True,
    ),
    "go_to_place": SkillSpec(
        "go_to_place",
        ("base",),
        "Navigate to an arbitrary configured place region.",
        {"place": "name|place"},
        ("robot_at_place",),
        dispatch="dedicated",
    ),
    "approach_entity": SkillSpec(
        "approach_entity",
        ("base",),
        "Navigate to an entity's current live position (from locate_entity/perception), "
        "not a preconfigured place -- use this for 'go to <person/object>' when no named "
        "place applies. Fails if the entity has not been perceived recently. If a prior "
        "remember_entity call bound an alias to a specific `entity_id`, pass that "
        "entity_id too (alongside `target` as a human-readable label) -- this makes the "
        "approach identity-aware: it navigates to and re-verifies that SPECIFIC tracked "
        "entity, never falling back to 'nearest same-class instance' the way a bare "
        "`target` search does.",
        {"target": "entity|entity_id|object|person", "entity_id": "optional, from remember_entity/entity_tracks"},
        ("entity_approached",),
    ),
    "face_entity": SkillSpec(
        "face_entity",
        ("base",),
        "Rotate in place (no travel) to face an entity's current live position. A single "
        "bounded turn computed once from a snapshot -- does not continuously track a moving "
        "target; call again to re-aim. target must be a real, perceivable person/object name "
        "(e.g. 'the plant'), never a bare compass direction like 'right' or 'left' -- for "
        "'turn right'/'turn left' with no named target, use simple_move instead.",
        {"target": "entity|entity_id|object|person"},
        ("entity_faced",),
    ),
    "follow_entity": SkillSpec(
        "follow_entity",
        ("base",),
        "Continuously follow an arbitrary realtime entity.",
        {"target": "entity|person|object"},
        ("entity_following",),
        realtime=True,
    ),
    "remember_place": SkillSpec(
        "remember_place",
        (),
        "Save the robot's current location as a named place, usable later by go_to_place. "
        "Requires a map to be loaded; does not move the robot.",
        {"name": "place name to save"},
        ("place_remembered",),
        dispatch="dedicated",
    ),
    # FOUND LIVE 2026-08-31: naming a person used to go through no skill at all -- bare
    # chat text, no perception check, so a name could be "remembered" for a bearing the
    # robot never actually looked at. Mirrors search_for_entity's real-evidence contract
    # (checks the current view, then physically turns and looks before giving up) --
    # the target must actually be found before the name is bound.
    # FOUND LIVE 2026-09-01 (4th GPT review): calls _locate_with_scan (same as
    # search_for_entity) -- turns the base, not just gaze. Was declared "gaze" only.
    "remember_person": SkillSpec(
        "remember_person",
        ("base", "gaze"),
        "Bind a name to a person the robot can currently see or can find by looking "
        "around -- e.g. 'the person on your left is named Alice'. Actually locates the "
        "person first (turning to look if needed, same as search_for_entity); refuses "
        "to bind a name to someone it cannot actually find. `target` describes who "
        "(a bearing, description, or an id from a prior locate_entity/search_for_entity "
        "result), `name` is what to call them. If 2+ people are simultaneously visible "
        "(ambiguous which one is meant) AND the user's own words identify which one by a "
        "spatial relation TO YOU, pass `reference_constraint_id` (never a bare `relation` "
        "string -- see the system prompt's reference_constraints section for the full "
        "contract) referencing an entry in context_json.reference_constraints -- a "
        "pre-extracted, trusted list computed before planning, never something you author "
        "yourself -- resolved deterministically against real robot position/orientation, "
        "never guessed.",
        {"target": "entity|entity_id|person", "name": "name to bind",
         "reference_constraint_id": "optional: id of an entry in context_json."
         "reference_constraints -- only when 2+ same-class candidates could otherwise be "
         "meant and the user's own words identify which one via a spatial relation to you; "
         "never set relation directly"},
        ("person_named",),
        subsumes_locate_skills=("search_for_entity", "locate_entity"),
    ),
    # C.2 (2026-09-02): remember_person's generalization to any entity_tracks-
    # tracked class (person, chair, potted plant, ...), not just people --
    # "that plant is 小绿"/"remember this chair as my chair". Same real-
    # evidence contract (locates first, refuses if not genuinely found), plus
    # a same-class-name collision refusal and an already-bound-elsewhere
    # refusal neither of which remember_person itself needed to handle
    # (mc_embodied_skills/node.py's _remember_entity_skill has the details).
    "remember_entity": SkillSpec(
        "remember_entity",
        ("base", "gaze"),
        "Bind an alias to a specific physical entity (person, chair, plant, ...) -- "
        "e.g. 'that plant is 小绿'. If a prior locate_entity/search_for_entity/"
        "approach_entity result already resolved WHICH specific entity_id is meant "
        "(the normal case when multiple same-class entities, e.g. several people, are "
        "simultaneously visible -- pass that entity_id here, this is the ONLY way to "
        "bind 11/22/33-style aliases to specific different people unambiguously), pass "
        "`entity_id` and it is bound directly, no fresh look required. Otherwise pass "
        "`target` (a class name or description) and the robot actually locates it first "
        "(turning to look if needed); refuses to bind if it cannot find exactly one "
        "currently-tracked entity of that kind (2+ simultaneously visible -> ambiguous, "
        "use entity_id instead), or if the alias is already bound to a different entity. "
        "A THIRD way to disambiguate 2+ simultaneously visible same-class entities, "
        "alongside a prior locate's entity_id: if the user's own words identify which one "
        "by a spatial relation TO YOU, pass `reference_constraint_id` (never a bare "
        "`relation` string -- see the system prompt's reference_constraints section for "
        "the full contract) referencing an entry in context_json.reference_constraints "
        "instead -- a pre-extracted, trusted list computed before planning, never something "
        "you author yourself -- resolved deterministically against real robot "
        "position/orientation, never guessed.",
        {"target": "entity|object|person (class/description; omit if entity_id given)",
         "entity_id": "optional, exact entity_tracks id from a prior locate/search/approach result",
         "alias": "alias to bind",
         "reference_constraint_id": "optional: id of an entry in context_json."
         "reference_constraints -- only when 2+ same-class candidates could otherwise be "
         "meant and the user's own words "
         "identify which one via a spatial relation to you; never set relation directly"},
        ("entity_alias_bound",),
        subsumes_locate_skills=("search_for_entity", "locate_entity"),
    ),
    # FOUND LIVE 2026-08-31: "turn around and count everyone in the room" is one
    # reasonable request, but needing several separate simple_move turns to do it
    # blew the per-mission action budget outright (a real scan needs >=3 bounded
    # turns before a single detect/report action is even added). One bounded,
    # policy-counted skill for "sweep and count" instead of N base actions.
    "scan_room": SkillSpec(
        "scan_room",
        ("base", "gaze"),
        "Turn in place through a full sweep, counting DISTINCT people via stable "
        "cross-view identity tracking -- use for 'how many people are in the room' / "
        "'look around and count everyone'. observed_track_count is the number of "
        "distinct tracked identities confirmed during the sweep -- NOT a guaranteed-"
        "exact count in either direction: someone never detected during the sweep "
        "(occluded, out of view) is missed, and someone whose track is briefly lost "
        "and re-detected under a new identity could be double-counted. Takes no "
        "arguments.",
        {},
        ("room_scanned",),
    ),
    "guide_entity_to_place": SkillSpec(
        "guide_entity_to_place",
        ("base",),
        "Guide an arbitrary participant/entity to a configured place.",
        {"target": "entity|person", "place": "place|name"},
        ("entity_at_place",),
        realtime=True,
    ),
    "wait_for_participant": SkillSpec(
        "wait_for_participant",
        (),
        "Wait for a person matching an optional role to satisfy a condition in World "
        "State (present/hand_raised/hand_offered/facing_robot), polling until it "
        "happens or timeout_sec elapses. Does not itself move the robot.",
        {
            "role": "optional role name, matched against the person's roles[] "
            "(omit to match any visible person)",
            "condition": "one of present/hand_raised/hand_offered/facing_robot, "
            "default present",
        },
        ("participant_ready",),
    ),
    "say": SkillSpec(
        "say",
        ("voice", "face"),
        "Speak through the legacy TTS path.",
        {"text": "string"},
        ("say_submitted",),
        dispatch="voice",
    ),
    "request_human_confirmation": SkillSpec(
        "request_human_confirmation",
        (),
        "Ask an operator to approve or deny a bounded mission step.",
        {"prompt": "string"},
        ("human_confirmation",),
        dispatch="dedicated",
    ),
    "simple_move": SkillSpec(
        "simple_move",
        ("base",),
        "Bounded relative base movement primitive: move forward/backward by a distance in "
        "metres, or turn left/right in place by an angle in degrees, with no named target. "
        "This is what 'turn right'/'turn left'/'turn around' means -- use face_entity only "
        "when the intent names an actual person/object to turn toward.",
        {"action": "forward|backward|left|right", "value": "number"},
        ("relative_motion_completed",),
        dispatch="dedicated",
    ),
    "play_animation": SkillSpec(
        "play_animation",
        ("body",),
        _play_animation_description(),
        {"animation": "string"},
        ("animation_played",),
        dispatch="dedicated",
    ),
    "look_at": SkillSpec(
        "look_at",
        ("body",),
        "Compatibility wrapper for look_at_static direction controls.",
        {"direction": "direction"},
        # 2026-09-02: "animation_played" restored alongside its own more
        # specific predicate -- skill_adapters.py's dedicated dispatch for
        # look_at genuinely writes evidence["animation_played"]="look_at" on
        # execution (BootstrapPlanner/local_smoke.py's deterministic planner
        # relies on exactly this), so it is a real, checkable claim this
        # skill's execution evidence backs, not a looseness to remove. Found
        # the hard way: policy_guard.py's PolICY_TO_SKILLS derivation
        # (from result_predicates alone) initially omitted it and broke both
        # of those real callers -- result_predicates must list EVERY
        # predicate a skill's own evidence can actually satisfy, since it is
        # now the single source of truth for goal-alignment checking too.
        ("look_at_static_completed", "animation_played"),
        dispatch="dedicated",
    ),
    "point_at": SkillSpec(
        "point_at",
        ("body",),
        "Compatibility wrapper that points at an arbitrary target.",
        {"target": "object|place|x/y/z"},
        # Same reasoning as look_at above -- skill_adapters.py's point_at
        # dispatch also writes evidence["animation_played"]="point_at".
        ("point_at", "animation_played"),
        dispatch="dedicated",
    ),
    "come_to_me": SkillSpec(
        "come_to_me",
        ("base",),
        "Compatibility wrapper for navigating to the current interaction owner.",
        {},
        ("robot_near_interaction_owner",),
        dispatch="dedicated",
    ),
}


GENERIC_SKILL_NAMES = {name for name, spec in DEFAULT_SKILLS.items() if spec.dispatch != "voice"}


class SkillRegistry:
    def __init__(self, skills: dict[str, SkillSpec] | None = None) -> None:
        self._skills = dict(skills or DEFAULT_SKILLS)

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills.keys()))
