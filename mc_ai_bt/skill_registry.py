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


DEFAULT_SKILLS: dict[str, SkillSpec] = {
    "locate_entity": SkillSpec(
        "locate_entity",
        (),
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
        (),
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
        ("gaze",),
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
    ),
    "move_along_axis": SkillSpec(
        "move_along_axis",
        ("right_arm",),
        "Move a controlled frame along an axis by a bounded distance.",
        {"axis": "axis", "distance_m": "number"},
        ("axis_motion_completed",),
        realtime=True,
    ),
    "maintain_distance": SkillSpec(
        "maintain_distance",
        ("base",),
        "Maintain a bounded distance to a realtime entity.",
        {"target": "entity|person|object", "distance_m": "number"},
        ("distance_maintained",),
        realtime=True,
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
    ),
    "detect_contact": SkillSpec(
        "detect_contact",
        ("right_arm",),
        "Read generic contact evidence without interpreting task success.",
        {"target": "entity|object|person"},
        ("contact_detected",),
        realtime=True,
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
        "place applies. Fails if the entity has not been perceived recently.",
        {"target": "entity|entity_id|object|person"},
        ("entity_approached",),
    ),
    "face_entity": SkillSpec(
        "face_entity",
        ("base",),
        "Rotate in place (no travel) to face an entity's current live position. A single "
        "bounded turn computed once from a snapshot -- does not continuously track a moving "
        "target; call again to re-aim.",
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
        "Bounded relative base movement primitive.",
        {"action": "forward|backward|left|right", "value": "number"},
        ("relative_motion_completed",),
        dispatch="dedicated",
    ),
    "play_animation": SkillSpec(
        "play_animation",
        ("body",),
        "Play a named animator clip or generator.",
        {"animation": "string"},
        ("animation_played",),
        dispatch="dedicated",
    ),
    "look_at": SkillSpec(
        "look_at",
        ("body",),
        "Compatibility wrapper for look_at_static direction controls.",
        {"direction": "direction"},
        ("look_at_static_completed",),
        dispatch="dedicated",
    ),
    "point_at": SkillSpec(
        "point_at",
        ("body",),
        "Compatibility wrapper that points at an arbitrary target.",
        {"target": "object|place|x/y/z"},
        ("point_at",),
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
