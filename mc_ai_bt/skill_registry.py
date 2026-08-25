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


GENERIC_SKILL_NAMES = {
    "locate_entity",
    "track_entity",
    "track_frame",
    "get_pose",
    "check_relation",
    "search_for_entity",
    "look_at_static",
    "track_with_gaze",
    "reach_to",
    "align_axis",
    "move_along_axis",
    "maintain_distance",
    "hold_pose",
    "wait_for_contact",
    "detect_contact",
    "oscillate",
    "retract",
    "go_to_place",
    "follow_entity",
    "guide_entity_to_place",
    "wait_for_participant",
    "say",
    "play_animation",
}


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
        "Track an arbitrary entity from realtime perception.",
        {"target": "entity|entity_id|object|person"},
        ("track_entity_completed",),
        realtime=True,
    ),
    "track_frame": SkillSpec(
        "track_frame",
        ("gaze",),
        "Track a ROS frame from realtime TF/perception.",
        {"frame": "frame|target_frame"},
        ("track_completed",),
        realtime=True,
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
        "Continuously keep gaze on a realtime target.",
        {"target": "entity|frame"},
        ("track_with_gaze_completed",),
        realtime=True,
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
    ),
    "follow_entity": SkillSpec(
        "follow_entity",
        ("base",),
        "Continuously follow an arbitrary realtime entity.",
        {"target": "entity|person|object"},
        ("entity_following",),
        realtime=True,
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
        "Wait for a participant/role to become available in World State.",
        {"participant": "role|entity"},
        ("participant_ready",),
    ),
    "say": SkillSpec(
        "say",
        ("voice", "face"),
        "Speak through the legacy TTS path.",
        {"text": "string"},
        ("say_submitted",),
    ),
    "request_human_confirmation": SkillSpec(
        "request_human_confirmation",
        (),
        "Ask an operator to approve or deny a bounded mission step.",
        {"prompt": "string"},
        ("human_confirmation",),
    ),
    "simple_move": SkillSpec(
        "simple_move",
        ("base",),
        "Bounded relative base movement primitive.",
        {"action": "forward|backward|left|right", "value": "number"},
        ("relative_motion_completed",),
    ),
    "play_animation": SkillSpec(
        "play_animation",
        ("body",),
        "Play a named animator clip or generator.",
        {"animation": "string"},
        ("animation_played",),
    ),
    "look_at": SkillSpec(
        "look_at",
        ("body",),
        "Compatibility wrapper for look_at_static direction controls.",
        {"direction": "direction"},
        ("look_at_static_completed",),
    ),
    "point_at": SkillSpec(
        "point_at",
        ("body",),
        "Compatibility wrapper that points at an arbitrary target.",
        {"target": "object|place|x/y/z"},
        ("point_at",),
    ),
    "come_to_me": SkillSpec(
        "come_to_me",
        ("base",),
        "Compatibility wrapper for navigating to the current interaction owner.",
        {},
        ("robot_near_interaction_owner",),
    ),
}


class SkillRegistry:
    def __init__(self, skills: dict[str, SkillSpec] | None = None) -> None:
        self._skills = dict(skills or DEFAULT_SKILLS)

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills.keys()))
