from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorldFactUpdate:
    source: str
    scope: str
    key: str
    value: Any
    merge: bool = False


def world_fact_updates_for_execution(facts: dict[str, Any]) -> tuple[WorldFactUpdate, ...]:
    updates: list[WorldFactUpdate] = []

    robot_at_place = str(facts.get("robot_at_place") or "").strip()
    if robot_at_place:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.go_to_place",
                scope="navigation",
                key="current_place",
                value=robot_at_place,
            )
        )

    if facts.get("robot_near_interaction_owner") is True:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.come_to_me",
                scope="robot",
                key="near_interaction_owner",
                value=True,
            )
        )

    relative_motion = facts.get("relative_motion_completed")
    if isinstance(relative_motion, dict) and relative_motion:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.simple_move",
                scope="robot",
                key="last_relative_motion",
                value=dict(relative_motion),
                merge=False,
            )
        )

    animation = str(facts.get("animation_played") or "").strip()
    if animation:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.animation",
                scope="robot",
                key="last_animation",
                value=animation,
            )
        )

    said = str(facts.get("say_submitted") or "").strip()
    if said:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.say",
                scope="robot",
                key="last_utterance",
                value=said,
            )
        )

    confirmation = facts.get("human_confirmation")
    if isinstance(confirmation, dict) and confirmation:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.human_confirmation",
                scope="tasks",
                key="last_human_confirmation",
                value=dict(confirmation),
                merge=False,
            )
        )

    for key in (
        "reach_completed",
        "contact_detected",
        "target_state_changed",
        "axis_aligned",
        "axis_motion_completed",
        "distance_maintained",
        "pose_held",
        "oscillation_completed",
        "retracted",
    ):
        value = facts.get(key)
        if isinstance(value, dict) and value:
            updates.append(
                WorldFactUpdate(
                    source="mc_ai_bt.skill.embodied",
                    scope="robot",
                    key=key,
                    value=dict(value),
                    merge=False,
                )
            )

    following = facts.get("entity_following") or facts.get("person_following")
    if isinstance(following, dict) and following:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.follow_entity",
                scope="entities",
                key="following",
                value=dict(following),
                merge=False,
            )
        )

    entity_at_place = facts.get("entity_at_place") or facts.get("person_at_place")
    if isinstance(entity_at_place, dict) and entity_at_place:
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.guide_entity_to_place",
                scope="entities",
                key="entity_at_place",
                value=dict(entity_at_place),
                merge=False,
            )
        )

    return tuple(updates)
