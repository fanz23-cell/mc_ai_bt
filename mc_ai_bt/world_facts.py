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

    return tuple(updates)
