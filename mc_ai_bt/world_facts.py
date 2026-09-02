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

    # FOUND LIVE 2026-08-31: naming a person used to write nothing at all (no skill
    # existed for it) -- remember_person only ever produces this fact after actually
    # locating the person (see mc_embodied_skills/node.py's _remember_person_skill),
    # so a name reaching mc_world_state here is backed by real, fresh perception, not
    # bare chat text. A separate scope from "people"/"semantic_people" on purpose: a
    # name binding should not be clobbered by the next unrelated pose-detection frame.
    person_named = facts.get("person_named")
    if isinstance(person_named, dict) and person_named.get("person_id") and person_named.get("name"):
        person_id = str(person_named["person_id"])
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.remember_person",
                scope="people_names",
                key=person_id,
                value={"person_id": person_id, "name": str(person_named["name"])},
                merge=False,
            )
        )

    # C.2 (2026-09-02): the real, read alias-binding path -- mc_embodied_skills'
    # remember_entity (and remember_person, now a thin wrapper around it) only
    # ever produces this fact after resolving to exactly one currently-ACTIVE
    # entity_tracks candidate and checking for a conflicting existing binding
    # (see mc_embodied_skills/node.py's _remember_entity_skill) -- this write
    # is just recording that already-verified decision. A separate scope from
    # "entity_tracks" on purpose, same reasoning "people_names" (this scope's
    # now-unused predecessor) was kept separate from "people": a name binding
    # must not be clobbered by the next unrelated detection update.
    entity_alias_bound = facts.get("entity_alias_bound")
    if (
        isinstance(entity_alias_bound, dict)
        and entity_alias_bound.get("alias")
        and entity_alias_bound.get("entity_id")
    ):
        alias = str(entity_alias_bound["alias"])
        updates.append(
            WorldFactUpdate(
                source="mc_ai_bt.skill.remember_entity",
                scope="entity_aliases",
                key=alias,
                value={
                    "alias": alias,
                    "entity_id": str(entity_alias_bound["entity_id"]),
                    "entity_class": str(entity_alias_bound.get("entity_class") or ""),
                    "created_by": str(entity_alias_bound.get("created_by") or ""),
                },
                merge=False,
            )
        )

    return tuple(updates)
