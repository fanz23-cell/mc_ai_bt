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

    # Section 7 physical verification producer (2026-09-13, doc 61/70):
    # reach_completed/retracted/pose_held/axis_aligned/axis_motion_completed/
    # oscillation_completed are now written AUTHORITATIVELY, straight to this
    # exact same (scope="robot", key=<predicate>) address, by
    # mc_embodied_skills itself (RosActionSkillProvider._write_skill_evidence_
    # facts) the instant the underlying ROS action genuinely succeeds -- with
    # semantic_verified forced True, a real claim, not the skill's own
    # self-report. `facts` here is that SAME skill's own returned evidence
    # dict, which still (honestly) carries semantic_verified=False for these
    # keys (see _reach_to and friends' own hardcoded False) -- echoing it
    # here, AFTER mc_embodied_skills' own producer already ran, with
    # merge=False, would silently CLOBBER the real, verified fact back to an
    # unverified one on every single successful call. These six are
    # deliberately EXCLUDED from this echo loop for that reason.
    #
    # contact_detected/target_state_changed/distance_maintained have no
    # independent producer (no force/torque/contact sensor on this robot;
    # target-state and distance-maintained claims are unimplemented) -- for
    # those three, echoing the skill's own self-report here is still the only
    # source there is, unchanged from before this fix, and still correctly
    # stays semantic_verified=False (UNKNOWN downstream), never a regression.
    for key in (
        "contact_detected",
        "target_state_changed",
        "distance_maintained",
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

    # C.2/C.3 (2026-09-02) -- entity_alias_bound is deliberately NOT handled
    # here anymore (2026-09-03 architecture consolidation). It used to become
    # a generic WorldFactUpdate on the "entity_aliases" scope, indistinguishable
    # from any other plain fact write -- but binding an alias to a live
    # perception entity is a domain command with real identity invariants
    # (collision detection, idempotency, minting a durable semantic identity
    # that survives the live entity's own id later changing -- see
    # mc_world_state/entity_identity.py's own module docstring for why a
    # generic scope/key/value write cannot express this). skill_adapters.py's
    # _publish_success_facts now calls WorldStateWriter.bind_entity_alias
    # directly for this one fact, never routing it through the generic
    # updates this function returns.

    return tuple(updates)
