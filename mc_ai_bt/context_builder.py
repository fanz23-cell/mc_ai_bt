from __future__ import annotations

import json
from typing import Any, Callable

from .mission import Mission
from .policy_guard import POLICY_ENABLED_SKILLS, PolicyLimits
from .reference_extraction import extract_reference_constraints
from .skill_registry import SkillRegistry


SnapshotProvider = Callable[[tuple[str, ...], float], str]


DEFAULT_SCOPES = ("navigation", "people", "objects", "robot", "tasks")



def effective_execution_intent(mission: Mission, caller_context: dict) -> str:
    """The one text every execution-semantics consumer must plan against.

    FOUND LIVE 2026-09-05 (E1 TURN2, four attempts): the same six-character
    request, "去33那里。", reached the planner as four different sentences,
    because Omega composes its own intent_text. The fourth did not merely
    reword it -- it ADDED a physical goal the user never asked for ("stop when
    you are directly in front of them, facing them"), and the planner duly
    added locate_entity and face_entity. PolicyGuard refused, correctly: that
    is upstream task expansion, not a planning bug.

    So authority over what the robot was asked to do is decided here, once:

        user-originated mission   -> the user's own words are authoritative
        Omega-originated mission  -> Omega's intent_text is authoritative

    Omega keeps everything that is genuinely its job -- knowing who 33 is,
    recalling preferences, deciding whether to involve the robot at all,
    generating its own missions later. What it does not get is the ability to
    quietly widen a physical instruction while relaying it.

    Which case applies is not guessed here: the Bridge only carries
    user_utterance when its own per-turn reply stream was open, i.e. when Omega
    was answering that utterance (see that repo's last_user_utterance).

    The Bridge's machine-owned user_utterance when it carried one (see that
    repo's RobotGatewayBridge._recent_user_utterance), otherwise the submitted
    intent_text. Never both: a span quoted from one text cannot be checked
    against the other, and mixing them would let a constraint quoted from the
    user's words be validated against a rewrite that never contained them."""
    if isinstance(caller_context, dict):
        raw = caller_context.get("user_utterance")
        if isinstance(raw, str) and raw.strip():
            return raw
    return mission.intent_text


class ContextBuilder:
    """Build the bounded planner context passed to the BT planner."""

    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider | None = None,
        skill_registry: SkillRegistry | None = None,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        max_age_sec: float = 5.0,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._skills = skill_registry or SkillRegistry()
        self._limits = PolicyLimits()
        self._scopes = scopes
        self._max_age_sec = max_age_sec

    def build_json(self, mission: Mission, missions: tuple[Mission, ...]) -> str:
        caller_context = _safe_json_object(mission.context_json)
        context = {
            "schema": "mc_ai_bt.context.v1",
            "mission": {
                "mission_id": mission.identity.mission_id,
                "source": mission.source,
                "operator_id": mission.operator_id,
                "priority": mission.priority,
                "intent_text": mission.intent_text,
            },
            "caller_context": caller_context,
            "world": self._snapshot(),
            # FOUND LIVE 2026-09-03 (identity/grounding foundation live
            # verification): this used to serialize EVERY mission this node
            # has ever handled since its last cold start, unbounded --
            # MissionManager.all() is replayed from the persisted mission
            # journal on startup, so it never actually shrinks, even across
            # a redeploy. 204 accumulated missions this session alone were
            # enough to push a real gpt-4o-mini planner call over its
            # 128000-token limit (confirmed live: OpenAIContextOverflowError,
            # blocking every subsequent mission on the live system, not
            # just a test). Grep-confirmed nothing downstream -- planner.py's
            # prompt, goal_check.py, anywhere else -- ever reads
            # context.missions at all (task_projection.py has its own,
            # separate, differently-purposed missions field for the
            # monitor/status API). A first fix bounded this to the most
            # recent 10; GPT's re-review pointed out that "last 10" was
            # still an arbitrary middle ground for a field NOTHING reads --
            # genuinely empty is the honest reflection of "no consumer
            # exists today". If a future planner backend wants short
            # recent-history context, that is E2/Omega retrieval's job
            # (targeted, relevant summaries) not an ever-larger raw dump
            # here. `missions` param is accepted for interface stability
            # but intentionally unused.
            "missions": [],
            # Identity/Grounding foundation finalization (2026-09-03, GPT
            # re-review): a deterministic, TRUSTED menu of reference
            # constraints extracted from intent_text BEFORE the planner
            # ever runs -- see reference_extraction.py's own module
            # docstring for why this replaced letting the planner author
            # its own constraints. The planner may reference an entry here
            # by constraint_id (remember_person/remember_entity's
            # `reference_constraint_id` arg); it must never invent one of
            # its own.
            # FOUND LIVE 2026-09-05 (E1 TURN2): extract from the USER's own
            # words when the Bridge carried them, not from Omega's rewrite.
            # Omega composes its own intent_text, and three consecutive runs of
            # "去33那里。" arrived as three different sentences, each producing a
            # differently-shaped plan. Where provenance matters -- which entity
            # was meant, what name was given -- a paraphrase is not evidence of
            # what the user said.
            #
            # reference_source_text records WHICH text these came from, because
            # planning_pipeline's guard re-checks each source_span against it:
            # a constraint quoted from the user's words must be validated
            # against those words, not against Omega's rewrite of them.
            # Falls back to intent_text whenever the Bridge offered nothing
            # (Omega-initiated missions have no user utterance behind them).
            "reference_constraints": extract_reference_constraints(
                effective_execution_intent(mission, caller_context)),
            # The text every execution-semantics consumer plans against, plus
            # enough audit trail to see afterwards what each side contributed.
            # mission.intent_text above is deliberately left as Omega submitted
            # it -- overwriting it would erase the evidence of what Omega
            # actually proposed.
            "effective_execution_intent": effective_execution_intent(mission, caller_context),
            "effective_intent_source": (
                "user" if effective_execution_intent(mission, caller_context) != mission.intent_text
                else "omega"),
            "omega_submitted_intent": mission.intent_text,
            "skills": [
                {
                    "name": spec.name,
                    "resources": list(spec.resources),
                    "description": spec.description,
                    "policy_enabled": spec.name in POLICY_ENABLED_SKILLS,
                }
                for spec in (self._skills.get(name) for name in self._skills.names())
                # FOUND LIVE 2026-08-31: a skill registered with status="blocked"
                # (align_axis/move_along_axis/maintain_distance/wait_for_contact/
                # detect_contact today) unconditionally returns STATUS_BLOCKED at the
                # embodied-skills runtime -- it must never be advertised as something
                # the planner can actually use.
                if spec.status == "available"
            ],
            "constraints": {
                "planner_output_schema": "mc_ai_bt.plan.v1",
                "allowed_top_level_keys": ["schema", "root", "goal_spec", "context_json"],
                "unknown_is_not_false": True,
                "embodied_skills_require_resource_lease": True,
                "max_total_nodes": self._limits.max_total_nodes,
                "max_total_actions": self._limits.max_total_actions,
                "max_physical_actions": self._limits.max_physical_actions,
                "max_base_actions": self._limits.max_base_actions,
                "max_body_actions": self._limits.max_body_actions,
                "max_visual_checks": self._limits.max_visual_checks,
                "max_conditions": self._limits.max_conditions,
            },
        }
        return json.dumps(context, sort_keys=True, separators=(",", ":"))

    def _snapshot(self) -> dict[str, Any]:
        if self._snapshot_provider is None:
            return {}
        try:
            raw = self._snapshot_provider(self._scopes, self._max_age_sec)
        except Exception:
            return {}
        return _safe_json_object(raw)


def _safe_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
