from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .context_builder import ContextBuilder
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
    before this fix: replanning/rejecting beats guessing wrong."""
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
