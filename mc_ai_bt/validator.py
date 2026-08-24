from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .skill_registry import SkillRegistry


EXECUTABLE_NODE_TYPES = {
    "Sequence",
    "Fallback",
    "Condition",
    "Action",
    "GoalCheck",
    "VisualCheck",
    "Wait",
    "Retry",
}
FUTURE_NODE_TYPES: set[str] = set()
ALLOWED_NODE_TYPES = EXECUTABLE_NODE_TYPES | FUTURE_NODE_TYPES
ALLOWED_GOAL_TYPES = {"structured", "visual", "human", "hybrid"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


class PlanValidator:
    def __init__(self, skill_registry: SkillRegistry | None = None) -> None:
        self._skills = skill_registry or SkillRegistry()

    def validate_json(self, plan_json: str) -> ValidationResult:
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return ValidationResult(False, (f"invalid json: {exc}",))
        return self.validate(plan)

    def validate(self, plan: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(plan, dict):
            return ValidationResult(False, ("plan must be an object",))

        root = plan.get("root")
        goal_spec = plan.get("goal_spec")
        if root is None:
            errors.append("plan.root is required")
        else:
            self._validate_node(root, "root", errors)
        if goal_spec is None:
            errors.append("plan.goal_spec is required")
        else:
            self._validate_goal_spec(goal_spec, errors)
        return ValidationResult(not errors, tuple(errors))

    def _validate_node(self, node: Any, path: str, errors: list[str]) -> None:
        if not isinstance(node, dict):
            errors.append(f"{path} must be an object")
            return

        node_type = node.get("type")
        if node_type not in ALLOWED_NODE_TYPES:
            errors.append(f"{path}.type is not allowed: {node_type!r}")
            return
        if node_type in FUTURE_NODE_TYPES:
            errors.append(f"{path}.type is planned but not executable yet: {node_type!r}")
            return

        timeout = node.get("timeout_sec")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 600):
            errors.append(f"{path}.timeout_sec must be in (0, 600]")

        if node_type in {"Sequence", "Fallback"}:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                errors.append(f"{path}.children must be a non-empty list")
                return
            for idx, child in enumerate(children):
                self._validate_node(child, f"{path}.children[{idx}]", errors)
            return

        if node_type == "Retry":
            max_attempts = node.get("max_attempts")
            if not isinstance(max_attempts, int) or max_attempts < 1 or max_attempts > 5:
                errors.append(f"{path}.max_attempts must be an integer in [1, 5]")
            child = node.get("child")
            if child is None:
                errors.append(f"{path}.child is required")
            else:
                self._validate_node(child, f"{path}.child", errors)
            return

        if node_type == "Condition":
            if not isinstance(node.get("predicate"), str) or not node["predicate"].strip():
                errors.append(f"{path}.predicate is required")
            args = node.get("args", {})
            if not isinstance(args, dict):
                errors.append(f"{path}.args must be an object")
            return

        if node_type == "Action":
            skill = node.get("skill")
            if not isinstance(skill, str) or not skill.strip():
                errors.append(f"{path}.skill is required")
            elif not self._skills.has(skill):
                errors.append(f"{path}.skill is unknown: {skill}")
            args = node.get("args", {})
            if not isinstance(args, dict):
                errors.append(f"{path}.args must be an object")
            return

        if node_type == "Wait":
            duration = node.get("duration_sec")
            if not isinstance(duration, (int, float)) or duration <= 0 or duration > 300:
                errors.append(f"{path}.duration_sec must be in (0, 300]")
            return

        if node_type == "GoalCheck":
            check = node.get("check")
            if not isinstance(check, dict):
                errors.append(f"{path}.check must be an object")
            elif not _looks_like_goal_check(check):
                errors.append(f"{path}.check must include a goal type or predicate")
            return

        if node_type == "VisualCheck":
            check = node.get("check")
            if not isinstance(check, dict):
                errors.append(f"{path}.check must be an object")
            elif not _looks_like_visual_check(check):
                errors.append(f"{path}.check must include a query, goal type, or predicate")
            return

    def _validate_goal_spec(self, goal_spec: Any, errors: list[str]) -> None:
        if not isinstance(goal_spec, dict):
            errors.append("goal_spec must be an object")
            return
        goal_type = goal_spec.get("type")
        if goal_type not in ALLOWED_GOAL_TYPES:
            errors.append(f"goal_spec.type is not allowed: {goal_type!r}")
        verification = goal_spec.get("verification")
        if not isinstance(verification, dict) or not verification:
            errors.append("goal_spec.verification must be a non-empty object")
        if goal_type == "structured" and "predicate" not in goal_spec:
            errors.append("structured goal_spec requires predicate")
        if goal_type == "visual" and "query" not in goal_spec:
            errors.append("visual goal_spec requires query")


def _looks_like_goal_check(check: dict[str, Any]) -> bool:
    if isinstance(check.get("type"), str) and check["type"].strip():
        return True
    return isinstance(check.get("predicate"), str) and check["predicate"].strip()


def _looks_like_visual_check(check: dict[str, Any]) -> bool:
    if _looks_like_goal_check(check):
        return True
    return isinstance(check.get("query"), str) and check["query"].strip()
