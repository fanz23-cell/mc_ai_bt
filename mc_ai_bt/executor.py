from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Protocol

from .visual_check import visual_check_goal_spec


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    message: str
    facts: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


class SkillExecutor(Protocol):
    def execute_skill(
        self,
        name: str,
        args: dict[str, Any],
        cancel_event: Event | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        ...


class CheckExecutor(Protocol):
    def check(self, goal_spec: dict[str, Any], execution: ExecutionResult) -> Any:
        ...


ProgressCallback = Callable[[str, float], None]


@dataclass
class _ProgressState:
    total: int
    completed: int = 0


class BtExecutor:
    """Small synchronous executor for validated bootstrap BTs.

    This is intentionally narrow. Long-running robot skills will move to
    awaitable adapters, but the first end-to-end route needs one safe action:
    say through the existing voice pipeline.
    """

    def execute_json(
        self,
        root_json: str,
        skills: SkillExecutor,
        cancel_event: Event | None = None,
        checks: CheckExecutor | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        try:
            root = json.loads(root_json)
        except json.JSONDecodeError as exc:
            return ExecutionResult(False, f"invalid bt_json: {exc}")
        return self.execute(
            root,
            skills,
            cancel_event=cancel_event,
            checks=checks,
            progress_callback=progress_callback,
        )

    def execute(
        self,
        node: dict[str, Any],
        skills: SkillExecutor,
        cancel_event: Event | None = None,
        checks: CheckExecutor | None = None,
        blackboard: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        _path: str = "root",
        _progress_state: _ProgressState | None = None,
    ) -> ExecutionResult:
        if progress_callback is not None and _progress_state is None:
            _progress_state = _ProgressState(max(1, _count_progress_leaves(node)))
        if cancel_event is not None and cancel_event.is_set():
            return ExecutionResult(False, "mission canceled")
        facts = dict(blackboard or {})
        node_type = node.get("type")
        if node_type == "Action":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: self._execute_action(node, skills, cancel_event),
                progress_callback,
                _progress_state,
            )
        if node_type == "Sequence":
            for idx, child in enumerate(node.get("children", [])):
                result = self.execute(
                    child,
                    skills,
                    cancel_event=cancel_event,
                    checks=checks,
                    blackboard=facts,
                    progress_callback=progress_callback,
                    _path=f"{_path}.children[{idx}]",
                    _progress_state=_progress_state,
                )
                facts.update(result.facts)
                if not result.success:
                    return ExecutionResult(False, result.message, facts, result.blocked)
            return ExecutionResult(True, "sequence succeeded", facts)
        if node_type == "Fallback":
            last = ExecutionResult(False, "fallback had no children")
            for idx, child in enumerate(node.get("children", [])):
                last = self.execute(
                    child,
                    skills,
                    cancel_event=cancel_event,
                    checks=checks,
                    blackboard=facts,
                    progress_callback=progress_callback,
                    _path=f"{_path}.children[{idx}]",
                    _progress_state=_progress_state,
                )
                facts.update(last.facts)
                if last.blocked:
                    return ExecutionResult(False, last.message, facts, True)
                if last.success:
                    return ExecutionResult(True, last.message, facts)
            return ExecutionResult(False, last.message, facts)
        if node_type == "Wait":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: self._execute_wait(node, cancel_event),
                progress_callback,
                _progress_state,
            )
        if node_type == "Retry":
            return self._execute_retry(
                node,
                skills,
                cancel_event,
                checks,
                facts,
                progress_callback,
                _progress_state,
                _path,
            )
        if node_type == "Condition":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: self._execute_condition(node, checks, facts),
                progress_callback,
                _progress_state,
            )
        if node_type == "GoalCheck":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: self._execute_goal_check(node, checks, facts),
                progress_callback,
                _progress_state,
            )
        if node_type == "VisualCheck":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: self._execute_visual_check(node, checks, facts),
                progress_callback,
                _progress_state,
            )
        return ExecutionResult(False, f"unsupported executor node type: {node_type}")

    def _execute_progress_leaf(
        self,
        label: str,
        run,
        progress_callback: ProgressCallback | None,
        progress_state: _ProgressState | None,
    ) -> ExecutionResult:
        if progress_callback is not None and progress_state is not None:
            progress_callback(label, _progress_fraction(progress_state))
        result = run()
        if progress_callback is not None and progress_state is not None:
            progress_state.completed += 1
            progress_callback(label, _progress_fraction(progress_state))
        return result

    def _execute_action(
        self,
        node: dict[str, Any],
        skills: SkillExecutor,
        cancel_event: Event | None,
    ) -> ExecutionResult:
        skill = node.get("skill")
        args = node.get("args") or {}
        if not isinstance(skill, str) or not skill:
            return ExecutionResult(False, "action skill is required")
        if not isinstance(args, dict):
            return ExecutionResult(False, "action args must be an object")
        timeout = _optional_positive_timeout(node)
        if isinstance(timeout, str):
            return ExecutionResult(False, timeout)
        return skills.execute_skill(skill, args, cancel_event, timeout_sec=timeout)

    def _execute_wait(
        self,
        node: dict[str, Any],
        cancel_event: Event | None,
    ) -> ExecutionResult:
        try:
            duration = float(node.get("duration_sec"))
        except (TypeError, ValueError):
            return ExecutionResult(False, "wait duration_sec must be numeric")
        if duration <= 0 or duration > 300:
            return ExecutionResult(False, "wait duration_sec must be in (0, 300]")
        if cancel_event is None:
            Event().wait(timeout=duration)
            return ExecutionResult(True, f"waited {duration:g}s")
        if cancel_event.wait(timeout=duration):
            return ExecutionResult(False, "mission canceled")
        return ExecutionResult(True, f"waited {duration:g}s")

    def _execute_retry(
        self,
        node: dict[str, Any],
        skills: SkillExecutor,
        cancel_event: Event | None,
        checks: CheckExecutor | None,
        blackboard: dict[str, Any],
        progress_callback: ProgressCallback | None,
        progress_state: _ProgressState | None,
        path: str,
    ) -> ExecutionResult:
        child = node.get("child")
        if not isinstance(child, dict):
            return ExecutionResult(False, "retry child must be an object")
        try:
            max_attempts = int(node.get("max_attempts"))
        except (TypeError, ValueError):
            return ExecutionResult(False, "retry max_attempts must be an integer")
        if max_attempts < 1 or max_attempts > 5:
            return ExecutionResult(False, "retry max_attempts must be in [1, 5]")

        facts: dict[str, Any] = dict(blackboard)
        last = ExecutionResult(False, "retry had no attempts")
        for attempt in range(1, max_attempts + 1):
            if cancel_event is not None and cancel_event.is_set():
                return ExecutionResult(False, "mission canceled", facts)
            last = self.execute(
                child,
                skills,
                cancel_event=cancel_event,
                checks=checks,
                blackboard=facts,
                progress_callback=progress_callback,
                _path=f"{path}.child[{attempt}]",
                _progress_state=progress_state,
            )
            facts.update(last.facts)
            if last.blocked:
                return ExecutionResult(False, last.message, facts, True)
            if last.success:
                return ExecutionResult(True, f"retry succeeded on attempt {attempt}", facts)
        return ExecutionResult(
            False,
            f"retry exhausted after {max_attempts} attempts: {last.message}",
            facts,
        )

    def _execute_condition(
        self,
        node: dict[str, Any],
        checks: CheckExecutor | None,
        facts: dict[str, Any],
    ) -> ExecutionResult:
        goal_spec = {
            "type": "structured",
            "predicate": node.get("predicate"),
            "args": node.get("args") or {},
            "verification": node.get("verification") or {"mode": "world_state"},
        }
        return _check_as_execution_result(checks, goal_spec, facts, label="condition")

    def _execute_goal_check(
        self,
        node: dict[str, Any],
        checks: CheckExecutor | None,
        facts: dict[str, Any],
    ) -> ExecutionResult:
        check = node.get("check")
        if not isinstance(check, dict):
            return ExecutionResult(False, "goal check requires check object")
        goal_spec = _normalise_goal_check_spec(check)
        return _check_as_execution_result(checks, goal_spec, facts, label="goal check")

    def _execute_visual_check(
        self,
        node: dict[str, Any],
        checks: CheckExecutor | None,
        facts: dict[str, Any],
    ) -> ExecutionResult:
        check = node.get("check")
        if not isinstance(check, dict):
            return ExecutionResult(False, "visual check requires check object")
        goal_spec = visual_check_goal_spec(check)
        if not goal_spec:
            return ExecutionResult(
                False,
                "visual check query is not supported by the local checker",
                facts,
                True,
            )
        return _check_as_execution_result(checks, goal_spec, facts, label="visual check")


def _check_as_execution_result(
    checks: CheckExecutor | None,
    goal_spec: dict[str, Any],
    facts: dict[str, Any],
    *,
    label: str,
) -> ExecutionResult:
    if checks is None:
        return ExecutionResult(False, f"{label} checker is not configured", facts, True)
    result = checks.check(goal_spec, ExecutionResult(True, f"{label} input", dict(facts)))
    state = _tri_state_value(getattr(result, "state", "UNKNOWN"))
    message = str(getattr(result, "message", ""))
    if state == "TRUE":
        return ExecutionResult(True, f"{label} TRUE: {message}", facts)
    if state == "FALSE":
        return ExecutionResult(False, f"{label} FALSE: {message}", facts)
    return ExecutionResult(False, f"{label} UNKNOWN: {message}", facts, True)


def _normalise_goal_check_spec(check: dict[str, Any]) -> dict[str, Any]:
    if "type" in check:
        return check
    return {
        "type": "structured",
        "predicate": check.get("predicate"),
        "args": check.get("args") or {},
        "verification": check.get("verification") or {"mode": "world_state"},
        "summary": check.get("summary", ""),
    }


def _tri_state_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _optional_positive_timeout(node: dict[str, Any]) -> float | None | str:
    raw = node.get("timeout_sec")
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or raw <= 0 or raw > 600:
        return "action timeout_sec must be in (0, 600]"
    return float(raw)


def _count_progress_leaves(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    node_type = node.get("type")
    if node_type in {"Action", "Wait", "Condition", "GoalCheck", "VisualCheck"}:
        return 1
    if node_type in {"Sequence", "Fallback"}:
        return sum(_count_progress_leaves(child) for child in node.get("children", []) or [])
    if node_type == "Retry":
        try:
            attempts = int(node.get("max_attempts", 1) or 1)
        except (TypeError, ValueError):
            attempts = 1
        return max(1, attempts) * _count_progress_leaves(node.get("child"))
    return 0


def _node_label(node: dict[str, Any], path: str) -> str:
    node_type = str(node.get("type") or "Node")
    if node_type == "Action":
        skill = str(node.get("skill") or "")
        return f"{path}:Action:{skill}" if skill else f"{path}:Action"
    if node_type == "Condition":
        predicate = str(node.get("predicate") or "")
        return f"{path}:Condition:{predicate}" if predicate else f"{path}:Condition"
    if node_type in {"GoalCheck", "VisualCheck"}:
        check = node.get("check") if isinstance(node.get("check"), dict) else {}
        predicate = str(check.get("predicate") or check.get("query") or "")
        return f"{path}:{node_type}:{predicate}" if predicate else f"{path}:{node_type}"
    return f"{path}:{node_type}"


def _progress_fraction(progress_state: _ProgressState) -> float:
    if progress_state.total <= 0:
        return 0.0
    return max(0.0, min(1.0, progress_state.completed / progress_state.total))
