from __future__ import annotations

import json
import threading
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
    # True only for the subset of `blocked` failures that are a genuine
    # evidentiary/semantic gap -- a Condition/GoalCheck/VisualCheck node came
    # back UNKNOWN -- as opposed to a mechanical dead-end (a Timeout expired,
    # a Parallel child never reported, a checker isn't wired up). node.py's
    # _run_mission uses this to decide whether a mission that hits this
    # should pause (resumable once new evidence arrives; see mission.py's
    # pause/resume) or terminate as BLOCKED (a plain retry/replan won't help
    # a config error or an expired timeout the way it can resolve missing
    # evidence). See OMEGACLAW_AI_BT_INTEGRATION.md §3.5.
    needs_decision: bool = False


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
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


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
                    return ExecutionResult(
                        False, result.message, facts, result.blocked, result.needs_decision)
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
                    return ExecutionResult(False, last.message, facts, True, last.needs_decision)
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
        if node_type == "Parallel":
            return self._execute_parallel(
                node,
                skills,
                cancel_event,
                checks,
                facts,
                progress_callback,
                _progress_state,
                _path,
            )
        if node_type == "Timeout":
            return self._execute_timeout(
                node,
                skills,
                cancel_event,
                checks,
                facts,
                progress_callback,
                _progress_state,
                _path,
            )
        if node_type == "NoAction":
            return self._execute_progress_leaf(
                _node_label(node, _path),
                lambda: ExecutionResult(True, str(node.get("reason") or "no action"), facts),
                progress_callback,
                _progress_state,
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
            with progress_state.lock:
                start_progress = _progress_fraction(progress_state)
            progress_callback(label, start_progress)
        result = run()
        if progress_callback is not None and progress_state is not None:
            with progress_state.lock:
                progress_state.completed += 1
                end_progress = _progress_fraction(progress_state)
            progress_callback(label, end_progress)
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
                return ExecutionResult(False, last.message, facts, True, last.needs_decision)
            if last.success:
                return ExecutionResult(True, f"retry succeeded on attempt {attempt}", facts)
        return ExecutionResult(
            False,
            f"retry exhausted after {max_attempts} attempts: {last.message}",
            facts,
        )

    def _execute_parallel(
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
        children = node.get("children")
        if not isinstance(children, list) or not children:
            return ExecutionResult(False, "parallel children must be a non-empty list")
        cancel_on_failure = bool(node.get("cancel_on_failure", True))
        child_cancel = _LinkedCancelEvent(cancel_event)
        results: list[ExecutionResult | None] = [None] * len(children)

        def _run(idx: int, child: dict[str, Any]) -> None:
            result = self.execute(
                child,
                skills,
                cancel_event=child_cancel,
                checks=checks,
                blackboard=dict(blackboard),
                progress_callback=progress_callback,
                _path=f"{path}.children[{idx}]",
                _progress_state=progress_state,
            )
            results[idx] = result
            if cancel_on_failure and not result.success:
                child_cancel.set()

        threads = [
            threading.Thread(target=_run, args=(idx, child), name=f"bt-parallel-{idx}", daemon=True)
            for idx, child in enumerate(children)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        facts: dict[str, Any] = dict(blackboard)
        failures: list[ExecutionResult] = []
        blocked = False
        needs_decision = False
        for result in results:
            if result is None:
                # A child thread that never reported is a bug/race, not an
                # evidentiary gap -- needs_decision deliberately NOT set.
                failures.append(ExecutionResult(False, "parallel child did not report a result", {}, True))
                blocked = True
                continue
            facts.update(result.facts)
            if not result.success:
                failures.append(result)
                blocked = blocked or result.blocked
                needs_decision = needs_decision or result.needs_decision
        if failures:
            message = "; ".join(result.message for result in failures)
            return ExecutionResult(False, f"parallel failed: {message}", facts, blocked, needs_decision)
        return ExecutionResult(True, "parallel succeeded", facts)

    def _execute_timeout(
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
            return ExecutionResult(False, "timeout child must be an object")
        try:
            timeout_sec = float(node.get("timeout_sec"))
        except (TypeError, ValueError):
            return ExecutionResult(False, "timeout timeout_sec must be numeric")
        if timeout_sec <= 0 or timeout_sec > 600:
            return ExecutionResult(False, "timeout timeout_sec must be in (0, 600]")
        child_cancel = _LinkedCancelEvent(cancel_event)
        box: dict[str, ExecutionResult] = {}

        def _run() -> None:
            box["result"] = self.execute(
                child,
                skills,
                cancel_event=child_cancel,
                checks=checks,
                blackboard=dict(blackboard),
                progress_callback=progress_callback,
                _path=f"{path}.child",
                _progress_state=progress_state,
            )

        thread = threading.Thread(target=_run, name="bt-timeout-child", daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            child_cancel.set()
            return ExecutionResult(False, f"timeout after {timeout_sec:g}s", dict(blackboard), True)
        return box.get("result", ExecutionResult(False, "timeout child produced no result", dict(blackboard), True))

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
        if goal_spec and goal_spec.get("type") != "visual":
            structured = _check_as_execution_result(checks, goal_spec, facts, label="visual check")
            if not structured.blocked:
                return structured

        visual_check = getattr(checks, "visual_check", None)
        if callable(visual_check):
            return _visual_check_as_execution_result(visual_check, check, facts)
        return ExecutionResult(False, "visual check UNKNOWN: VisualCheck service is not configured", facts, True)


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
    return ExecutionResult(False, f"{label} UNKNOWN: {message}", facts, True, needs_decision=True)


def _visual_check_as_execution_result(
    visual_check,
    check: dict[str, Any],
    facts: dict[str, Any],
) -> ExecutionResult:
    result = visual_check(check, facts)
    state = _tri_state_value(getattr(result, "state", "UNKNOWN"))
    message = str(getattr(result, "message", ""))
    if state == "TRUE":
        return ExecutionResult(True, f"visual check TRUE: {message}", facts)
    if state == "FALSE":
        return ExecutionResult(False, f"visual check FALSE: {message}", facts)
    return ExecutionResult(
        False, f"visual check UNKNOWN: {message}", facts, True, needs_decision=True)


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
    if node_type in {"Action", "Wait", "Condition", "GoalCheck", "VisualCheck", "NoAction"}:
        return 1
    if node_type in {"Sequence", "Fallback", "Parallel"}:
        return sum(_count_progress_leaves(child) for child in node.get("children", []) or [])
    if node_type == "Retry":
        try:
            attempts = int(node.get("max_attempts", 1) or 1)
        except (TypeError, ValueError):
            attempts = 1
        return max(1, attempts) * _count_progress_leaves(node.get("child"))
    if node_type == "Timeout":
        return _count_progress_leaves(node.get("child"))
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


class _LinkedCancelEvent:
    def __init__(self, parent: Event | None = None) -> None:
        self._local = Event()
        self._parent = parent

    def is_set(self) -> bool:
        return self._local.is_set() or (self._parent is not None and self._parent.is_set())

    def set(self) -> None:
        self._local.set()

    def wait(self, timeout: float | None = None) -> bool:
        step = 0.05
        if timeout is None:
            while not self.is_set():
                self._local.wait(step)
            return True
        remaining = max(0.0, float(timeout))
        while remaining > 0:
            if self.is_set():
                return True
            slice_sec = min(step, remaining)
            self._local.wait(slice_sec)
            remaining -= slice_sec
        return self.is_set()
