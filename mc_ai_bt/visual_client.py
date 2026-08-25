from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import Any

from action_msgs.msg import GoalStatus
from mc_one.action import VisualCheck
from mc_one.msg import VisualCheckResult
from rclpy.action import ActionClient
from rclpy.node import Node

from .goal_check import CheckResult, TriState
from .identity import Identity
from .ros_identity import identity_to_msg


@dataclass(frozen=True)
class VisualCheckSettings:
    action_name: str = "/mc_multimodal/visual_check"
    camera_source: str = "head"
    max_age_sec: float = 2.0
    timeout_sec: float = 15.0
    expected_schema_version: str = "mc_multimodal.visual_check.v1"


class VisualCheckClient:
    def __init__(
        self,
        node: Node,
        *,
        settings: VisualCheckSettings | None = None,
        callback_group=None,
    ) -> None:
        self._node = node
        self._settings = settings or VisualCheckSettings()
        self._client = ActionClient(
            node,
            VisualCheck,
            self._settings.action_name,
            callback_group=callback_group,
        )

    def check(
        self,
        *,
        identity: Identity,
        check: dict[str, Any],
        facts: dict[str, Any],
        cancel_event: Event | None = None,
    ) -> CheckResult:
        if not _action_server_ready(self._client, timeout_sec=1.0):
            return CheckResult(TriState.UNKNOWN, "VisualCheck action server is not ready")

        goal = VisualCheck.Goal()
        goal.identity = identity_to_msg(identity)
        goal.query = str(check.get("query") or check.get("summary") or "").strip()
        if not goal.query:
            predicate = str(check.get("predicate") or "")
            goal.query = _query_from_predicate(predicate, check.get("args") or {})
        if not goal.query:
            return CheckResult(TriState.UNKNOWN, "VisualCheck query is empty")
        goal.context_json = _context_json(check, facts)
        goal.camera_source = str(check.get("camera_source") or self._settings.camera_source)
        goal.max_age_sec = float(check.get("max_age_sec") or self._settings.max_age_sec)
        goal.timeout_sec = float(check.get("timeout_sec") or self._settings.timeout_sec)
        goal.expected_schema_version = self._settings.expected_schema_version

        ok, goal_handle_or_message = _wait_future(
            self._client.send_goal_async(goal),
            timeout_sec=5.0,
            cancel_event=cancel_event,
        )
        if not ok:
            return CheckResult(TriState.UNKNOWN, f"VisualCheck goal send failed: {goal_handle_or_message}")
        goal_handle = goal_handle_or_message
        if not getattr(goal_handle, "accepted", False):
            return CheckResult(TriState.UNKNOWN, "VisualCheck goal rejected")

        ok, wrapped_or_message = _wait_future(
            goal_handle.get_result_async(),
            timeout_sec=max(1.0, goal.timeout_sec + 2.0),
            cancel_event=cancel_event,
            on_cancel=lambda: _cancel_goal(goal_handle),
        )
        if not ok:
            return CheckResult(TriState.UNKNOWN, f"VisualCheck result failed: {wrapped_or_message}")

        wrapped = wrapped_or_message
        status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
        if status != GoalStatus.STATUS_SUCCEEDED:
            return CheckResult(TriState.UNKNOWN, f"VisualCheck ended with {_status_name(status)}")
        action_result = getattr(wrapped, "result", None)
        result = getattr(action_result, "result", None)
        if result is None:
            return CheckResult(TriState.UNKNOWN, "VisualCheck returned no result")
        return _check_result_from_msg(result)


class MissionCheckExecutor:
    """Wrap structured GoalChecker with mission-scoped VisualCheck fallback."""

    def __init__(
        self,
        *,
        goal_checker,
        visual_client: VisualCheckClient,
        identity: Identity,
        cancel_event: Event | None = None,
    ) -> None:
        self._goal_checker = goal_checker
        self._visual_client = visual_client
        self._identity = identity
        self._cancel_event = cancel_event

    def check_json(self, goal_spec_json: str, execution) -> CheckResult:
        try:
            goal_spec = json.loads(goal_spec_json)
        except json.JSONDecodeError as exc:
            return CheckResult(TriState.FALSE, f"invalid goal_spec_json: {exc}")
        return self.check(goal_spec, execution)

    def check(self, goal_spec: dict[str, Any], execution) -> CheckResult:
        result = self._goal_checker.check(goal_spec, execution)
        if result.state is not TriState.UNKNOWN:
            return result
        if _allows_visual_fallback(goal_spec):
            visual = self.visual_check(goal_spec, execution.facts)
            if visual.state is not TriState.UNKNOWN:
                return visual
            return CheckResult(TriState.UNKNOWN, f"{result.message}; VisualCheck UNKNOWN: {visual.message}")
        return result

    def visual_check(self, check: dict[str, Any], facts: dict[str, Any]) -> CheckResult:
        return self._visual_client.check(
            identity=self._identity,
            check=check,
            facts=facts,
            cancel_event=self._cancel_event,
        )


def _allows_visual_fallback(goal_spec: dict[str, Any]) -> bool:
    goal_type = goal_spec.get("type")
    if goal_type == "visual":
        return True
    verification = goal_spec.get("verification")
    if not isinstance(verification, dict):
        return False
    mode = str(verification.get("mode") or "")
    return "visual_check" in mode


def _check_result_from_msg(result: VisualCheckResult) -> CheckResult:
    state = int(getattr(result, "state", VisualCheckResult.STATE_UNKNOWN))
    reason = str(getattr(result, "reason", "") or "")
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    if state == VisualCheckResult.STATE_TRUE:
        return CheckResult(TriState.TRUE, f"{reason} (confidence={confidence:.2f})")
    if state == VisualCheckResult.STATE_FALSE:
        return CheckResult(TriState.FALSE, f"{reason} (confidence={confidence:.2f})")
    return CheckResult(TriState.UNKNOWN, f"{reason or 'unknown'} (confidence={confidence:.2f})")


def _context_json(check: dict[str, Any], facts: dict[str, Any]) -> str:
    explicit = check.get("context_json")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    context = {
        "schema": "mc_ai_bt.visual_check_context.v1",
        "check": check,
        "blackboard_facts": facts,
    }
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


def _query_from_predicate(predicate: str, args: dict[str, Any]) -> str:
    if predicate == "object_visible":
        name = str(args.get("name") or args.get("object") or "").strip()
        return f"is the {name} visible?" if name else ""
    if predicate == "person_visible":
        person_id = str(args.get("person_id") or args.get("id") or "").strip()
        return f"is person {person_id} visible?" if person_id else "is any person visible?"
    return ""


def _wait_future(
    future,
    *,
    timeout_sec: float,
    cancel_event: Event | None = None,
    on_cancel=None,
):
    done = threading.Event()
    box = {}

    def _done(fut):
        try:
            box["result"] = fut.result()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    future.add_done_callback(_done)
    deadline = monotonic() + max(0.0, timeout_sec)
    canceled = False
    while not done.wait(timeout=0.05):
        if cancel_event is not None and cancel_event.is_set():
            if not canceled and on_cancel is not None:
                on_cancel()
                canceled = True
            return False, "mission canceled"
        if monotonic() >= deadline:
            return False, "future timed out"
    if "error" in box:
        exc = box["error"]
        return False, f"{type(exc).__name__}: {exc}"
    return True, box.get("result")


def _action_server_ready(client, *, timeout_sec: float) -> bool:
    try:
        if client.server_is_ready():
            return True
    except Exception:
        return False
    try:
        return bool(client.wait_for_server(timeout_sec=timeout_sec))
    except TypeError:
        return bool(client.wait_for_server(timeout_sec))
    except Exception:
        return False


def _cancel_goal(goal_handle) -> None:
    try:
        goal_handle.cancel_goal_async()
    except Exception:
        pass


def _status_name(status: int) -> str:
    names = {
        GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
        GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
        GoalStatus.STATUS_EXECUTING: "EXECUTING",
        GoalStatus.STATUS_CANCELING: "CANCELING",
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }
    return names.get(status, f"STATUS_{status}")
