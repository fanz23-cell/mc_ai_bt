"""The real DecisionResolver implementation for D v1 (in-place UNKNOWN resolution).

See executor.py's DecisionResolver/DecisionOutcome docstrings for the boundary this
respects: BtExecutor only ever calls resolve() and looks at the returned state, it
knows nothing about ROS, perception, or Omega.

Two resolution policies, and nothing implicit about which predicates get which:

PHYSICAL_PREDICATES -- entity_approached/object_visible/person_visible. Each has an
explicit, bespoke evidence strategy that re-derives TRUE/FALSE from a fresh,
independently-obtained observation (a live LocalizeObject query, or re-reading the
continuously-updating people/visible_people world-state fact a moment later) --
never from a cognitive assertion. FOUND LIVE 2026-09-01 (3rd-party review): earlier
drafts of this assumed a generic "look up snapshot_scope, write predicate as the
key" approach would work for any physical predicate; it does not -- goal_check.py's
_check_snapshot has a bespoke branch per predicate (robot_at_place reads
navigation.current_place, object_visible calls find_object_fact + a specific
value shape, entity_approached is the ONLY one of these three on the generic
PREDICATE_REGISTRY path) with no common schema at all. Predicates without a
strategy here are deliberately unsupported -- they fall straight through to
today's exact needs_decision=True -> pause/resume path, not a guess.

SEMANTIC_PREDICATES -- reserved for a future genuine ambiguity/policy predicate
(no PREDICATE_REGISTRY entry needs this today). Resolved via the SAME
RequestHumanConfirmation/RespondHumanConfirmation channel already wired end-to-end
to Omega (confirmed live: ros_adapter.py's _on_confirmation_status forwards the
prompt, codey_robot.py's confirm_request replies -- no new mc_one interface
needed). The result is a DecisionOutcome the calling Condition/GoalCheck consumes
directly; it is never written into execution facts or WorldState under the
predicate's own name, and never conflated with a literal human_confirmation fact --
provenance stays "an external decision was made for THIS check", not "the world
state changed".
"""
from __future__ import annotations

import json
import math
import threading
import uuid
from time import monotonic
from typing import Any

from rclpy.action import ActionClient
from rclpy.node import Node

from mc_one.action import RequestHumanConfirmation
from mc_one.msg import AiBtIdentity
from mc_one.srv import LocalizeObject

from .executor import CheckExecutor, DecisionOutcome, ExecutionResult
from .world_state_client import WorldStateWriter

# FOUND LIVE 2026-08-31 (mc_embodied_skills/node.py's _verify_entity_approached):
# "walked up to", not "standing on top of" -- mirrored here since this predicate's
# canonical evidence strategy needs the identical threshold, and the two packages
# have no shared import path (separate Docker images).
_APPROACH_DISTANCE_TOLERANCE_M = 1.5

PHYSICAL_PREDICATES = {"entity_approached", "object_visible", "person_visible"}
SEMANTIC_PREDICATES: set[str] = set()


class LiveObjectLocator:
    """A direct, on-demand /mc_perception/localize_object query with fresh=True --
    the same real service mc_embodied_skills/node.py's ObjectLocalizerClient uses,
    called independently here (a ROS service is a network endpoint, not a Python
    import -- no cross-repo Python coupling, unlike importing another package's
    module directly)."""

    def __init__(
        self, node: Node, *, service_name: str = "/mc_perception/localize_object", callback_group=None
    ) -> None:
        self._client = node.create_client(LocalizeObject, service_name, callback_group=callback_group)

    def locate(self, object_name: str, *, timeout_sec: float = 2.0) -> tuple[dict[str, Any] | None, str]:
        object_name = str(object_name or "").strip()
        if not object_name:
            return None, "empty target"
        if not self._client.service_is_ready():
            return None, "object localization service unavailable"
        request = LocalizeObject.Request()
        request.object_name = object_name
        request.score_thr = 0.0
        request.fresh = True
        request.max_age = 0.0
        ok, response_or_message = _wait_future(self._client.call_async(request), timeout_sec=timeout_sec)
        if not ok:
            return None, f"{object_name}: localize_object call failed: {response_or_message}"
        response = response_or_message
        if not bool(getattr(response, "success", False)) or not list(getattr(response, "objects", [])):
            return None, str(getattr(response, "message", "") or f"{object_name}: not found")
        nearest = list(response.objects)[0]  # LocalizeObject.srv: NEAREST FIRST
        return {
            "x": float(nearest.x), "y": float(nearest.y), "z": float(nearest.z),
            "score": float(nearest.score),
        }, "found"


class DecisionBroker:
    """The concrete DecisionResolver node.py wires into BtExecutor.execute_json."""

    def __init__(
        self,
        node: Node,
        *,
        checks: CheckExecutor,
        object_locator: LiveObjectLocator,
        world_writer: WorldStateWriter,
        fast_window_sec: float = 4.0,
        poll_interval_sec: float = 0.5,
        confirmation_timeout_sec: float = 4.0,
        callback_group=None,
    ) -> None:
        self._node = node
        self._checks = checks
        self._object_locator = object_locator
        self._world_writer = world_writer
        self._fast_window_sec = max(0.5, float(fast_window_sec))
        self._poll_interval_sec = max(0.1, float(poll_interval_sec))
        self._confirmation_timeout_sec = max(0.5, float(confirmation_timeout_sec))
        self._confirmation_client = ActionClient(
            node, RequestHumanConfirmation, "/mc_ai_bt/request_human_confirmation",
            callback_group=callback_group,
        )

    def resolve(
        self,
        *,
        predicate: str,
        args: dict[str, Any],
        reason: str,
        facts: dict[str, Any],
        cancel_event,
    ) -> DecisionOutcome:
        if predicate in PHYSICAL_PREDICATES:
            return self._resolve_physical(predicate, args, cancel_event)
        if predicate in SEMANTIC_PREDICATES:
            return self._resolve_semantic(predicate, args, reason, cancel_event)
        return DecisionOutcome("UNKNOWN", f"no resolution strategy defined for {predicate!r}")

    # --- physical: fresh, independently-obtained evidence only ---------------------

    def _resolve_physical(self, predicate: str, args: dict[str, Any], cancel_event) -> DecisionOutcome:
        goal_spec = {"type": "structured", "predicate": predicate, "args": args,
                     "verification": {"mode": "world_state"}}
        if predicate == "entity_approached":
            refresh = lambda: self._refresh_entity_approached(args)  # noqa: E731
        elif predicate == "object_visible":
            refresh = lambda: self._refresh_object_visible(args)  # noqa: E731
        else:
            refresh = lambda: None  # noqa: E731 -- person_visible: nothing to actively
            # produce; the pose-detection topic already refreshes visible_people
            # continuously, poll_until_known just re-reads it a moment later.
        return self._poll_until_known(goal_spec, refresh, cancel_event)

    def _poll_until_known(self, goal_spec: dict[str, Any], refresh, cancel_event) -> DecisionOutcome:
        deadline = monotonic() + self._fast_window_sec
        waiter = cancel_event if cancel_event is not None else threading.Event()
        while True:
            refresh()
            result = self._checks.check(goal_spec, ExecutionResult(True, "decision resolver input", {}))
            state = str(getattr(getattr(result, "state", None), "value", getattr(result, "state", "UNKNOWN")))
            message = str(getattr(result, "message", ""))
            if state == "TRUE":
                return DecisionOutcome("TRUE", message)
            if state == "FALSE":
                return DecisionOutcome("FALSE", message)
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                return DecisionOutcome("UNKNOWN", f"fast resolution window elapsed: {message}")
            if waiter.wait(timeout=min(self._poll_interval_sec, remaining)):
                return DecisionOutcome("UNKNOWN", "canceled while awaiting resolution")

    def _refresh_entity_approached(self, args: dict[str, Any]) -> None:
        target = str(args.get("target") or args.get("entity") or args.get("entity_id") or "").strip()
        if not target:
            return
        fact, _reason = self._object_locator.locate(target)
        if fact is None:
            # FOUND LIVE 2026-09-01 (caught by this file's own end-to-end test): a
            # single missed fresh locate is not proof the approach failed -- the
            # target may just be momentarily out of frame. Write nothing, same
            # convention as mc_embodied_skills/node.py's _verify_entity_approached
            # (no shared import path, mirrored intentionally): absence stays
            # UNKNOWN here, it must not become a confirmed matched=False, or a
            # target that reappears one poll later can never resolve TRUE again --
            # goal_check.py's own _direct_predicate_result has no "retract a FALSE"
            # path, only a fresh write can move it.
            return
        distance = math.hypot(float(fact["x"]), float(fact["y"]))
        value = {
            "matched": distance <= _APPROACH_DISTANCE_TOLERANCE_M,
            "target": target,
            "distance_after_arrival_m": distance,
            "verification_basis": "decision_broker_fresh_check",
        }
        self._world_writer.update_fact(
            source="mc_ai_bt.decision_broker", scope="objects", key="entity_approached", value=value)

    def _refresh_object_visible(self, args: dict[str, Any]) -> None:
        object_name = str(
            args.get("name") or args.get("object_name") or args.get("object") or args.get("target") or ""
        ).strip()
        if not object_name:
            return
        fact, _reason = self._object_locator.locate(object_name)
        if fact is None:
            value = {"object_name": object_name, "visible": False}
        else:
            value = {"object_name": object_name, "visible": True, "score": float(fact["score"])}
        key = "object:" + _normalise(object_name)
        self._world_writer.update_fact(
            source="mc_ai_bt.decision_broker", scope="objects", key=key, value=value)

    # --- semantic/policy: an external decision IS the answer, mission-local ---------

    def _resolve_semantic(self, predicate: str, args: dict[str, Any], reason: str, cancel_event) -> DecisionOutcome:
        if not self._confirmation_client.wait_for_server(timeout_sec=1.0):
            return DecisionOutcome("UNKNOWN", "confirmation channel unavailable")
        goal = RequestHumanConfirmation.Goal()
        goal.identity = AiBtIdentity()
        goal.request_id = uuid.uuid4().hex
        goal.prompt = f"{predicate} is unknown: {reason} (args={json.dumps(args, sort_keys=True)})"
        goal.context_json = "{}"
        goal.required_role = ""
        goal.timeout_sec = self._confirmation_timeout_sec
        ok, goal_handle_or_message = _wait_future(
            self._confirmation_client.send_goal_async(goal), timeout_sec=2.0)
        if not ok or not getattr(goal_handle_or_message, "accepted", False):
            return DecisionOutcome("UNKNOWN", "confirmation request was not accepted")
        goal_handle = goal_handle_or_message
        ok, wrapped_or_message = _wait_future(
            goal_handle.get_result_async(),
            timeout_sec=self._confirmation_timeout_sec + 2.0,
            cancel_event=cancel_event,
            on_cancel=lambda: _cancel_goal(goal_handle),
        )
        if not ok:
            return DecisionOutcome("UNKNOWN", f"confirmation result unavailable: {wrapped_or_message}")
        result = getattr(wrapped_or_message, "result", None)
        decision = int(getattr(result, "decision", RequestHumanConfirmation.Goal.DECISION_UNKNOWN))
        response_reason = str(getattr(result, "reason", "") or "")
        if decision == RequestHumanConfirmation.Goal.DECISION_APPROVED:
            return DecisionOutcome("TRUE", response_reason or "approved")
        if decision == RequestHumanConfirmation.Goal.DECISION_DENIED:
            return DecisionOutcome("FALSE", response_reason or "denied")
        return DecisionOutcome("UNKNOWN", response_reason or "confirmation timed out or was canceled")


def _normalise(name: str) -> str:
    import re
    text = str(name or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cancel_goal(goal_handle) -> None:
    try:
        future = goal_handle.cancel_goal_async()
    except Exception:
        return
    _wait_future(future, timeout_sec=1.0)


def _wait_future(future, *, timeout_sec: float, cancel_event=None, on_cancel=None):
    done = threading.Event()
    box: dict[str, Any] = {}

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
        if monotonic() >= deadline:
            if not canceled and on_cancel is not None:
                on_cancel()
            return False, "future timed out"
    if "error" in box:
        exc = box["error"]
        return False, f"{type(exc).__name__}: {exc}"
    return True, box.get("result")
