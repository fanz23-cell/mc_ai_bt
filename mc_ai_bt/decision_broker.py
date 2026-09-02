"""The real DecisionResolver implementation for D v1 (in-place UNKNOWN resolution).

See executor.py's DecisionResolver/DecisionOutcome docstrings for the boundary this
respects: BtExecutor only ever calls resolve() and looks at the returned state, it
knows nothing about ROS, perception, or Omega.

Two resolution policies, and nothing implicit about which predicates get which:

PHYSICAL_PREDICATES -- entity_approached/object_visible/person_visible. Each has an
explicit, bespoke evidence strategy that re-derives TRUE/FALSE from a fresh,
independently-obtained observation -- never from a cognitive assertion, and never by
writing a synthetic global WorldState fact under the predicate's own flat name.

FOUND LIVE 2026-09-01 (4th-party review of the first D v1 cut), three real bugs in
that first cut, all now fixed here:
  1. LiveObjectLocator.locate() collapsed "service unavailable/timeout/exception" and
     "queried fine, genuinely not there" into the same `None` -- object_visible then
     wrote a confirmed visible=False for a plain perception outage. Now a typed
     LocateResult (FOUND/NOT_FOUND/INCONCLUSIVE) keeps them apart; only NOT_FOUND
     (a completed query with nothing detected) can ever produce a real FALSE.
  2. entity_approached wrote to a single global objects.entity_approached key with no
     target scoping at all -- goal_check.py's generic PREDICATE_REGISTRY path
     (_direct_predicate_result) never compares value.target against args.target for
     this predicate, so confirming Alice was approached could satisfy a LATER,
     unrelated check for Bob reading the same stale global key. object_visible had
     the same problem one level worse: its flat "object:<name>" key sits ahead of
     mc_world_state's own per-instance "object:<name>:<cell>" keys in
     find_object_fact()'s own lookup order (direct key wins first), so a bad write
     here could shadow a real, currently-visible object recorded under a different
     key entirely -- see mc_world_state/node.py's _on_object_localization multi-
     instance fix.
  Both are fixed the same way: entity_approached and object_visible no longer touch
  WorldState at all. A fresh, independently-obtained observation is enough on its
  own to answer THIS check directly (a DecisionOutcome), with no round trip through
  a shared, unscoped fact a different check could misread later.
person_visible is the one predicate here that legitimately still goes through
GoalChecker + WorldState: there is no on-demand "give me a fresh person detection"
service (unlike LocalizeObject), so this waits a short beat and re-reads the
continuously-updating people.visible_people fact the always-running pose-detection
topic already refreshes -- not writing anything synthetic itself.

Predicates without a strategy here are deliberately unsupported -- they fall
straight through to today's exact needs_decision=True -> pause/resume path, not a
guess (this specifically includes robot_at_place -- goal_check.py's _check_snapshot
has a bespoke branch per physical predicate, not one generic "scope.get(predicate)"
write target, so a strategy must be added deliberately per predicate, never
inferred from snapshot_scope or any other generic metadata).

SEMANTIC_PREDICATES -- reserved for a future genuine ambiguity/policy predicate (no
real PREDICATE_REGISTRY entry needs this today; see _D_V1_SEMANTIC_SMOKE_TEST_PREDICATE
below for how the mechanism itself is proven end-to-end regardless). Resolved via
the SAME RequestHumanConfirmation/RespondHumanConfirmation channel already wired
end-to-end to Omega (confirmed live: ros_adapter.py's _on_confirmation_status
forwards the prompt, codey_robot.py's confirm_request replies -- no new mc_one
interface needed). The result is a DecisionOutcome the calling Condition/GoalCheck
consumes directly; it is never written into execution facts or WorldState under the
predicate's own name, and never conflated with a literal human_confirmation fact --
provenance stays "an external decision was made for THIS check", not "the world
state changed".

KNOWN BOUNDARY, not fixed here: entity_approached's strategy only helps when an
UNKNOWN reaches an explicit Condition/GoalCheck node (or the mission's final
goal_spec, now that node.py's _run_mission wires this through too). It does NOT
help approach_entity's own Action-level failure -- when mc_embodied_skills' post-
arrival fresh check can't confirm the target, skill_adapters.py's embodied-skill
result handling (the `status != GoalStatus.STATUS_SUCCEEDED or not success` branch)
returns ExecutionResult(blocked=True) with needs_decision left at its default
(False) and evidence_json never read on that path at all -- so a Sequence with
approach_entity followed by a Condition never even reaches the Condition; the
Action itself already ended the Sequence with a mechanical-looking BLOCKED, not an
evidentiary one. Giving that failure path needs_decision=True (and reading its
evidence_json) so it can reach a resolver too is real, separate work against
skill_adapters.py, deliberately out of scope here.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from dataclasses import dataclass
from time import monotonic
from typing import Any

from rclpy.action import ActionClient
from rclpy.node import Node

from mc_one.action import RequestHumanConfirmation
from mc_one.msg import AiBtIdentity
from mc_one.srv import LocalizeObject

from .executor import DecisionOutcome

# FOUND LIVE 2026-08-31 (mc_embodied_skills/node.py's _verify_entity_approached):
# "walked up to", not "standing on top of" -- mirrored here since this predicate's
# canonical evidence strategy needs the identical threshold, and the two packages
# have no shared import path (separate Docker images).
_APPROACH_DISTANCE_TOLERANCE_M = 1.5

# Any real, permanent COCO class works here -- this exists purely to learn what
# mc_perception currently considers visible, never to look for a person
# specifically. Mirrors mc_embodied_skills/node.py's own _PROBE_CLASS_NAME (same
# reasoning, no shared import path).
_PROBE_CLASS_NAME = "person"

PHYSICAL_PREDICATES = {"entity_approached", "object_visible", "person_visible"}
# Not a real planner-facing predicate -- exists purely so an integration test can
# exercise the PUBLIC resolve() -> RequestHumanConfirmation -> Omega -> DecisionOutcome
# path end to end, the same way it will work the day a genuine semantic/policy
# predicate is registered here. Never returned to a real Condition/GoalCheck in
# production: nothing in goal_check.py's PREDICATE_REGISTRY is named this.
_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE = "_d_v1_semantic_smoke_test"
SEMANTIC_PREDICATES: set[str] = {_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE}


@dataclass(frozen=True)
class LocateResult:
    """FOUND: a completed, fresh query that found the target -- fact is set.
    NOT_FOUND: a completed, fresh query that conclusively did not (LocalizeObject.srv:
    "Empty with success=true means looked, nothing of that class in view") -- a real
    negative, safe to treat as FALSE for a "is X visible right now" predicate.
    INCONCLUSIVE: the query itself did not complete (service down, timeout, an
    unrecognized class name, any exception) -- says nothing about the target at all,
    must never be treated as evidence of absence."""

    status: str  # "FOUND" | "NOT_FOUND" | "INCONCLUSIVE"
    fact: dict[str, Any] | None = None
    reason: str = ""


class LiveObjectLocator:
    """A direct, on-demand /mc_perception/localize_object query with fresh=True --
    the same real service mc_embodied_skills/node.py's ObjectLocalizerClient uses,
    called independently here (a ROS service is a network endpoint, not a Python
    import -- no cross-repo Python coupling, unlike importing another package's
    module directly). Mirrors that client's own probe-then-match-available-classes
    step (own module docstring: "plant" must still resolve against the class
    "potted plant") -- skipping it here would make D v1 strictly worse at finding
    the same targets the rest of this codebase already handles correctly.
    """

    def __init__(
        self, node: Node, *, service_name: str = "/mc_perception/localize_object", callback_group=None
    ) -> None:
        self._client = node.create_client(LocalizeObject, service_name, callback_group=callback_group)

    def locate(self, target: str, *, timeout_sec: float = 2.0) -> LocateResult:
        target = str(target or "").strip()
        if not target:
            return LocateResult("INCONCLUSIVE", reason="empty target")
        if not self._client.service_is_ready():
            return LocateResult("INCONCLUSIVE", reason="object localization service unavailable")
        probe = self._call(_PROBE_CLASS_NAME, timeout_sec=timeout_sec)
        if probe.status == "INCONCLUSIVE":
            return probe
        candidates = _match_available_classes(target, probe.fact["available"] if probe.fact else [])
        if not candidates:
            available = probe.fact["available"] if probe.fact else []
            if available:
                return LocateResult("NOT_FOUND", reason=f"{target}: not among currently visible objects ({', '.join(available)})")
            return LocateResult("NOT_FOUND", reason=f"{target}: nothing currently visible")
        last = LocateResult("NOT_FOUND", reason=f"{target}: not found")
        for name in candidates:
            result = self._call(name, timeout_sec=timeout_sec)
            if result.status == "INCONCLUSIVE":
                return result
            if result.status == "FOUND":
                return result
            last = result
        return last

    def _call(self, object_name: str, *, timeout_sec: float) -> LocateResult:
        request = LocalizeObject.Request()
        request.object_name = object_name
        request.score_thr = 0.0
        request.fresh = True
        request.max_age = 0.0
        ok, response_or_message = _wait_future(self._client.call_async(request), timeout_sec=timeout_sec)
        if not ok:
            return LocateResult("INCONCLUSIVE", reason=f"{object_name}: localize_object call failed: {response_or_message}")
        response = response_or_message
        available = [str(name) for name in (getattr(response, "available", None) or [])]
        if not bool(getattr(response, "success", False)):
            return LocateResult(
                "INCONCLUSIVE", reason=str(getattr(response, "message", "") or f"{object_name}: query failed"))
        objects = list(getattr(response, "objects", []))
        if not objects:
            return LocateResult(
                "NOT_FOUND", fact={"available": available},
                reason=str(getattr(response, "message", "") or f"{object_name}: not found"))
        nearest = objects[0]  # LocalizeObject.srv: NEAREST FIRST
        return LocateResult("FOUND", fact={
            "x": float(nearest.x), "y": float(nearest.y), "z": float(nearest.z),
            "score": float(nearest.score), "available": available,
        })


def _match_available_classes(target: str, available: list[str]) -> list[str]:
    """Mirrors mc_embodied_skills/node.py's own _match_available_classes verbatim
    (no shared import path between the two images) -- see that function's docstring
    for why both match directions are needed ("plant" as a target must match the
    class "potted plant", found live 2026-08-31)."""
    lowered = str(target or "").strip().lower()
    if not lowered:
        return []
    words = set(lowered.split())
    seen: set[str] = set()
    matches: list[str] = []
    for name in available:
        name_lower = str(name or "").strip().lower()
        if not name_lower or name_lower in seen:
            continue
        if name_lower == lowered or name_lower in words or name_lower in lowered or lowered in name_lower:
            seen.add(name_lower)
            matches.append(name)
    return matches


class DecisionBroker:
    """The concrete DecisionResolver node.py wires into BtExecutor.execute_json.
    Bind a mission's identity with .for_mission(identity) before passing to
    execute_json -- see MissionBoundDecisionResolver below."""

    def __init__(
        self,
        node: Node,
        *,
        object_locator: LiveObjectLocator,
        person_visible_checker=None,
        fast_window_sec: float = 4.0,
        poll_interval_sec: float = 0.5,
        confirmation_timeout_sec: float = 4.0,
        callback_group=None,
    ) -> None:
        self._node = node
        self._object_locator = object_locator
        # Only person_visible ever calls this -- a real CheckExecutor (GoalChecker),
        # used exactly like _run_mission's own checks.check(), never written to.
        self._person_visible_checker = person_visible_checker
        self._fast_window_sec = max(0.5, float(fast_window_sec))
        self._poll_interval_sec = max(0.1, float(poll_interval_sec))
        self._confirmation_timeout_sec = max(0.5, float(confirmation_timeout_sec))
        self._confirmation_client = ActionClient(
            node, RequestHumanConfirmation, "/mc_ai_bt/request_human_confirmation",
            callback_group=callback_group,
        )

    def for_mission(self, identity) -> "MissionBoundDecisionResolver":
        return MissionBoundDecisionResolver(self, identity)

    def resolve(
        self,
        *,
        predicate: str,
        args: dict[str, Any],
        reason: str,
        facts: dict[str, Any],
        cancel_event,
        identity=None,
    ) -> DecisionOutcome:
        if predicate in PHYSICAL_PREDICATES:
            return self._resolve_physical(predicate, args, cancel_event)
        if predicate in SEMANTIC_PREDICATES:
            return self._resolve_semantic(predicate, args, reason, cancel_event, identity)
        return DecisionOutcome("UNKNOWN", f"no resolution strategy defined for {predicate!r}")

    # --- physical: fresh, independently-obtained evidence only, check-local --------

    def _resolve_physical(self, predicate: str, args: dict[str, Any], cancel_event) -> DecisionOutcome:
        if predicate == "entity_approached":
            return self._poll_direct(cancel_event, lambda: self._check_entity_approached(args))
        if predicate == "object_visible":
            return self._poll_direct(cancel_event, lambda: self._check_object_visible(args))
        return self._poll_via_checker(predicate, args, cancel_event)

    def _poll_direct(self, cancel_event, check_once) -> DecisionOutcome:
        """For predicates resolved directly from a fresh observation -- never through
        GoalChecker/WorldState, so there is no shared fact for a different check to
        misread (see this module's own docstring, bugs 1-2)."""
        deadline = monotonic() + self._fast_window_sec
        waiter = cancel_event if cancel_event is not None else threading.Event()
        last_message = ""
        while True:
            outcome = check_once()
            if outcome.state in ("TRUE", "FALSE"):
                return outcome
            last_message = outcome.message
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                return DecisionOutcome("UNKNOWN", f"fast resolution window elapsed: {last_message}")
            if waiter.wait(timeout=min(self._poll_interval_sec, remaining)):
                return DecisionOutcome("UNKNOWN", "canceled while awaiting resolution")

    def _check_entity_approached(self, args: dict[str, Any]) -> DecisionOutcome:
        target = str(args.get("target") or args.get("entity") or args.get("entity_id") or "").strip()
        if not target:
            return DecisionOutcome("UNKNOWN", "entity_approached requires a target")
        located = self._object_locator.locate(target)
        if located.status != "FOUND":
            # INCONCLUSIVE (perception outage) and NOT_FOUND (a single missed fresh
            # query) are both treated as "no answer yet" for this predicate, not a
            # confirmed FALSE -- a momentary miss is not proof the approach failed,
            # same convention mc_embodied_skills/node.py's _verify_entity_approached
            # already established.
            return DecisionOutcome("UNKNOWN", f"{target}: {located.reason}")
        fact = located.fact
        distance = math.hypot(float(fact["x"]), float(fact["y"]))
        if distance <= _APPROACH_DISTANCE_TOLERANCE_M:
            return DecisionOutcome("TRUE", f"{target} is {distance:.2f}m away (fresh check)")
        return DecisionOutcome(
            "FALSE", f"{target} is {distance:.2f}m away, not within {_APPROACH_DISTANCE_TOLERANCE_M}m")

    def _check_object_visible(self, args: dict[str, Any]) -> DecisionOutcome:
        object_name = str(
            args.get("name") or args.get("object_name") or args.get("object") or args.get("target") or ""
        ).strip()
        if not object_name:
            return DecisionOutcome("UNKNOWN", "object_visible requires a name")
        min_score = _min_score(args)
        located = self._object_locator.locate(object_name)
        if located.status == "INCONCLUSIVE":
            return DecisionOutcome("UNKNOWN", f"{object_name}: {located.reason}")
        if located.status == "NOT_FOUND":
            # A completed query that genuinely found nothing IS a real answer for
            # "is X visible right now" (unlike entity_approached's past-tense
            # "did I successfully arrive" claim) -- matches goal_check.py's own
            # object_visibility_evidence, which already treats this shape as FALSE.
            return DecisionOutcome("FALSE", f"{object_name}: {located.reason}")
        score = float(located.fact["score"])
        if min_score > 0.0 and score < min_score:
            return DecisionOutcome("FALSE", f"{object_name}: score {score:.2f} below min {min_score:.2f}")
        return DecisionOutcome("TRUE", f"{object_name}: visible (score={score:.2f}, fresh check)")

    def _poll_via_checker(self, predicate: str, args: dict[str, Any], cancel_event) -> DecisionOutcome:
        """person_visible only: no on-demand force-fresh service exists, so this
        just re-reads the continuously-updating people.visible_people world-state
        fact a moment later (a real, live-refreshing perception source, never
        anything this broker writes itself)."""
        if self._person_visible_checker is None:
            return DecisionOutcome("UNKNOWN", "person_visible checker is not configured")
        goal_spec = {"type": "structured", "predicate": predicate, "args": args,
                     "verification": {"mode": "world_state"}}
        deadline = monotonic() + self._fast_window_sec
        waiter = cancel_event if cancel_event is not None else threading.Event()
        while True:
            result = self._person_visible_checker.check(goal_spec, _CheckInput(facts={}))
            state = _tri_state_value(result)
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

    # --- semantic/policy: an external decision IS the answer, mission-local ---------

    def _resolve_semantic(
        self, predicate: str, args: dict[str, Any], reason: str, cancel_event, identity,
    ) -> DecisionOutcome:
        if not self._confirmation_client.wait_for_server(timeout_sec=1.0):
            return DecisionOutcome("UNKNOWN", "confirmation channel unavailable")
        goal = RequestHumanConfirmation.Goal()
        goal.identity = identity if identity is not None else AiBtIdentity()
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


class MissionBoundDecisionResolver:
    """Binds one mission's identity to a shared DecisionBroker -- same per-run
    binding idiom visual_client.py's MissionCheckExecutor already uses for the same
    reason (constructed fresh in _run_mission with identity=mission.identity).
    Implements exactly the DecisionResolver protocol executor.py expects (no
    identity param) -- BtExecutor stays unaware this exists at all."""

    def __init__(self, broker: DecisionBroker, identity) -> None:
        self._broker = broker
        self._identity = identity

    def resolve(self, *, predicate, args, reason, facts, cancel_event) -> DecisionOutcome:
        return self._broker.resolve(
            predicate=predicate, args=args, reason=reason, facts=facts,
            cancel_event=cancel_event, identity=self._identity,
        )


@dataclass(frozen=True)
class _CheckInput:
    facts: dict[str, Any]
    success: bool = True
    message: str = "decision resolver input"
    blocked: bool = False
    needs_decision: bool = False


def _min_score(args: dict[str, Any]) -> float:
    try:
        value = float(args.get("min_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _tri_state_value(result: Any) -> str:
    return str(getattr(getattr(result, "state", None), "value", getattr(result, "state", "UNKNOWN")))


def _cancel_goal(goal_handle) -> None:
    try:
        future = goal_handle.cancel_goal_async()
    except Exception:
        return
    _wait_future(future, timeout_sec=1.0)


def _wait_future(future, *, timeout_sec: float, cancel_event=None, on_cancel=None):
    """FOUND LIVE 2026-09-01 (4th-party review): the earlier version called on_cancel()
    when cancel_event fired but kept waiting for the ORIGINAL deadline regardless --
    a mission cancel during a semantic wait could sit for the full confirmation
    timeout instead of returning promptly. Now a cancel gets a short grace period of
    its own (the cancel ack, not the original decision) and then returns."""
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
    cancel_deadline: float | None = None
    while True:
        if done.wait(timeout=0.05):
            break
        if cancel_event is not None and cancel_event.is_set():
            if cancel_deadline is None:
                if on_cancel is not None:
                    on_cancel()
                cancel_deadline = monotonic() + 1.0  # a short, separate grace period
            if monotonic() >= cancel_deadline:
                return False, "canceled"
            continue
        if monotonic() >= deadline:
            return False, "future timed out"
    if "error" in box:
        exc = box["error"]
        return False, f"{type(exc).__name__}: {exc}"
    return True, box.get("result")
