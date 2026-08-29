from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import Any, Iterator

from action_msgs.msg import GoalStatus
from mc_one.action import (
    ComeToMe,
    EmbodiedSkill,
    GoToPlace,
    PlayAnimation,
    RequestHumanConfirmation,
    SimpleMove,
)
from mc_one.msg import AiBtIdentity, Utterance
from rclpy.action import ActionClient
from rclpy.node import Node

from .executor import ExecutionResult
from .identity import Identity
from .lease_timing import lease_renew_interval
from .resource_client import ResourceLeaseClient
from .ros_identity import empty_identity_msg, identity_to_msg
from .skill_registry import DEFAULT_SKILLS
from .world_facts import world_fact_updates_for_execution
from .world_state_client import WorldStateWriter


# Computed from skill_registry.py's dispatch field, not hand-copied -- see
# SkillSpec's docstring and OMEGACLAW_AI_BT_INTEGRATION.md §9 for why. Adding a new
# embodied skill now means adding one DEFAULT_SKILLS entry; this recomputes automatically.
EMBODIED_SKILLS = {name for name, spec in DEFAULT_SKILLS.items() if spec.dispatch == "embodied"}


@dataclass(frozen=True)
class ActionBinding:
    client: Any
    goal_type: Any
    resources: tuple[str, ...]
    timeout_sec: float
    success_facts: dict[str, Any]


class RosSkillExecutor:
    """Awaitable skill adapters for AI-BT.

    These adapters call existing robot actions and wait for their terminal
    result. They are separate from the legacy voice inline dispatcher, which is
    intentionally fire-and-forget.
    """

    def __init__(
        self,
        node: Node,
        *,
        leases: ResourceLeaseClient | None = None,
        world_state: WorldStateWriter | None = None,
    ) -> None:
        self._node = node
        self._identity_context = threading.local()
        self._leases = leases or ResourceLeaseClient(node)
        self._world_state = self._build_world_state_writer(world_state)
        self._speak_pub = node.create_publisher(Utterance, "/pipeline/speak", 10)
        self._go_to_place = ActionClient(node, GoToPlace, "/mc_navigation/go_to_place")
        self._come_to_me = ActionClient(node, ComeToMe, "/mc_navigation/come_to_me")
        self._simple_move = ActionClient(node, SimpleMove, "/mc_navigation/simple_move")
        self._play_animation = ActionClient(node, PlayAnimation, "/mc_animator/play")
        self._embodied_skill = ActionClient(node, EmbodiedSkill, "/mc_embodied_skills/execute")
        self._human_confirmation = ActionClient(
            node,
            RequestHumanConfirmation,
            "/mc_ai_bt/request_human_confirmation",
        )
        self._current_lock = threading.Lock()
        self._current_goal_handle: Any = None

    @contextmanager
    def use_identity(self, identity: Identity) -> Iterator[None]:
        previous = getattr(self._identity_context, "identity", None)
        self._identity_context.identity = identity
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._identity_context.identity
                except AttributeError:
                    pass
            else:
                self._identity_context.identity = previous

    def execute_skill(
        self,
        name: str,
        args: dict[str, Any],
        cancel_event: Event | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        if name == "say":
            return self._say(args)
        if name == "request_human_confirmation":
            return self._request_human_confirmation(args, cancel_event, timeout_sec=timeout_sec)
        if name == "go_to_place":
            goal = GoToPlace.Goal()
            goal.name = str(args.get("name") or args.get("place") or "").strip()
            return self._run_action(
                name,
                ActionBinding(
                    self._go_to_place,
                    GoToPlace,
                    ("base",),
                    300.0,
                    {"robot_at_place": goal.name},
                ),
                timeout_sec,
                goal,
                cancel_event,
            )
        if name == "come_to_me":
            return self._run_action(
                name,
                ActionBinding(
                    self._come_to_me,
                    ComeToMe,
                    ("base",),
                    300.0,
                    {"robot_near_interaction_owner": True},
                ),
                timeout_sec,
                ComeToMe.Goal(),
                cancel_event,
            )
        if name == "simple_move":
            goal = SimpleMove.Goal()
            goal.action = str(args.get("action") or "").strip()
            try:
                goal.value = float(args.get("value"))
            except (TypeError, ValueError):
                return ExecutionResult(False, "simple_move.value must be numeric")
            return self._run_action(
                name,
                ActionBinding(
                    self._simple_move,
                    SimpleMove,
                    ("base",),
                    120.0,
                    {
                        "relative_motion_completed": {
                            "action": goal.action,
                            "value": goal.value,
                        }
                    },
                ),
                timeout_sec,
                goal,
                cancel_event,
            )
        if name == "play_animation":
            animation = str(args.get("animation") or args.get("name") or "").strip()
            return self._play_animation_skill(
                name,
                animation,
                args.get("args", ""),
                args,
                {"animation_played": animation},
                cancel_event,
                timeout_sec=timeout_sec,
            )
        if name == "look_at":
            direction = str(args.get("direction") or "front").strip().lower()
            animation_args: dict[str, Any] = {"direction": direction}
            if "hold" in args:
                animation_args["hold"] = args["hold"]
            return self._play_animation_skill(
                name,
                "look_at",
                animation_args,
                args,
                {
                    "animation_played": "look_at",
                    "look_at": {"direction": direction, **({"hold": args["hold"]} if "hold" in args else {})},
                },
                cancel_event,
                timeout_sec=timeout_sec,
            )
        if name == "point_at":
            animation, animation_args = _point_at_animation(args)
            return self._play_animation_skill(
                name,
                animation,
                animation_args,
                args,
                {
                    "animation_played": "point_at",
                    "animation_clip": animation,
                    "point_at": dict(animation_args),
                },
                cancel_event,
                timeout_sec=timeout_sec,
            )
        if name in EMBODIED_SKILLS:
            return self._run_embodied_skill(name, args, cancel_event, timeout_sec=timeout_sec)
        return ExecutionResult(False, f"unsupported skill: {name}")

    def _request_human_confirmation(
        self,
        args: dict[str, Any],
        cancel_event: Event | None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        if not _action_server_ready(self._human_confirmation, timeout_sec=1.0):
            return ExecutionResult(False, "human confirmation action server is not ready", blocked=True)
        prompt = str(args.get("prompt") or args.get("text") or "").strip()
        if not prompt:
            return ExecutionResult(False, "human confirmation prompt is empty")
        goal = RequestHumanConfirmation.Goal()
        goal.identity = self._lease_identity_msg()
        goal.request_id = str(args.get("request_id") or _confirmation_request_id(goal.identity))
        goal.prompt = prompt
        context = args.get("context_json", args.get("context", {}))
        goal.context_json = _json_text(context)
        goal.required_role = str(args.get("required_role") or "operator").strip()
        goal.timeout_sec = float(timeout_sec or args.get("timeout_sec") or 30.0)

        ok, goal_handle_or_message = _wait_future(
            self._human_confirmation.send_goal_async(goal),
            timeout_sec=5.0,
            cancel_event=cancel_event,
        )
        if not ok:
            return ExecutionResult(False, f"human confirmation goal send failed: {goal_handle_or_message}")
        goal_handle = goal_handle_or_message
        if not getattr(goal_handle, "accepted", False):
            return ExecutionResult(False, "human confirmation goal rejected", blocked=True)

        with self._current_lock:
            self._current_goal_handle = goal_handle
        try:
            ok, wrapped_result_or_message = _wait_future(
                goal_handle.get_result_async(),
                timeout_sec=max(0.1, goal.timeout_sec + 2.0),
                cancel_event=cancel_event,
                on_cancel=lambda: _cancel_goal(goal_handle),
            )
            if not ok:
                return ExecutionResult(False, f"human confirmation result failed: {wrapped_result_or_message}")
            wrapped = wrapped_result_or_message
            result = getattr(wrapped, "result", None)
            decision = int(getattr(result, "decision", 0))
            facts = {
                "human_confirmation": {
                    "request_id": goal.request_id,
                    "decision": decision,
                    "approved": decision == RequestHumanConfirmation.Goal.DECISION_APPROVED,
                    "responder_id": str(getattr(result, "responder_id", "") or ""),
                    "reason": str(getattr(result, "reason", "") or ""),
                    "required_role": goal.required_role,
                }
            }
            if decision == RequestHumanConfirmation.Goal.DECISION_APPROVED:
                execution = ExecutionResult(True, "human confirmation approved", facts)
                self._publish_success_facts(execution.facts)
                return execution
            return ExecutionResult(
                False,
                f"human confirmation {_confirmation_decision_name(decision)}",
                facts,
                blocked=True,
            )
        finally:
            with self._current_lock:
                if self._current_goal_handle is goal_handle:
                    self._current_goal_handle = None

    def _run_embodied_skill(
        self,
        skill_name: str,
        args: dict[str, Any],
        cancel_event: Event | None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        if not _action_server_ready(self._embodied_skill, timeout_sec=1.0):
            return ExecutionResult(False, "embodied skill action server is not ready")
        goal = EmbodiedSkill.Goal()
        goal.identity = self._lease_identity_msg()
        goal.skill_name = skill_name
        goal.args_json = json.dumps(_embodied_public_args(args), sort_keys=True, separators=(",", ":"))
        goal.timeout_sec = float(timeout_sec or args.get("timeout_sec") or 120.0)
        goal.expected_schema_version = "mc_embodied_skills.v1"
        goal.parent_lease_id = str(args.get("parent_lease_id") or "")
        goal.resource_scope_id = str(args.get("resource_scope_id") or "")
        inherited = (
            args.get("inherited_lease_ids")
            if isinstance(args.get("inherited_lease_ids"), list)
            else []
        )
        goal.inherited_lease_ids = [str(item) for item in inherited]
        resources = (
            args.get("inherited_resources")
            if isinstance(args.get("inherited_resources"), list)
            else []
        )
        goal.inherited_resources = [str(item) for item in resources]

        ok, goal_handle_or_message = _wait_future(
            self._embodied_skill.send_goal_async(goal),
            timeout_sec=5.0,
            cancel_event=cancel_event,
        )
        if not ok:
            return ExecutionResult(False, f"{skill_name} goal send failed: {goal_handle_or_message}")
        goal_handle = goal_handle_or_message
        if not getattr(goal_handle, "accepted", False):
            return ExecutionResult(False, f"{skill_name} goal rejected")

        with self._current_lock:
            self._current_goal_handle = goal_handle
        try:
            ok, wrapped_result_or_message = _wait_future(
                goal_handle.get_result_async(),
                timeout_sec=goal.timeout_sec,
                cancel_event=cancel_event,
                on_cancel=lambda: _cancel_goal(goal_handle),
            )
            if not ok:
                return ExecutionResult(False, f"{skill_name} result failed: {wrapped_result_or_message}")
            wrapped = wrapped_result_or_message
            status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
            result = getattr(wrapped, "result", None)
            message = str(getattr(result, "message", "") or _status_name(status))
            skill_status = int(getattr(result, "status", EmbodiedSkill.Goal.STATUS_UNKNOWN))
            success = bool(getattr(result, "success", False))
            if status != GoalStatus.STATUS_SUCCEEDED or not success:
                return ExecutionResult(
                    False,
                    f"{skill_name} blocked/failed: {message}",
                    blocked=_is_blocked(skill_status),
                )
            facts = _loads_evidence_json(str(getattr(result, "evidence_json", "") or ""))
            execution = ExecutionResult(True, message or f"{skill_name} succeeded", facts)
            self._publish_success_facts(execution.facts)
            return execution
        finally:
            with self._current_lock:
                if self._current_goal_handle is goal_handle:
                    self._current_goal_handle = None

    def _play_animation_skill(
        self,
        skill_name: str,
        animation: str,
        animation_args: Any,
        raw_args: dict[str, Any],
        success_facts: dict[str, Any],
        cancel_event: Event | None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        goal = PlayAnimation.Goal()
        goal.animation = str(animation or "").strip()
        if not goal.animation:
            return ExecutionResult(False, f"{skill_name}.animation must be non-empty")
        try:
            goal.duration = float(raw_args.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            return ExecutionResult(False, f"{skill_name}.duration must be numeric")
        if isinstance(animation_args, dict):
            goal.args = json.dumps(animation_args, sort_keys=True, separators=(",", ":"))
        else:
            goal.args = str(animation_args or "")
        return self._run_action(
            skill_name,
            ActionBinding(
                self._play_animation,
                PlayAnimation,
                ("body",),
                120.0,
                success_facts,
            ),
            timeout_sec,
            goal,
            cancel_event,
        )

    def _say(self, args: dict[str, Any]) -> ExecutionResult:
        text = str(args.get("text") or "").strip()
        if not text:
            return ExecutionResult(False, "say text is empty")
        identity = self._lease_identity_msg()
        lease = self._leases.acquire(
            resources=("voice", "face"),
            reason="mc_ai_bt skill: say",
            timeout_sec=1.0,
            identity=identity,
        )
        if not lease.success:
            return ExecutionResult(False, f"resource lease denied: {lease.message}")
        try:
            msg = Utterance()
            msg.source = "ai_bt"
            msg.utterance = text
            msg.language = str(args.get("language") or "en-US")
            msg.confidence = 1.0
            self._speak_pub.publish(msg)
            result = ExecutionResult(True, "say submitted", {"say_submitted": text})
            self._publish_success_facts(result.facts)
            return result
        finally:
            self._leases.release(lease.lease_id, reason="say submitted", identity=identity)

    def cancel_current(self, reason: str = "mission canceled") -> None:
        with self._current_lock:
            handle = self._current_goal_handle
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(
                f"current action cancel failed ({reason}): {type(exc).__name__}: {exc}"
            )

    def _run_action(
        self,
        skill_name: str,
        binding: ActionBinding,
        node_timeout_sec: float | None,
        goal,
        cancel_event: Event | None,
    ) -> ExecutionResult:
        binding = _binding_with_node_timeout(binding, node_timeout_sec)
        if not _action_server_ready(binding.client, timeout_sec=1.0):
            return ExecutionResult(False, f"{skill_name} action server is not ready")
        if cancel_event is not None and cancel_event.is_set():
            return ExecutionResult(False, "mission canceled")
        return self._send_and_wait(skill_name, binding, goal, cancel_event)

    def _start_lease_renewal(
        self,
        lease_id: str,
        identity: AiBtIdentity,
    ) -> tuple[threading.Event, threading.Thread | None]:
        stop = threading.Event()
        renew = getattr(self._leases, "renew", None)
        if not lease_id or not callable(renew):
            return stop, None
        ttl_sec = float(getattr(self._leases, "ttl_sec", 300.0) or 300.0)
        interval = lease_renew_interval(ttl_sec)

        def _loop() -> None:
            while not stop.wait(interval):
                result = renew(lease_id, timeout_sec=1.0, identity=identity)
                if not getattr(result, "success", False):
                    message = str(getattr(result, "message", "renew failed"))
                    self._node.get_logger().warning(f"resource lease renew failed: {message}")

        thread = threading.Thread(target=_loop, name="mc_ai_bt_lease_renewal", daemon=True)
        thread.start()
        return stop, thread

    def _lease_identity_msg(self) -> AiBtIdentity:
        identity = getattr(self._identity_context, "identity", None)
        if isinstance(identity, Identity):
            return identity_to_msg(identity)
        return empty_identity_msg()

    def _send_and_wait(
        self,
        skill_name: str,
        binding: ActionBinding,
        goal,
        cancel_event: Event | None,
    ) -> ExecutionResult:
        ok, goal_handle_or_message = _wait_future(
            binding.client.send_goal_async(goal),
            timeout_sec=5.0,
            cancel_event=cancel_event,
        )
        if not ok:
            return ExecutionResult(False, f"{skill_name} goal send failed: {goal_handle_or_message}")

        goal_handle = goal_handle_or_message
        if not getattr(goal_handle, "accepted", False):
            return ExecutionResult(False, f"{skill_name} goal rejected")

        with self._current_lock:
            self._current_goal_handle = goal_handle
        try:
            ok, wrapped_result_or_message = _wait_future(
                goal_handle.get_result_async(),
                timeout_sec=binding.timeout_sec,
                cancel_event=cancel_event,
                on_cancel=lambda: _cancel_goal(goal_handle),
            )
            if not ok:
                return ExecutionResult(False, f"{skill_name} result failed: {wrapped_result_or_message}")

            wrapped = wrapped_result_or_message
            status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
            result = getattr(wrapped, "result", None)
            message = str(getattr(result, "message", "") or _status_name(status))
            success_field = bool(getattr(result, "success", status == GoalStatus.STATUS_SUCCEEDED))
            if status != GoalStatus.STATUS_SUCCEEDED:
                return ExecutionResult(False, f"{skill_name} ended with {_status_name(status)}: {message}")
            if not success_field:
                return ExecutionResult(False, f"{skill_name} result was unsuccessful: {message}")
            execution = ExecutionResult(
                True,
                message or f"{skill_name} succeeded",
                dict(binding.success_facts),
            )
            self._publish_success_facts(execution.facts)
            return execution
        finally:
            with self._current_lock:
                if self._current_goal_handle is goal_handle:
                    self._current_goal_handle = None

    def _build_world_state_writer(
        self,
        writer: WorldStateWriter | None,
    ) -> WorldStateWriter | None:
        if writer is not None:
            return writer
        try:
            return WorldStateWriter(self._node)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(
                f"world state writer unavailable: {type(exc).__name__}: {exc}"
            )
            return None

    def _publish_success_facts(self, facts: dict[str, Any]) -> None:
        if self._world_state is None:
            return
        for update in world_fact_updates_for_execution(facts):
            ok, message = self._world_state.update(update)
            if not ok:
                self._node.get_logger().debug(f"world state update skipped: {message}")


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
    step = 0.05
    while not done.wait(timeout=step):
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
    wait_for_server = getattr(client, "wait_for_server", None)
    if not callable(wait_for_server):
        return False
    try:
        return bool(wait_for_server(timeout_sec=timeout_sec))
    except TypeError:
        return bool(wait_for_server(timeout_sec))
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


def _embodied_public_args(args: dict[str, Any]) -> dict[str, Any]:
    control_keys = {
        "parent_lease_id",
        "resource_scope_id",
        "inherited_lease_ids",
        "inherited_resources",
        "timeout_sec",
    }
    return {key: value for key, value in args.items() if key not in control_keys}


def _confirmation_request_id(identity: AiBtIdentity) -> str:
    mission_id = str(identity.mission_id or "manual")
    prefix = mission_id[:12] if mission_id else "manual"
    return f"confirm-{prefix}-{uuid.uuid4().hex[:12]}"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return json.dumps({"text": text}, sort_keys=True, separators=(",", ":"))
        return text
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _confirmation_decision_name(decision: int) -> str:
    names = {
        RequestHumanConfirmation.Goal.DECISION_UNKNOWN: "UNKNOWN",
        RequestHumanConfirmation.Goal.DECISION_APPROVED: "APPROVED",
        RequestHumanConfirmation.Goal.DECISION_DENIED: "DENIED",
        RequestHumanConfirmation.Goal.DECISION_TIMEOUT: "TIMEOUT",
        RequestHumanConfirmation.Goal.DECISION_CANCELED: "CANCELED",
    }
    return names.get(decision, f"DECISION_{decision}")


def _loads_evidence_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _is_blocked(skill_status: int) -> bool:
    return skill_status in {
        EmbodiedSkill.Goal.STATUS_BLOCKED,
        EmbodiedSkill.Goal.STATUS_TIMEOUT,
    }


def _point_at_animation(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    arm = str(args.get("arm") or "right").strip().lower()
    animation = "left_point" if arm == "left" else "right_point"
    out: dict[str, Any] = {"arm": "left" if arm == "left" else "right"}

    target = str(args.get("target") or "").strip()
    has_explicit_target = any(key in args for key in ("object", "place", "x", "y", "z"))
    for key in (
        "object",
        "place",
        "x",
        "y",
        "z",
        "frame",
        "hold",
        "score_thr",
        "fresh",
        "max_age",
        "timeout",
        "place_distance",
        "place_z",
    ):
        if key in args:
            out[key] = args[key]
    if target and not has_explicit_target:
        out["object"] = target
    return animation, out


def _binding_with_node_timeout(
    binding: ActionBinding,
    node_timeout_sec: float | None,
) -> ActionBinding:
    if node_timeout_sec is None:
        return binding
    try:
        requested = float(node_timeout_sec)
    except (TypeError, ValueError):
        return binding
    if requested <= 0:
        return binding
    return ActionBinding(
        client=binding.client,
        goal_type=binding.goal_type,
        resources=binding.resources,
        timeout_sec=min(binding.timeout_sec, requested),
        success_facts=binding.success_facts,
    )
