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
from mc_one.srv import SavePlaceHere
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
    # Optional Callable[[feedback, result], tuple[bool, dict[str, Any]]]. When set, called
    # after the ROS action itself reports SUCCEEDED, with the LAST feedback message seen
    # (or None) and the terminal result -- real, independently-produced telemetry, not
    # something this adapter asserted about itself. Returning (False, ...) turns a
    # ROS-level success into an ExecutionResult(False, ...): the action controller's own
    # "I did it" is not enough on its own. Returning (True, facts) uses `facts` in place of
    # the static success_facts. None (every skill except go_to_place/come_to_me right now)
    # keeps today's behavior exactly -- see verify_go_to_place/verify_come_to_me for why
    # only these two currently have real corroborating telemetry to check against.
    verify: Any = None


# go_to_place/come_to_me's own ARRIVAL_DISTANCE_TOLERANCE_M -- both actions' Feedback
# carries distance_remaining, "metres left to the goal, as Nav2's planner estimates it"
# (GoToPlace.action/ComeToMe.action's own comments), mirrored from real NavigateToPose
# feedback -- Nav2's own telemetry, not something this adapter computed or asserted itself.
_ARRIVAL_DISTANCE_TOLERANCE_M = 1.0


def _arrival_verifier(facts_on_success: dict[str, Any]):
    """go_to_place/come_to_me's real (if partial) success verification.

    FOUND LIVE 2026-08-31: before this, ROS-level SUCCEEDED alone was enough --
    success_facts (for come_to_me, a literal hardcoded {"...": True}) were applied
    unconditionally, and goal_check later read that same self-report back as if it were
    independent evidence. This checks the goal's own last-reported distance_remaining
    against a real, small tolerance before trusting SUCCEEDED at all.

    HONEST LIMIT, not papered over: this is corroboration against Nav2's own telemetry,
    not a true independently-measured final pose. mc_ai_bt has no TF capability and
    mc_world_state publishes no robot-pose fact (checked live 2026-08-29, see
    mc_embodied_skills' _face_entity_skill's own comment on the same gap) -- closing this
    the rest of the way needs one of those to exist first. This is deliberately scoped to
    what's achievable with data already flowing through the action interface tonight.
    """

    def _verify(feedback, _result) -> tuple[bool, dict[str, Any]]:
        distance = getattr(feedback, "distance_remaining", None) if feedback is not None else None
        if distance is not None:
            if float(distance) > _ARRIVAL_DISTANCE_TOLERANCE_M:
                return False, {}
            facts = dict(facts_on_success)
            facts["_verification_basis"] = "distance_remaining_confirmed"
            return True, facts
        # FOUND LIVE 2026-09-01: a goal that completes before its first feedback tick
        # (the common real case: the robot was already at/near the target, so Nav2
        # never entered its "driving" phase at all) must not be punished with a false
        # rejection here just because no feedback exists to check -- distance is
        # genuinely None, not "far". But silently treating that the same as an
        # actually-confirmed reading would recreate the exact gap this verifier exists
        # to close, just one layer down: something claiming "verified" that never
        # really was. Still accept (today's behavior, no new false failures), but mark
        # it plainly as unverified rather than letting it masquerade as confirmed.
        facts = dict(facts_on_success)
        facts["_verification_basis"] = "no_feedback_received_trusted_at_face_value"
        return True, facts

    return _verify


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
        self._save_place_here = node.create_client(SavePlaceHere, "/mc_navigation/save_place_here")
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
                    verify=_arrival_verifier({"robot_at_place": goal.name}),
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
                    verify=_arrival_verifier({"robot_near_interaction_owner": True}),
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
        if name == "remember_place":
            return self._remember_place(args)
        if name in EMBODIED_SKILLS:
            return self._run_embodied_skill(name, args, cancel_event, timeout_sec=timeout_sec)
        return ExecutionResult(False, f"unsupported skill: {name}")

    def _remember_place(self, args: dict[str, Any]) -> ExecutionResult:
        # SavePlaceHere (mc_navigation) is a plain service, not an action -- unlike every
        # other dedicated skill above, there is no goal/result lifecycle, just a request/
        # response. It does its own map->base_link TF lookup and yaw derivation server-side
        # (nav_orchestrator.py's _on_save_place_here), so this adapter only needs the name.
        name = str(args.get("name") or args.get("place") or args.get("place_name") or "").strip()
        if not name:
            return ExecutionResult(False, "remember_place requires name")
        if not _service_ready(self._save_place_here, timeout_sec=1.0):
            return ExecutionResult(False, "remember_place service is not ready", blocked=True)
        request = SavePlaceHere.Request()
        request.name = name
        ok, response_or_message = _wait_future(self._save_place_here.call_async(request), timeout_sec=10.0)
        if not ok:
            return ExecutionResult(False, f"remember_place failed: {response_or_message}")
        response = response_or_message
        if not bool(getattr(response, "success", False)):
            return ExecutionResult(False, f"remember_place failed: {getattr(response, 'message', '')}")
        execution = ExecutionResult(
            True,
            str(getattr(response, "message", "") or f"remembered place {name}"),
            {"place_remembered": name},
        )
        return self._finalize(execution)

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
                return self._finalize(execution)
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
                blocked = _is_blocked(skill_status)
                # A2: evidence_json used to be silently discarded on this whole
                # branch -- only parsed when blocked, so a skill's own
                # "_resolution_request" marker (see decision_broker.py's
                # ACTION_POSTCONDITION_POLICY / executor.py's
                # _resolve_action_postcondition) can actually reach the executor.
                # Harmless for every skill that doesn't use this: its facts just
                # go unread, exactly as before.
                blocked_facts = _loads_evidence_json(str(getattr(result, "evidence_json", "") or "")) if blocked else {}
                return ExecutionResult(
                    False,
                    f"{skill_name} blocked/failed: {message}",
                    blocked_facts,
                    blocked=blocked,
                )
            facts = _loads_evidence_json(str(getattr(result, "evidence_json", "") or ""))
            execution = ExecutionResult(True, message or f"{skill_name} succeeded", facts)
            return self._finalize(execution)
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
            payload = dict(animation_args)
        elif animation_args:
            # play_animation's own "args" is a raw passthrough (unlike
            # look_at/point_at, which build animation_args themselves from
            # named fields) — a plan can legally hand this generator payload
            # over pre-serialized as a JSON string. Parse it so "_caller" can
            # still be merged in below.
            try:
                parsed = json.loads(animation_args) if isinstance(animation_args, str) else None
            except (TypeError, ValueError):
                parsed = None
            payload = parsed if isinstance(parsed, dict) else None
        else:
            payload = {}
        if payload is None:
            # FAIL CLOSED, not open. A mission request that can't be tagged
            # with a real caller identity would fall back to mc_animator's
            # default identity — indistinguishable from idle or any other
            # untagged caller — silently reproducing the exact preemption bug
            # this whole change exists to close. Better to refuse the
            # animation than to send it unprotected.
            return ExecutionResult(
                False, f"{skill_name}.args is malformed (not a JSON object): {animation_args!r}"
            )
        # PlayAnimation.action has no identity field of its own (unlike
        # EmbodiedSkill/RequestHumanConfirmation) — "_caller" rides inside
        # the generator-args JSON instead, which the .action's own
        # contract already promises ignores unknown keys. mc_animator's
        # resource lease reads it to tell "this mission" apart from
        # another caller entirely, instead of every /mc_animator/play
        # request looking identical to mc_resource_authority. See
        # resource_gate.py's ResourceLeaseGate.acquire.
        #
        # TRUST BOUNDARY: "_caller" is trusted execution metadata ONLY
        # because this method is the sole place that sets it, unconditionally
        # overwriting anything a plan supplied under the same key (see the
        # payload construction above and policy_guard.py's belt-and-braces
        # rejection of a planner-supplied "_caller"). It is NOT an
        # authenticated identity at the ROS layer — PlayAnimation.action
        # carries it as plain JSON text, so any other node with a client for
        # /mc_animator/play could claim to be "mc_ai_bt" and inherit its
        # preemption priority. That is out of scope here (mc_animator's
        # gate/mission planner are the only current senders, per the
        # trust-boundary audit), but a real fix — identity as first-class
        # Action metadata set by mc_animator's own goal-request middleware,
        # not caller-supplied JSON — is the correct long-term shape.
        identity = self._lease_identity_msg()
        payload["_caller"] = {
            "source": str(identity.source or "mc_ai_bt"),
            "operator_id": str(identity.operator_id or ""),
            "mission_id": str(identity.mission_id or ""),
            "execution_id": str(identity.execution_id or ""),
        }
        goal.args = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
            return self._finalize(result)
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
        # Every dedicated-dispatch skill (go_to_place/come_to_me/simple_move/
        # play_animation/look_at/point_at, all of them via ActionBinding.resources)
        # goes through here -- before this, only `say` ever acquired a resource
        # lease at all, so two of these could race on the same base/body resource
        # with no arbitration, only whichever ROS goal happened to preempt the
        # other at the animator/behavior-server level. Found live 2026-08-29
        # (§C5): a Parallel of look_at + look_at_static genuinely fought over
        # gaze, "preempted" by raw single-goal action-server semantics, not a
        # graceful lease denial. Same lease client `say` already uses -- not a
        # new mechanism.
        binding = _binding_with_node_timeout(binding, node_timeout_sec)
        if not _action_server_ready(binding.client, timeout_sec=1.0):
            return ExecutionResult(False, f"{skill_name} action server is not ready")
        if cancel_event is not None and cancel_event.is_set():
            return ExecutionResult(False, "mission canceled")
        identity = self._lease_identity_msg()
        lease = self._leases.acquire(
            resources=binding.resources,
            reason=f"mc_ai_bt skill: {skill_name}",
            timeout_sec=1.0,
            identity=identity,
        )
        if not lease.success:
            return ExecutionResult(False, f"resource lease denied: {lease.message}")
        stop_renewal, renewal_thread = self._start_lease_renewal(lease.lease_id, identity)
        try:
            return self._send_and_wait(skill_name, binding, goal, cancel_event)
        finally:
            stop_renewal.set()
            if renewal_thread is not None:
                renewal_thread.join(timeout=1.0)
            self._leases.release(lease.lease_id, reason=f"{skill_name} finished", identity=identity)

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
        last_feedback_box: dict[str, Any] = {}

        def _on_feedback(feedback_msg) -> None:
            last_feedback_box["value"] = getattr(feedback_msg, "feedback", feedback_msg)

        ok, goal_handle_or_message = _wait_future(
            binding.client.send_goal_async(goal, feedback_callback=_on_feedback),
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
            if binding.verify is not None:
                verified, facts = binding.verify(last_feedback_box.get("value"), result)
                if not verified:
                    return ExecutionResult(
                        False,
                        f"{skill_name} action reported success but independent verification "
                        f"did not confirm it: {message}",
                    )
            else:
                facts = dict(binding.success_facts)
            execution = ExecutionResult(True, message or f"{skill_name} succeeded", facts)
            return self._finalize(execution)
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

    def _publish_success_facts(self, facts: dict[str, Any]) -> str:
        """Returns "" when every world-state write either succeeded or is
        legitimately best-effort (the ordinary facts below always are --
        losing a robot.last_animation write is not worth failing a mission
        over). Returns a non-empty error message ONLY when entity_alias_bound
        itself was rejected by the identity registry -- callers MUST
        downgrade their own ExecutionResult from success to failure in that
        case.

        FOUND LIVE 2026-09-03 (GPT review, Gate-1): this used to return None
        unconditionally, called AFTER the caller had already built an
        ExecutionResult(success=True, ...) -- a rejected bind_entity_alias
        call (identity collision, alias conflict, live track expired between
        the skill's own locate and this call, service down) was only ever
        logged at debug level, with the skill's own SUCCESS untouched. That
        is a direct violation of this whole system's core principle (a
        skill claiming success must never disagree with the real world) --
        entity_alias_bound is not a best-effort fact, it is the skill's own
        actual claimed outcome (`remember_entity: bound {alias!r} to
        {entity_id}`), so if the identity registry refuses it, the mission
        genuinely did not succeed."""
        if self._world_state is None:
            return ""
        for update in world_fact_updates_for_execution(facts):
            ok, message = self._world_state.update(update)
            if not ok:
                self._node.get_logger().debug(f"world state update skipped: {message}")

        # 2026-09-03 architecture consolidation: entity_alias_bound is a
        # domain command (bind_entity_alias), not a generic fact write --
        # see world_facts.py's own comment on why this is handled here
        # directly rather than folded into world_fact_updates_for_execution.
        entity_alias_bound = facts.get("entity_alias_bound")
        if (
            isinstance(entity_alias_bound, dict)
            and entity_alias_bound.get("alias")
            and entity_alias_bound.get("entity_id")
            and hasattr(self._world_state, "bind_entity_alias")
        ):
            ok, message = self._world_state.bind_entity_alias(
                alias=str(entity_alias_bound["alias"]),
                entity_class=str(entity_alias_bound.get("entity_class") or ""),
                live_entity_id=str(entity_alias_bound["entity_id"]),
                created_by=str(entity_alias_bound.get("created_by") or ""),
            )
            if not ok:
                self._node.get_logger().warning(f"bind_entity_alias rejected: {message}")
                return f"entity_alias_bound was not accepted by the identity registry: {message}"
        return ""

    def _finalize(self, execution: ExecutionResult) -> ExecutionResult:
        """The one call site every _*_skill/action handler below must route
        its final success-shaped ExecutionResult through: publishes world-
        state facts, and downgrades success to failure if entity_alias_bound
        (when present) was rejected by the identity registry -- see
        _publish_success_facts's own docstring for why this cannot be a
        fire-and-forget side effect."""
        if not execution.success:
            return execution
        bind_error = self._publish_success_facts(execution.facts)
        if bind_error:
            return ExecutionResult(False, bind_error, execution.facts, blocked=True)
        return execution


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


def _service_ready(client, *, timeout_sec: float) -> bool:
    try:
        if client.service_is_ready():
            return True
    except Exception:
        return False
    try:
        return bool(client.wait_for_service(timeout_sec=timeout_sec))
    except Exception:
        return False


def _cancel_goal(goal_handle) -> None:
    # FOUND LIVE 2026-09-01: this used to fire cancel_goal_async() and return
    # immediately -- "requested a cancel" is not the same claim as "the action server
    # actually accepted the cancel", and nothing here ever knew the difference. Waits
    # briefly (bounded, best-effort) for that acknowledgement; still returns either way
    # -- the caller already treats the goal as canceled regardless (see _wait_future's
    # own on_cancel contract), so this only trades a small bounded delay for a real
    # confirmation instead of none at all. Does NOT wait for the goal's own terminal
    # status (CANCELED) -- that is a separate, potentially much longer wait for the
    # controller to actually stop, which _wait_future's caller does not block on today.
    try:
        future = goal_handle.cancel_goal_async()
    except Exception:
        return
    _wait_future(future, timeout_sec=1.0)


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
        verify=binding.verify,
    )
