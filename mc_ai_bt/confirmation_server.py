from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import monotonic

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from mc_one.action import RequestHumanConfirmation
from mc_one.msg import AiBtIdentity, HumanConfirmationStatus
from mc_one.srv import RespondHumanConfirmation


DECISION_UNKNOWN = 0
DECISION_APPROVED = 1
DECISION_DENIED = 2
DECISION_TIMEOUT = 3
DECISION_CANCELED = 4


@dataclass
class PendingConfirmation:
    identity: AiBtIdentity
    request_id: str
    prompt: str
    context_json: str
    required_role: str
    deadline_monotonic: float
    decision: int = DECISION_UNKNOWN
    responder_id: str = ""
    reason: str = ""
    condition: threading.Condition = field(default_factory=threading.Condition)


class HumanConfirmationServer(Node):
    def __init__(self) -> None:
        super().__init__("human_confirmation", namespace="/mc_ai_bt")
        self.declare_parameter("default_timeout_sec", 30.0)
        self.declare_parameter("max_timeout_sec", 300.0)
        self.declare_parameter("feedback_period_sec", 0.5)
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._pending: dict[str, PendingConfirmation] = {}
        self._status_pub = self.create_publisher(
            HumanConfirmationStatus,
            "/mc_ai_bt/human_confirmation_status",
            10,
        )
        self.create_service(
            RespondHumanConfirmation,
            "/mc_ai_bt/respond_human_confirmation",
            self._handle_response,
            callback_group=self._callback_group,
        )
        self._action_server = ActionServer(
            self,
            RequestHumanConfirmation,
            "/mc_ai_bt/request_human_confirmation",
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )

    def _goal_callback(self, goal: RequestHumanConfirmation.Goal) -> GoalResponse:
        request_id = str(goal.request_id or "").strip()
        prompt = str(goal.prompt or "").strip()
        if not request_id or not prompt:
            return GoalResponse.REJECT
        with self._lock:
            if request_id in self._pending:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> RequestHumanConfirmation.Result:
        goal = goal_handle.request
        timeout_sec = self._timeout_sec(float(getattr(goal, "timeout_sec", 0.0) or 0.0))
        pending = PendingConfirmation(
            identity=goal.identity,
            request_id=str(goal.request_id),
            prompt=str(goal.prompt),
            context_json=str(goal.context_json or ""),
            required_role=str(goal.required_role or ""),
            deadline_monotonic=monotonic() + timeout_sec,
        )
        with self._lock:
            self._pending[pending.request_id] = pending
        self._publish_status(pending, HumanConfirmationStatus.STATUS_PENDING)
        try:
            decision = self._wait_for_decision(goal_handle, pending)
        finally:
            with self._lock:
                self._pending.pop(pending.request_id, None)

        status = _status_for_decision(decision)
        self._publish_status(pending, status)
        result = RequestHumanConfirmation.Result()
        result.success = decision == DECISION_APPROVED
        result.decision = decision
        result.responder_id = pending.responder_id
        result.reason = pending.reason
        result.decided_at = self.get_clock().now().to_msg()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return result

    def _wait_for_decision(self, goal_handle, pending: PendingConfirmation) -> int:
        period = max(0.1, float(self.get_parameter("feedback_period_sec").value or 0.5))
        while True:
            remaining = max(0.0, pending.deadline_monotonic - monotonic())
            if goal_handle.is_cancel_requested:
                with pending.condition:
                    pending.decision = DECISION_CANCELED
                    pending.reason = pending.reason or "confirmation action canceled"
                    return DECISION_CANCELED
            with pending.condition:
                if pending.decision != DECISION_UNKNOWN:
                    return pending.decision
                if remaining <= 0.0:
                    pending.decision = DECISION_TIMEOUT
                    pending.reason = pending.reason or "confirmation timed out"
                    return DECISION_TIMEOUT
                pending.condition.wait(timeout=min(period, remaining))
            self._publish_feedback(goal_handle, pending)
            self._publish_status(pending, HumanConfirmationStatus.STATUS_PENDING)

    def _handle_response(
        self,
        request: RespondHumanConfirmation.Request,
        response: RespondHumanConfirmation.Response,
    ) -> RespondHumanConfirmation.Response:
        request_id = str(request.request_id or "").strip()
        decision = int(request.decision)
        if decision not in {DECISION_APPROVED, DECISION_DENIED, DECISION_CANCELED}:
            response.success = False
            response.message = f"unsupported confirmation decision: {decision}"
            response.status = self._status_from_request(request)
            return response

        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None:
            response.success = False
            response.message = f"unknown or completed confirmation request: {request_id}"
            response.status = self._status_from_request(request)
            return response
        if not _identity_matches_if_present(request.identity, pending.identity):
            response.success = False
            response.message = "confirmation identity does not match pending request"
            response.status = self._status_msg(pending, HumanConfirmationStatus.STATUS_PENDING)
            return response

        with pending.condition:
            if pending.decision != DECISION_UNKNOWN:
                response.success = False
                response.message = "confirmation request was already resolved"
            else:
                pending.decision = decision
                pending.responder_id = str(request.responder_id or "operator")
                pending.reason = str(request.reason or _reason_for_decision(decision))
                pending.condition.notify_all()
                response.success = True
                response.message = _reason_for_decision(decision)
        response.status = self._status_msg(pending, _status_for_decision(pending.decision))
        self._status_pub.publish(response.status)
        return response

    def _publish_feedback(self, goal_handle, pending: PendingConfirmation) -> None:
        feedback = RequestHumanConfirmation.Feedback()
        feedback.identity = pending.identity
        feedback.request_id = pending.request_id
        feedback.time_remaining_sec = float(max(0.0, pending.deadline_monotonic - monotonic()))
        feedback.status_text = "waiting for operator confirmation"
        try:
            goal_handle.publish_feedback(feedback)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(f"confirmation feedback skipped: {type(exc).__name__}: {exc}")

    def _publish_status(self, pending: PendingConfirmation, status: int) -> None:
        self._status_pub.publish(self._status_msg(pending, status))

    def _status_msg(self, pending: PendingConfirmation, status: int) -> HumanConfirmationStatus:
        msg = HumanConfirmationStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.identity = pending.identity
        msg.request_id = pending.request_id
        msg.prompt = pending.prompt
        msg.context_json = pending.context_json
        msg.required_role = pending.required_role
        msg.status = int(status)
        msg.responder_id = pending.responder_id
        msg.reason = pending.reason
        msg.time_remaining_sec = float(max(0.0, pending.deadline_monotonic - monotonic()))
        return msg

    def _status_from_request(
        self,
        request: RespondHumanConfirmation.Request,
    ) -> HumanConfirmationStatus:
        pending = PendingConfirmation(
            identity=request.identity,
            request_id=str(request.request_id or ""),
            prompt="",
            context_json="",
            required_role="",
            deadline_monotonic=monotonic(),
            decision=int(getattr(request, "decision", DECISION_UNKNOWN)),
            responder_id=str(getattr(request, "responder_id", "") or ""),
            reason=str(getattr(request, "reason", "") or ""),
        )
        return self._status_msg(pending, _status_for_decision(pending.decision))

    def _timeout_sec(self, requested: float) -> float:
        default = float(self.get_parameter("default_timeout_sec").value or 30.0)
        maximum = float(self.get_parameter("max_timeout_sec").value or 300.0)
        value = requested if requested > 0 else default
        return max(0.1, min(value, maximum))


def _identity_matches_if_present(candidate: AiBtIdentity, expected: AiBtIdentity) -> bool:
    mission_id = str(candidate.mission_id or "")
    if mission_id and mission_id != str(expected.mission_id or ""):
        return False
    plan_version = int(candidate.plan_version)
    if plan_version and plan_version != int(expected.plan_version):
        return False
    execution_id = str(candidate.execution_id or "")
    if execution_id and execution_id != str(expected.execution_id or ""):
        return False
    return True


def _status_for_decision(decision: int) -> int:
    if decision == DECISION_APPROVED:
        return HumanConfirmationStatus.STATUS_APPROVED
    if decision == DECISION_DENIED:
        return HumanConfirmationStatus.STATUS_DENIED
    if decision == DECISION_TIMEOUT:
        return HumanConfirmationStatus.STATUS_TIMEOUT
    if decision == DECISION_CANCELED:
        return HumanConfirmationStatus.STATUS_CANCELED
    return HumanConfirmationStatus.STATUS_PENDING


def _reason_for_decision(decision: int) -> str:
    if decision == DECISION_APPROVED:
        return "approved"
    if decision == DECISION_DENIED:
        return "denied"
    if decision == DECISION_CANCELED:
        return "canceled"
    if decision == DECISION_TIMEOUT:
        return "timed out"
    return "pending"


def main() -> None:
    rclpy.init()
    node = HumanConfirmationServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        _safe_rclpy_shutdown()


def _safe_rclpy_shutdown() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
