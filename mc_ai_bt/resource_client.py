from __future__ import annotations

import threading
from dataclasses import dataclass

from builtin_interfaces.msg import Duration
from mc_one.msg import AiBtIdentity
from mc_one.srv import AcquireResourceLease, ReleaseResourceLease, RenewResourceLease
from rclpy.node import Node

from .ros_identity import copy_identity_msg


@dataclass(frozen=True)
class LeaseResult:
    success: bool
    message: str
    lease_id: str = ""


class ResourceLeaseClient:
    def __init__(
        self,
        node: Node,
        *,
        acquire_service: str = "/mc_resource_authority/acquire",
        renew_service: str = "/mc_resource_authority/renew",
        release_service: str = "/mc_resource_authority/release",
        owner_id: str = "mc_ai_bt",
        priority: int = 20,
        ttl_sec: float = 300.0,
        callback_group=None,
    ) -> None:
        self._node = node
        self._acquire = node.create_client(
            AcquireResourceLease,
            acquire_service,
            callback_group=callback_group,
        )
        self._renew = node.create_client(
            RenewResourceLease,
            renew_service,
            callback_group=callback_group,
        )
        self._release = node.create_client(
            ReleaseResourceLease,
            release_service,
            callback_group=callback_group,
        )
        self._owner_id = owner_id
        self._priority = max(0, min(int(priority), 255))
        self._ttl_sec = float(ttl_sec)

    @property
    def ttl_sec(self) -> float:
        return self._ttl_sec

    def acquire(
        self,
        *,
        resources: tuple[str, ...],
        reason: str,
        timeout_sec: float = 3.0,
        identity: AiBtIdentity | None = None,
    ) -> LeaseResult:
        if not resources:
            return LeaseResult(True, "no resources required")
        if not self._acquire.service_is_ready():
            return LeaseResult(False, "resource authority acquire service is not ready")

        request = AcquireResourceLease.Request()
        request.identity = copy_identity_msg(identity)
        request.owner_id = self._owner_id
        request.resources = list(resources)
        request.priority = self._priority
        request.ttl = _duration_msg(self._ttl_sec)
        request.allow_preempt = True
        request.reason = reason

        ok, response_or_message = _wait_future(
            self._acquire.call_async(request),
            timeout_sec=timeout_sec,
        )
        if not ok:
            return LeaseResult(False, response_or_message)
        response = response_or_message
        return LeaseResult(
            bool(response.success),
            str(response.message),
            str(response.lease_id),
        )

    def renew(
        self,
        lease_id: str,
        *,
        timeout_sec: float = 1.0,
        identity: AiBtIdentity | None = None,
    ) -> LeaseResult:
        if not lease_id:
            return LeaseResult(False, "lease_id is required")
        if not self._renew.service_is_ready():
            return LeaseResult(False, "resource authority renew service is not ready", lease_id)
        request = RenewResourceLease.Request()
        request.identity = copy_identity_msg(identity)
        request.lease_id = lease_id
        request.ttl = _duration_msg(self._ttl_sec)
        request.reason = "renewed by mc_ai_bt"
        ok, response_or_message = _wait_future(
            self._renew.call_async(request),
            timeout_sec=timeout_sec,
        )
        if not ok:
            return LeaseResult(False, response_or_message, lease_id)
        response = response_or_message
        return LeaseResult(bool(response.success), str(response.message), lease_id)

    def release(
        self,
        lease_id: str,
        *,
        reason: str = "done",
        identity: AiBtIdentity | None = None,
    ) -> None:
        if not lease_id or not self._release.service_is_ready():
            return
        request = ReleaseResourceLease.Request()
        request.identity = copy_identity_msg(identity)
        request.lease_id = lease_id
        request.reason = reason
        self._release.call_async(request)


def _duration_msg(seconds: float) -> Duration:
    msg = Duration()
    bounded = max(0.0, float(seconds))
    whole = int(bounded)
    msg.sec = whole
    msg.nanosec = int((bounded - whole) * 1_000_000_000)
    return msg


def _wait_future(future, *, timeout_sec: float):
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
    if not done.wait(timeout=max(0.0, timeout_sec)):
        return False, "future timed out"
    if "error" in box:
        exc = box["error"]
        return False, f"{type(exc).__name__}: {exc}"
    return True, box.get("result")
