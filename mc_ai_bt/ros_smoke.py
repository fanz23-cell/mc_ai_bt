from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import rclpy
from mc_one.msg import TaskStatus
from mc_one.srv import (
    ListMissions,
    ListResourceLeases,
    QueryWorld,
    SubmitTaskIntent,
)


TERMINAL_STATES = {
    TaskStatus.STATE_SUCCEEDED,
    TaskStatus.STATE_FAILED,
    TaskStatus.STATE_CANCELED,
    TaskStatus.STATE_BLOCKED,
}


@dataclass
class SmokeSummary:
    ok: bool
    stage: str
    message: str
    mission_id: str = ""
    final_state: int = 0
    final_status_text: str = ""
    query_answers: list[dict[str, Any]] = field(default_factory=list)
    active_leases: int = 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    launch_proc: subprocess.Popen | None = None
    if not args.no_launch:
        launch_proc = _start_fake_launch(args)

    rclpy.init(args=None)
    node = rclpy.create_node("ai_bt_local_fake_ros_smoke")
    try:
        summary = _run_smoke(node, args)
    except KeyboardInterrupt:
        summary = SmokeSummary(False, "interrupted", "interrupted by user")
    finally:
        node.destroy_node()
        _safe_rclpy_shutdown()
        if launch_proc is not None and not args.keep_launch:
            _stop_launch(launch_proc)
        if getattr(args, "generated_mission_journal", False) and not args.keep_launch:
            _unlink_quietly(args.mission_journal_path)

    print(json.dumps(summary.__dict__, sort_keys=True))
    return 0 if summary.ok else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a ROS end-to-end smoke for fan_bt core using the local fake "
            "navigation/animator/perception launch."
        )
    )
    parser.add_argument(
        "--intent",
        default="go to test_place then point at the test_object",
        help="Task intent submitted to /mc_ai_bt/submit_task.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="Overall wait timeout for services and mission completion.",
    )
    parser.add_argument(
        "--mission-journal-path",
        default="",
        help="Mission journal path passed to fan_bt_local_fake.launch.py.",
    )
    parser.add_argument(
        "--fake-delay-sec",
        type=float,
        default=0.02,
        help="Fake action completion delay passed to fan_bt_local_fake.launch.py.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Do not start fan_bt_local_fake.launch.py; use an already-running stack.",
    )
    parser.add_argument(
        "--keep-launch",
        action="store_true",
        help="Leave the launched fake stack running after the smoke finishes.",
    )
    parser.add_argument(
        "--skip-query-world",
        action="store_true",
        help="Skip read-only /mc_ai_bt/query_world checks.",
    )
    parser.add_argument(
        "--skip-lease-check",
        action="store_true",
        help="Skip active resource lease drain check.",
    )
    args = parser.parse_args(argv)
    args.generated_mission_journal = not bool(args.mission_journal_path)
    if args.generated_mission_journal:
        args.mission_journal_path = f"/tmp/mc_ai_bt_fake_missions_{os.getpid()}.jsonl"
    return args


def _start_fake_launch(args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/ros2_launch_logs")
    if getattr(args, "generated_mission_journal", False):
        _unlink_quietly(args.mission_journal_path)
    cmd = [
        "ros2",
        "launch",
        "mc_ai_bt",
        "fan_bt_local_fake.launch.py",
        f"mission_journal_path:={args.mission_journal_path}",
        f"fake_delay_sec:={args.fake_delay_sec:g}",
    ]
    return subprocess.Popen(cmd, env=env)


def _stop_launch(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=3.0)


def _safe_rclpy_shutdown() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def _unlink_quietly(path: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _run_smoke(node, args: argparse.Namespace) -> SmokeSummary:
    deadline = time.monotonic() + max(1.0, float(args.timeout_sec))
    submit = node.create_client(SubmitTaskIntent, "/mc_ai_bt/submit_task")
    missions = node.create_client(ListMissions, "/mc_ai_bt/list_missions")
    query = node.create_client(QueryWorld, "/mc_ai_bt/query_world")
    leases = node.create_client(ListResourceLeases, "/mc_resource_authority/list")

    required = [
        (submit, "/mc_ai_bt/submit_task"),
        (missions, "/mc_ai_bt/list_missions"),
    ]
    if not args.skip_query_world:
        required.append((query, "/mc_ai_bt/query_world"))
    if not args.skip_lease_check:
        required.append((leases, "/mc_resource_authority/list"))

    ok, message = _wait_services(required, deadline)
    if not ok:
        return SmokeSummary(False, "wait_services", message)

    accepted, mission_id, message = _submit_task(node, submit, args.intent, deadline)
    if not accepted:
        return SmokeSummary(False, "submit_task", message, mission_id=mission_id)

    status, message = _wait_mission_terminal(node, missions, mission_id, deadline)
    if status is None:
        return SmokeSummary(False, "mission_terminal", message, mission_id=mission_id)
    if int(status.state) != TaskStatus.STATE_SUCCEEDED:
        return SmokeSummary(
            False,
            "mission_terminal",
            status.status_text or f"mission ended in state {status.state}",
            mission_id=mission_id,
            final_state=int(status.state),
            final_status_text=status.status_text,
        )

    answers: list[dict[str, Any]] = []
    if not args.skip_query_world:
        for question in ("where are you?", "do you see the test_object?"):
            answer = _query_world(node, query, question, deadline)
            answers.append(answer)
            if not answer.get("success"):
                return SmokeSummary(
                    False,
                    "query_world",
                    str(answer.get("message") or question),
                    mission_id=mission_id,
                    final_state=int(status.state),
                    final_status_text=status.status_text,
                    query_answers=answers,
                )

    active_leases = 0
    if not args.skip_lease_check:
        active_leases, message = _wait_active_leases_empty(node, leases, deadline)
        if active_leases != 0:
            return SmokeSummary(
                False,
                "resource_leases",
                message,
                mission_id=mission_id,
                final_state=int(status.state),
                final_status_text=status.status_text,
                query_answers=answers,
                active_leases=active_leases,
            )

    return SmokeSummary(
        True,
        "done",
        "fan_bt local fake ROS smoke passed",
        mission_id=mission_id,
        final_state=int(status.state),
        final_status_text=status.status_text,
        query_answers=answers,
        active_leases=active_leases,
    )


def _wait_services(clients: list[tuple[Any, str]], deadline: float) -> tuple[bool, str]:
    waiting = list(clients)
    while waiting and time.monotonic() < deadline:
        waiting = [
            (client, name)
            for client, name in waiting
            if not client.wait_for_service(timeout_sec=0.25)
        ]
    if waiting:
        return False, "service(s) unavailable: " + ", ".join(name for _client, name in waiting)
    return True, "ok"


def _submit_task(
    node,
    client,
    intent: str,
    deadline: float,
) -> tuple[bool, str, str]:
    request = SubmitTaskIntent.Request()
    request.source = "local_fake_ros_smoke"
    request.operator_id = "local"
    request.intent_text = intent
    request.parent_mission_id = ""
    request.priority = 10
    request.allow_queue = True
    request.context_json = json.dumps(
        {"schema": "mc_ai_bt.ros_smoke_context.v1"},
        sort_keys=True,
        separators=(",", ":"),
    )
    response, message = _call(node, client, request, deadline)
    if response is None:
        return False, "", message
    identity = getattr(response, "identity", None)
    mission_id = str(getattr(identity, "mission_id", "") or "")
    return bool(response.accepted), mission_id, str(response.message or "")


def _wait_mission_terminal(node, client, mission_id: str, deadline: float):
    last_message = "mission did not reach terminal state"
    while time.monotonic() < deadline:
        request = ListMissions.Request()
        request.mission_id = mission_id
        request.include_terminal = True
        response, message = _call(node, client, request, deadline)
        if response is None:
            last_message = message
            time.sleep(0.1)
            continue
        statuses = list(getattr(response, "statuses", []) or [])
        if statuses:
            status = statuses[0]
            if int(getattr(status, "state", 0)) in TERMINAL_STATES:
                return status, "terminal"
            last_message = status.status_text or f"state={status.state}"
        time.sleep(0.25)
    return None, last_message


def _query_world(node, client, question: str, deadline: float) -> dict[str, Any]:
    request = QueryWorld.Request()
    request.source = "local_fake_ros_smoke"
    request.query_text = question
    request.scopes = []
    request.max_age_sec = 5.0
    request.context_json = json.dumps(
        {"schema": "mc_ai_bt.ros_smoke_context.v1"},
        sort_keys=True,
        separators=(",", ":"),
    )
    response, message = _call(node, client, request, deadline)
    if response is None:
        return {"question": question, "success": False, "message": message}
    return {
        "question": question,
        "success": bool(response.success),
        "message": str(response.message or ""),
        "certainty": int(response.certainty),
        "answer_text": str(response.answer_text or ""),
    }


def _wait_active_leases_empty(node, client, deadline: float) -> tuple[int, str]:
    last_count = -1
    while time.monotonic() < deadline:
        request = ListResourceLeases.Request()
        request.resources = []
        request.include_inactive = False
        response, message = _call(node, client, request, deadline)
        if response is None:
            return -1, message
        last_count = len(list(getattr(response, "statuses", []) or []))
        if last_count == 0:
            return 0, "no active leases"
        time.sleep(0.25)
    return last_count, f"{last_count} active lease(s) remained"


def _call(node, client, request, deadline: float):
    future = client.call_async(request)
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not future.done():
        return None, "service call timed out"
    try:
        return future.result(), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
