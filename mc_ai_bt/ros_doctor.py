from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str = ""
    kind: str = "check"


ServiceSpec = tuple[str, Any, str]
ActionSpec = tuple[str, Any, str]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import rclpy

    rclpy.init(args=None)
    node = rclpy.create_node("ai_bt_doctor")
    try:
        results = run_doctor(node, args)
    except KeyboardInterrupt:
        results = [DoctorCheck("interrupted", False, "interrupted by user", "doctor")]
    finally:
        node.destroy_node()
        _safe_rclpy_shutdown(rclpy)

    if args.json:
        print(json.dumps([asdict(result) for result in results], sort_keys=True))
    else:
        print(_format_results(results))
    return 0 if _overall_ok(results) else 1


def _safe_rclpy_shutdown(rclpy_module) -> None:
    try:
        if rclpy_module.ok():
            rclpy_module.shutdown()
    except Exception:
        pass


def run_doctor(node, args: argparse.Namespace) -> list[DoctorCheck]:
    results: list[DoctorCheck] = []
    service_clients = {}
    for label, srv_type, name in _core_services():
        client = node.create_client(srv_type, name)
        service_clients[label] = client
        ready = _wait_for_service(client, timeout_sec=args.timeout_sec)
        detail = name if ready else f"{name} unavailable after {args.timeout_sec:g}s"
        results.append(DoctorCheck(label, ready, detail, "service"))

    if not args.skip_read_only_calls:
        results.extend(_read_only_call_checks(node, service_clients, args.timeout_sec))

    if args.require_skill_actions:
        from rclpy.action import ActionClient

        for label, action_type, name in _skill_actions():
            client = ActionClient(node, action_type, name)
            ready = _wait_for_action(client, timeout_sec=args.timeout_sec)
            detail = name if ready else f"{name} unavailable after {args.timeout_sec:g}s"
            results.append(DoctorCheck(label, ready, detail, "action"))

    return results


def _read_only_call_checks(
    node,
    service_clients: dict[str, Any],
    timeout_sec: float,
) -> list[DoctorCheck]:
    checks: list[tuple[str, str, Callable[[], Any], Callable[[Any], tuple[bool, str]]]] = [
        (
            "call.ai_bt.list_missions",
            "ai_bt.list_missions",
            _list_missions_request,
            lambda response: (
                bool(getattr(response, "success", False)),
                str(getattr(response, "message", "") or "list_missions returned"),
            ),
        ),
        (
            "call.ai_bt.query_world",
            "ai_bt.query_world",
            _query_world_request,
            lambda response: (
                bool(getattr(response, "success", False)),
                str(getattr(response, "answer_text", "") or getattr(response, "message", "") or "query_world returned"),
            ),
        ),
        (
            "call.world_state.get_snapshot",
            "world_state.get_snapshot",
            _snapshot_request,
            lambda response: (
                bool(getattr(response, "success", False)),
                str(getattr(response, "message", "") or "snapshot returned"),
            ),
        ),
        (
            "call.resource_authority.list",
            "resource_authority.list",
            _list_leases_request,
            lambda response: (
                bool(getattr(response, "success", False)),
                str(getattr(response, "message", "") or "resource leases returned"),
            ),
        ),
    ]

    results: list[DoctorCheck] = []
    for label, service_label, request_factory, response_check in checks:
        client = service_clients.get(service_label)
        if client is None or not _service_ready_now(client):
            results.append(DoctorCheck(label, False, f"{service_label} is not ready", "call"))
            continue
        response, message = _call(node, client, request_factory(), timeout_sec=timeout_sec)
        if response is None:
            results.append(DoctorCheck(label, False, message, "call"))
            continue
        ok, detail = response_check(response)
        results.append(DoctorCheck(label, ok, detail, "call"))
    return results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only runtime doctor for the fan_bt AI-BT ROS stack."
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=2.0,
        help="Per-service/action wait timeout.",
    )
    parser.add_argument(
        "--require-skill-actions",
        action="store_true",
        help="Also require navigation and animator action servers used by skill adapters.",
    )
    parser.add_argument(
        "--skip-read-only-calls",
        action="store_true",
        help="Only check service readiness; do not call list/query/snapshot services.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args(argv)


def _core_services() -> tuple[ServiceSpec, ...]:
    from mc_one.srv import (
        CancelMission,
        GetWorldSnapshot,
        ListMissions,
        ListResourceLeases,
        PauseMission,
        QueryWorld,
        ResumeMission,
        SubmitTaskIntent,
        UpdateWorldFacts,
    )

    return (
        ("ai_bt.submit_task", SubmitTaskIntent, "/mc_ai_bt/submit_task"),
        ("ai_bt.cancel_mission", CancelMission, "/mc_ai_bt/cancel_mission"),
        ("ai_bt.pause_mission", PauseMission, "/mc_ai_bt/pause_mission"),
        ("ai_bt.resume_mission", ResumeMission, "/mc_ai_bt/resume_mission"),
        ("ai_bt.list_missions", ListMissions, "/mc_ai_bt/list_missions"),
        ("ai_bt.query_world", QueryWorld, "/mc_ai_bt/query_world"),
        ("world_state.get_snapshot", GetWorldSnapshot, "/mc_world_state/get_snapshot"),
        ("world_state.update_facts", UpdateWorldFacts, "/mc_world_state/update_facts"),
        ("resource_authority.list", ListResourceLeases, "/mc_resource_authority/list"),
    )


def _skill_actions() -> tuple[ActionSpec, ...]:
    from mc_one.action import ComeToMe, GoToPlace, PlayAnimation, SimpleMove

    return (
        ("skill.go_to_place", GoToPlace, "/mc_navigation/go_to_place"),
        ("skill.come_to_me", ComeToMe, "/mc_navigation/come_to_me"),
        ("skill.simple_move", SimpleMove, "/mc_navigation/simple_move"),
        ("skill.play_animation", PlayAnimation, "/mc_animator/play"),
    )


def _list_missions_request():
    from mc_one.srv import ListMissions

    request = ListMissions.Request()
    request.mission_id = ""
    request.include_terminal = True
    return request


def _query_world_request():
    from mc_one.srv import QueryWorld

    request = QueryWorld.Request()
    request.source = "ai_bt_doctor"
    request.query_text = "where are you?"
    request.scopes = []
    request.max_age_sec = 5.0
    request.context_json = json.dumps(
        {"schema": "mc_ai_bt.doctor_context.v1"},
        sort_keys=True,
        separators=(",", ":"),
    )
    return request


def _snapshot_request():
    from mc_one.srv import GetWorldSnapshot

    request = GetWorldSnapshot.Request()
    request.scopes = []
    request.max_age_sec = 0.0
    request.include_private = False
    request.query_json = "{}"
    return request


def _list_leases_request():
    from mc_one.srv import ListResourceLeases

    request = ListResourceLeases.Request()
    request.resources = []
    request.include_inactive = False
    return request


def _wait_for_service(client, *, timeout_sec: float) -> bool:
    try:
        return bool(client.wait_for_service(timeout_sec=max(0.0, timeout_sec)))
    except TypeError:
        return bool(client.wait_for_service(max(0.0, timeout_sec)))
    except Exception:
        return False


def _service_ready_now(client) -> bool:
    try:
        return bool(client.service_is_ready())
    except Exception:
        return False


def _wait_for_action(client, *, timeout_sec: float) -> bool:
    try:
        if client.server_is_ready():
            return True
    except Exception:
        return False
    try:
        return bool(client.wait_for_server(timeout_sec=max(0.0, timeout_sec)))
    except TypeError:
        return bool(client.wait_for_server(max(0.0, timeout_sec)))
    except Exception:
        return False


def _call(node, client, request, *, timeout_sec: float):
    import rclpy

    deadline = time.monotonic() + max(0.0, timeout_sec)
    try:
        future = client.call_async(request)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not future.done():
        return None, f"call timed out after {timeout_sec:g}s"
    try:
        return future.result(), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _overall_ok(results: list[DoctorCheck]) -> bool:
    return all(result.ok for result in results)


def _format_results(results: list[DoctorCheck]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        detail = f" -- {result.detail}" if result.detail else ""
        lines.append(f"{status} {result.kind} {result.name}{detail}")
    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    lines.append(f"summary: PASS {passed}  FAIL {failed}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
