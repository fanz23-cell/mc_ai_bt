from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from .executor import BtExecutor, ExecutionResult
from .goal_check import GoalChecker, TriState
from .planner import BootstrapPlanner
from .policy_guard import PolicyGuard
from .validator import PlanValidator


@dataclass(frozen=True)
class SmokeResult:
    ok: bool
    stage: str
    message: str
    plan_json: str = ""
    facts: dict[str, Any] | None = None


class FakeSkillExecutor:
    def execute_skill(
        self,
        name: str,
        args: dict[str, Any],
        cancel_event=None,
        *,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        if name == "say":
            return ExecutionResult(True, "say simulated", {"say_submitted": args.get("text", "")})
        if name == "go_to_place":
            return ExecutionResult(True, "nav simulated", {"robot_at_place": args.get("name", "")})
        if name == "come_to_me":
            return ExecutionResult(True, "come_to_me simulated", {"robot_near_interaction_owner": True})
        if name == "simple_move":
            return ExecutionResult(
                True,
                "simple_move simulated",
                {
                    "relative_motion_completed": {
                        "action": args.get("action", ""),
                        "value": args.get("value"),
                    }
                },
            )
        if name == "play_animation":
            return ExecutionResult(True, "animation simulated", {"animation_played": args.get("animation", "")})
        if name == "look_at":
            return ExecutionResult(
                True,
                "look_at simulated",
                {
                    "animation_played": "look_at",
                    "look_at": {"direction": args.get("direction", "front")},
                },
            )
        if name == "point_at":
            return ExecutionResult(
                True,
                "point_at simulated",
                {
                    "animation_played": "point_at",
                    "point_at": dict(args),
                },
            )
        return ExecutionResult(False, f"unsupported fake skill: {name}")


def run_smoke(
    intent_text: str,
    *,
    context_json: str = "{}",
    world_json: str = "",
) -> SmokeResult:
    planner = BootstrapPlanner()
    plan_json = planner.plan(intent_text, context_json)

    validation = PlanValidator().validate_json(plan_json)
    if not validation.ok:
        return SmokeResult(False, "validator", "; ".join(validation.errors), plan_json)

    plan = json.loads(plan_json)
    policy = PolicyGuard().check(plan)
    if not policy.ok:
        return SmokeResult(False, "policy", "; ".join(policy.errors), plan_json)

    checker = GoalChecker(_snapshot_provider(world_json))
    execution = BtExecutor().execute(plan["root"], FakeSkillExecutor(), checks=checker)
    if not execution.success:
        return SmokeResult(False, "executor", execution.message, plan_json, execution.facts)

    check = checker.check(plan["goal_spec"], execution)
    if check.state is not TriState.TRUE:
        return SmokeResult(False, "goal_check", f"{check.state.value}: {check.message}", plan_json, execution.facts)

    return SmokeResult(True, "done", check.message, plan_json, execution.facts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local AI-BT smoke without ROS/DDS.")
    parser.add_argument("intent", nargs="+", help="Intent text to plan and execute with fake skills.")
    parser.add_argument("--context-json", default="{}", help="Planner context JSON.")
    parser.add_argument(
        "--world-json",
        default="",
        help="Optional world snapshot JSON for local Condition/GoalCheck/VisualCheck nodes.",
    )
    parser.add_argument("--print-plan", action="store_true", help="Print the generated plan JSON.")
    args = parser.parse_args(argv)

    result = run_smoke(
        " ".join(args.intent),
        context_json=args.context_json,
        world_json=args.world_json,
    )
    if args.print_plan:
        print(result.plan_json)
    print(json.dumps(_result_dict(result), sort_keys=True))
    return 0 if result.ok else 1


def _result_dict(result: SmokeResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "stage": result.stage,
        "message": result.message,
        "facts": result.facts or {},
    }


def _snapshot_provider(world_json: str):
    if not world_json:
        return None

    def _provide(_scopes, _max_age):
        return world_json

    return _provide


if __name__ == "__main__":
    raise SystemExit(main())
