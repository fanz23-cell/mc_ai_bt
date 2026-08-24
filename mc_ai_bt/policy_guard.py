from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .skill_registry import SkillRegistry


ALLOWED_TOP_LEVEL_KEYS = {"schema", "root", "goal_spec", "context_json"}
PHYSICAL_SKILLS = {"go_to_place", "come_to_me", "simple_move", "play_animation", "look_at", "point_at"}
BASE_SKILLS = {"go_to_place", "come_to_me", "simple_move"}
BODY_SKILLS = {"play_animation", "look_at", "point_at"}
POLICY_ENABLED_SKILLS = {
    "say",
    "go_to_place",
    "come_to_me",
    "simple_move",
    "play_animation",
    "look_at",
    "point_at",
}
LOOK_AT_DIRECTIONS = {
    "front",
    "front_up",
    "front_down",
    "left",
    "left_up",
    "left_down",
    "right",
    "right_up",
    "right_down",
}
POINT_AT_ALLOWED_KEYS = {
    "arm",
    "object",
    "target",
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
    "duration",
}
POINT_AT_FRAMES = {"torso", "body", "body_ish", "chest_camera", "chest_camera_link", "camera", "base", "base_link"}


@dataclass(frozen=True)
class PolicyResult:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PolicyLimits:
    max_total_nodes: int = 32
    max_total_actions: int = 12
    max_physical_actions: int = 4
    max_base_actions: int = 3
    max_body_actions: int = 3
    max_visual_checks: int = 4
    max_conditions: int = 8
    max_wait_total_sec: float = 30.0
    max_retry_product: int = 5
    max_say_chars: int = 500
    max_forward_m: float = 1.0
    max_turn_deg: float = 180.0
    max_animation_duration_sec: float = 30.0
    max_look_hold_sec: float = 10.0
    max_point_hold_sec: float = 10.0
    max_point_abs_m: float = 3.0
    min_point_z_m: float = -0.2
    max_point_z_m: float = 2.2


class PolicyGuard:
    """Robot policy checks that sit after schema validation.

    The validator answers "is this a well-formed BT using known node/skill
    names?". PolicyGuard answers "is this bounded enough to let a planner,
    especially an LLM planner, run on the robot right now?".
    """

    def __init__(
        self,
        *,
        skill_registry: SkillRegistry | None = None,
        limits: PolicyLimits | None = None,
    ) -> None:
        self._skills = skill_registry or SkillRegistry()
        self._limits = limits or PolicyLimits()

    def check(self, plan: dict[str, Any]) -> PolicyResult:
        errors: list[str] = []
        if not isinstance(plan, dict):
            return PolicyResult(False, ("plan must be an object",))

        unknown_keys = sorted(set(plan) - ALLOWED_TOP_LEVEL_KEYS)
        if unknown_keys:
            errors.append(f"unknown top-level keys: {unknown_keys}")

        root = plan.get("root")
        goal_spec = plan.get("goal_spec")
        stats = _Stats()
        self._walk(root, "root", stats, errors, retry_multiplier=1)
        self._check_totals(stats, errors)
        self._check_goal_alignment(goal_spec, stats, errors)
        return PolicyResult(not errors, tuple(errors))

    def _walk(
        self,
        node: Any,
        path: str,
        stats: "_Stats",
        errors: list[str],
        *,
        retry_multiplier: int,
    ) -> None:
        if not isinstance(node, dict):
            return
        stats.total_nodes += retry_multiplier
        node_type = node.get("type")
        if node_type in {"Sequence", "Fallback"}:
            for idx, child in enumerate(node.get("children", []) or []):
                self._walk(
                    child,
                    f"{path}.children[{idx}]",
                    stats,
                    errors,
                    retry_multiplier=retry_multiplier,
                )
            return
        if node_type == "Retry":
            max_attempts = int(node.get("max_attempts", 1) or 1)
            stats.retry_product *= max_attempts
            child = node.get("child")
            self._walk(
                child,
                f"{path}.child",
                stats,
                errors,
                retry_multiplier=retry_multiplier * max_attempts,
            )
            return
        if node_type == "Wait":
            duration = float(node.get("duration_sec", 0.0) or 0.0)
            stats.wait_total_sec += duration * retry_multiplier
            return
        if node_type == "Action":
            self._check_action(node, path, stats, errors, retry_multiplier=retry_multiplier)
            return
        if node_type == "Condition":
            stats.conditions += retry_multiplier
            predicate = str(node.get("predicate") or "")
            if predicate.startswith("__") or "." in predicate:
                errors.append(f"{path}.predicate is not policy-safe: {predicate!r}")
            return
        if node_type == "GoalCheck":
            check = node.get("check") if isinstance(node.get("check"), dict) else {}
            predicate = str(check.get("predicate") or "")
            if predicate.startswith("__") or "." in predicate:
                errors.append(f"{path}.check.predicate is not policy-safe: {predicate!r}")
            return
        if node_type == "VisualCheck":
            stats.visual_checks += retry_multiplier
            check = node.get("check") if isinstance(node.get("check"), dict) else {}
            predicate = str(check.get("predicate") or "")
            if predicate.startswith("__") or "." in predicate:
                errors.append(f"{path}.check.predicate is not policy-safe: {predicate!r}")

    def _check_action(
        self,
        node: dict[str, Any],
        path: str,
        stats: "_Stats",
        errors: list[str],
        *,
        retry_multiplier: int,
    ) -> None:
        skill = str(node.get("skill") or "")
        args = node.get("args") or {}
        stats.total_actions += retry_multiplier
        if skill in PHYSICAL_SKILLS:
            stats.physical_actions += retry_multiplier
            stats.physical_skill_counts[skill] = stats.physical_skill_counts.get(skill, 0) + retry_multiplier
        if skill in BASE_SKILLS:
            stats.base_actions += retry_multiplier
        if skill in BODY_SKILLS:
            stats.body_actions += retry_multiplier

        if not self._skills.has(skill):
            errors.append(f"{path}.skill is not registered: {skill!r}")
            return
        if skill == "say":
            text = str(args.get("text") or "")
            if len(text) > self._limits.max_say_chars:
                errors.append(f"{path}.args.text exceeds {self._limits.max_say_chars} chars")
        elif skill == "simple_move":
            self._check_simple_move(args, path, errors)
        elif skill == "go_to_place":
            name = str(args.get("name") or args.get("place") or "")
            if not name or len(name) > 64:
                errors.append(f"{path}.args.name must be a non-empty place slug <= 64 chars")
        elif skill == "play_animation":
            animation = str(args.get("animation") or args.get("name") or "")
            if not animation or len(animation) > 64:
                errors.append(f"{path}.args.animation must be non-empty and <= 64 chars")
            self._check_optional_duration(args, path, errors)
        elif skill == "look_at":
            self._check_look_at(args, path, errors)
        elif skill == "point_at":
            self._check_point_at(args, path, errors)
        elif skill not in POLICY_ENABLED_SKILLS:
            errors.append(f"{path}.skill is registered but not policy-enabled yet: {skill}")

    def _check_simple_move(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        action = str(args.get("action") or "")
        try:
            value = float(args.get("value"))
        except (TypeError, ValueError):
            errors.append(f"{path}.args.value must be numeric")
            return
        if action in {"forward", "backward"} and abs(value) > self._limits.max_forward_m:
            errors.append(f"{path}.args.value exceeds {self._limits.max_forward_m:g}m")
        elif action in {"left", "right"} and abs(value) > self._limits.max_turn_deg:
            errors.append(f"{path}.args.value exceeds {self._limits.max_turn_deg:g}deg")
        elif action not in {"forward", "backward", "left", "right"}:
            errors.append(f"{path}.args.action is not allowed: {action!r}")

    def _check_look_at(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        direction = str(args.get("direction") or "front").strip().lower()
        if direction not in LOOK_AT_DIRECTIONS:
            errors.append(f"{path}.args.direction is not allowed: {direction!r}")
        hold = _optional_float(args, "hold", path, errors)
        if hold is not None and (hold < 0.0 or hold > self._limits.max_look_hold_sec):
            errors.append(f"{path}.args.hold must be in [0, {self._limits.max_look_hold_sec:g}]")
        self._check_optional_duration(args, path, errors)

    def _check_point_at(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        unknown = sorted(set(args) - POINT_AT_ALLOWED_KEYS)
        if unknown:
            errors.append(f"{path}.args has unknown point_at keys: {unknown}")

        arm = str(args.get("arm") or "right").strip().lower()
        if arm not in {"left", "right"}:
            errors.append(f"{path}.args.arm must be 'left' or 'right'")

        frame = str(args.get("frame") or "torso").strip().lower()
        if frame not in POINT_AT_FRAMES:
            errors.append(f"{path}.args.frame is not allowed: {frame!r}")

        object_name = _clean_label(args.get("object") or args.get("target"))
        place_name = _clean_label(args.get("place"))
        has_xyz = all(key in args for key in ("x", "y", "z"))
        partial_xyz = any(key in args for key in ("x", "y", "z")) and not has_xyz
        if partial_xyz:
            errors.append(f"{path}.args.x/y/z must be provided together")
        target_count = sum(bool(item) for item in (object_name, place_name)) + int(has_xyz)
        if target_count != 1:
            errors.append(f"{path}.args must specify exactly one target: object/target, place, or x/y/z")
        if object_name and len(object_name) > 64:
            errors.append(f"{path}.args.object must be <= 64 chars")
        if place_name and len(place_name) > 64:
            errors.append(f"{path}.args.place must be <= 64 chars")

        if has_xyz:
            x = _required_float(args, "x", path, errors)
            y = _required_float(args, "y", path, errors)
            z = _required_float(args, "z", path, errors)
            if x is not None and abs(x) > self._limits.max_point_abs_m:
                errors.append(f"{path}.args.x exceeds {self._limits.max_point_abs_m:g}m")
            if y is not None and abs(y) > self._limits.max_point_abs_m:
                errors.append(f"{path}.args.y exceeds {self._limits.max_point_abs_m:g}m")
            if z is not None and not (self._limits.min_point_z_m <= z <= self._limits.max_point_z_m):
                errors.append(
                    f"{path}.args.z must be in "
                    f"[{self._limits.min_point_z_m:g}, {self._limits.max_point_z_m:g}]"
                )

        hold = _optional_float(args, "hold", path, errors)
        if hold is not None and (hold < 0.0 or hold > self._limits.max_point_hold_sec):
            errors.append(f"{path}.args.hold must be in [0, {self._limits.max_point_hold_sec:g}]")
        for key in ("score_thr", "max_age", "timeout", "place_distance", "place_z"):
            _optional_float(args, key, path, errors)
        if "fresh" in args and not isinstance(args.get("fresh"), bool):
            errors.append(f"{path}.args.fresh must be boolean")
        self._check_optional_duration(args, path, errors)

    def _check_optional_duration(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        duration = _optional_float(args, "duration", path, errors)
        if duration is not None and (duration < 0.0 or duration > self._limits.max_animation_duration_sec):
            errors.append(
                f"{path}.args.duration must be in [0, {self._limits.max_animation_duration_sec:g}]"
            )

    def _check_totals(self, stats: "_Stats", errors: list[str]) -> None:
        limits = self._limits
        if stats.total_nodes > limits.max_total_nodes:
            errors.append(f"too many BT nodes: {stats.total_nodes} > {limits.max_total_nodes}")
        if stats.total_actions > limits.max_total_actions:
            errors.append(f"too many actions: {stats.total_actions} > {limits.max_total_actions}")
        if stats.physical_actions > limits.max_physical_actions:
            errors.append(
                f"too many physical actions: {stats.physical_actions} > {limits.max_physical_actions}"
            )
        if stats.base_actions > limits.max_base_actions:
            errors.append(f"too many base actions: {stats.base_actions} > {limits.max_base_actions}")
        if stats.body_actions > limits.max_body_actions:
            errors.append(f"too many body actions: {stats.body_actions} > {limits.max_body_actions}")
        if stats.visual_checks > limits.max_visual_checks:
            errors.append(f"too many visual checks: {stats.visual_checks} > {limits.max_visual_checks}")
        if stats.conditions > limits.max_conditions:
            errors.append(f"too many conditions: {stats.conditions} > {limits.max_conditions}")
        if stats.wait_total_sec > limits.max_wait_total_sec:
            errors.append(f"wait budget exceeds {limits.max_wait_total_sec:g}s")
        if stats.retry_product > limits.max_retry_product:
            errors.append(f"retry product exceeds {limits.max_retry_product}")

    def _check_goal_alignment(
        self,
        goal_spec: Any,
        stats: "_Stats",
        errors: list[str],
    ) -> None:
        if not isinstance(goal_spec, dict):
            return
        predicate = str(goal_spec.get("predicate") or "")
        if not predicate or not stats.physical_skill_counts:
            return
        expected = {
            "robot_at_place": ("go_to_place",),
            "robot_near_interaction_owner": ("come_to_me",),
            "relative_motion_completed": ("simple_move",),
            "animation_played": ("play_animation", "look_at", "point_at"),
        }.get(predicate)
        if expected and not any(skill in stats.physical_skill_counts for skill in expected):
            errors.append(
                f"goal predicate {predicate!r} does not match physical action "
                f"(expected one of {', '.join(expected)})"
            )


@dataclass
class _Stats:
    total_nodes: int = 0
    total_actions: int = 0
    physical_actions: int = 0
    base_actions: int = 0
    body_actions: int = 0
    visual_checks: int = 0
    conditions: int = 0
    wait_total_sec: float = 0.0
    retry_product: int = 1
    physical_skill_counts: dict[str, int] = field(default_factory=dict)


def _optional_float(
    args: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> float | None:
    if key not in args:
        return None
    return _required_float(args, key, path, errors)


def _required_float(
    args: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> float | None:
    value = args.get(key)
    if isinstance(value, bool):
        errors.append(f"{path}.args.{key} must be numeric")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{path}.args.{key} must be numeric")
        return None


def _clean_label(value: Any) -> str:
    return str(value or "").strip()
