from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .skill_registry import DEFAULT_SKILLS, SkillRegistry


ALLOWED_TOP_LEVEL_KEYS = {"schema", "root", "goal_spec", "context_json"}

# These five sets used to be independent, hand-written literals -- three separate
# production bugs in one afternoon (adding a single new skill, approach_entity) traced back
# to exactly this pattern: a name added to skill_registry.py but not copied into every one
# of these. Computed from DEFAULT_SKILLS instead; see skill_registry.py's SkillSpec
# docstring and OMEGACLAW_AI_BT_INTEGRATION.md §9 for the full story. Adding a new skill now
# means adding ONE entry to DEFAULT_SKILLS -- these five recompute automatically.
_BODY_RESOURCES = {"gaze", "left_arm", "right_arm", "body"}
BASE_SKILLS = {name for name, spec in DEFAULT_SKILLS.items() if "base" in spec.resources}
BODY_SKILLS = {
    name for name, spec in DEFAULT_SKILLS.items() if set(spec.resources) & _BODY_RESOURCES
}
PHYSICAL_SKILLS = BASE_SKILLS | BODY_SKILLS
CONTINUOUS_BASE_SKILLS = {name for name in BASE_SKILLS if DEFAULT_SKILLS[name].realtime}
POLICY_ENABLED_SKILLS = set(DEFAULT_SKILLS.keys())
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
    "entity",
    "entity_id",
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
POINT_AT_FRAMES = {
    "torso",
    "body",
    "body_ish",
    "chest_camera",
    "chest_camera_link",
    "camera",
    "base",
    "base_link",
}


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
    max_confirmation_prompt_chars: int = 240
    max_confirmation_timeout_sec: float = 300.0
    # WaitForEvent (§10.8) is deliberately NOT folded into max_wait_total_sec -- that
    # budget is for plain Wait's idle, no-information sleep (kept tight at 30s total).
    # WaitForEvent is the opposite: a bounded, productive poll this node type exists so
    # a plan doesn't need Retry(Sequence[Wait,Condition])'s ~1500s/5-attempt workaround
    # (see the capability audit's wait_for_event row). Its own ceiling is per-node, not
    # accumulated across a plan.
    min_wait_for_event_poll_sec: float = 0.5
    max_wait_for_event_poll_sec: float = 30.0
    max_wait_for_event_timeout_sec: float = 1800.0


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
        if node_type in {"Sequence", "Fallback", "Parallel"}:
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
            try:
                max_attempts = int(node.get("max_attempts", 1) or 1)
            except (TypeError, ValueError):
                max_attempts = 1
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
        if node_type == "Timeout":
            child = node.get("child")
            self._walk(
                child,
                f"{path}.child",
                stats,
                errors,
                retry_multiplier=retry_multiplier,
            )
            return
        if node_type == "NoAction":
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
        if node_type == "WaitForEvent":
            stats.conditions += retry_multiplier
            predicate = str(node.get("predicate") or "")
            if predicate.startswith("__") or "." in predicate:
                errors.append(f"{path}.predicate is not policy-safe: {predicate!r}")
            poll_interval_sec = node.get("poll_interval_sec", self._limits.min_wait_for_event_poll_sec)
            if not isinstance(poll_interval_sec, (int, float)) or not (
                self._limits.min_wait_for_event_poll_sec <= poll_interval_sec
                <= self._limits.max_wait_for_event_poll_sec
            ):
                errors.append(
                    f"{path}.poll_interval_sec must be in "
                    f"[{self._limits.min_wait_for_event_poll_sec:g}, "
                    f"{self._limits.max_wait_for_event_poll_sec:g}]"
                )
            timeout_sec = node.get("timeout_sec")
            if not isinstance(timeout_sec, (int, float)) or not (
                0 < timeout_sec <= self._limits.max_wait_for_event_timeout_sec
            ):
                errors.append(
                    f"{path}.timeout_sec must be in (0, {self._limits.max_wait_for_event_timeout_sec:g}]"
                )
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
        if skill in BASE_SKILLS or skill in CONTINUOUS_BASE_SKILLS:
            stats.base_actions += retry_multiplier
        if skill in BODY_SKILLS:
            stats.body_actions += retry_multiplier

        if not self._skills.has(skill):
            errors.append(f"{path}.skill is not registered: {skill!r}")
            return
        spec = self._skills.get(skill)
        if spec is not None and spec.status != "available":
            # Belt-and-braces: planner.py's prompt already omits non-available skills
            # from the catalog, but a plan reaching here with one anyway (a stale
            # cached plan, a hand-authored one, a future prompt regression) must be
            # rejected structurally, not discovered by trying it and getting
            # STATUS_BLOCKED back at execution time.
            errors.append(f"{path}.skill is not available (status={spec.status!r}): {skill!r}")
            return
        if skill == "say":
            text = str(args.get("text") or "")
            if len(text) > self._limits.max_say_chars:
                errors.append(f"{path}.args.text exceeds {self._limits.max_say_chars} chars")
        elif skill == "request_human_confirmation":
            self._check_human_confirmation(args, path, errors)
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
            # play_animation's nested "args" (generator args) is a raw
            # passthrough all the way to mc_animator, unlike point_at's
            # POINT_AT_ALLOWED_KEYS-checked args — a plan must never be able
            # to set "_caller" itself; that key identifies the REAL caller to
            # mc_animator's resource lease (see skill_adapters.py's
            # _play_animation_skill) and is only ever trustworthy when the
            # adapter sets it, not the planner.
            nested = args.get("args")
            if isinstance(nested, dict) and "_caller" in nested:
                errors.append(f"{path}.args.args must not set reserved key '_caller'")


        elif skill == "look_at":
            self._check_look_at(args, path, errors)
        elif skill == "point_at":
            self._check_point_at(args, path, errors)
        elif skill in {"locate_entity", "track_entity", "search_for_entity", "get_pose", "approach_entity", "face_entity"}:
            self._check_entity_target(args, path, errors, skill=skill)
        elif skill in {"look_at_static", "track_with_gaze"}:
            self._check_look_or_track(args, path, errors, skill=skill)
        elif skill == "track_frame":
            self._check_track_frame(args, path, errors)
        elif skill == "reach_to":
            self._check_reach_to(args, path, errors)
        elif skill == "check_relation":
            self._check_relation(args, path, errors)
        elif skill in {"align_axis", "move_along_axis"}:
            self._check_axis_skill(args, path, errors, skill=skill)
        elif skill in {"hold_pose", "wait_for_contact", "detect_contact", "oscillate", "retract"}:
            self._check_arm_generic(args, path, errors, skill=skill)
        elif skill in {"follow_entity", "maintain_distance"}:
            self._check_entity_target(args, path, errors, skill=skill)
        elif skill == "guide_entity_to_place":
            self._check_guide_entity_to_place(args, path, errors)
        elif skill == "wait_for_participant":
            self._check_wait_for_participant(args, path, errors)
        elif skill == "remember_place":
            self._check_remember_place(args, path, errors)
        elif skill == "remember_person":
            self._check_remember_person(args, path, errors)
        elif skill == "remember_entity":
            self._check_remember_entity(args, path, errors)
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

    def _check_human_confirmation(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        prompt = str(args.get("prompt") or args.get("text") or "")
        if not prompt.strip():
            errors.append(f"{path}.args.prompt must be non-empty")
        if len(prompt) > self._limits.max_confirmation_prompt_chars:
            errors.append(
                f"{path}.args.prompt exceeds {self._limits.max_confirmation_prompt_chars} chars"
            )
        role = str(args.get("required_role") or "operator").strip()
        if role not in {"operator", "owner", "admin", "safety_operator"}:
            errors.append(f"{path}.args.required_role is not allowed: {role!r}")
        timeout = _optional_float(args, "timeout_sec", path, errors)
        if timeout is not None and (
            timeout <= 0.0 or timeout > self._limits.max_confirmation_timeout_sec
        ):
            errors.append(
                f"{path}.args.timeout_sec must be in (0, {self._limits.max_confirmation_timeout_sec:g}]"
            )

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
        if not object_name:
            object_name = _clean_label(args.get("entity") or args.get("entity_id"))
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

    def _check_track_frame(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        frame = _clean_label(args.get("frame") or args.get("target_frame"))
        if not frame or len(frame) > 128:
            errors.append(f"{path}.args.frame must be a non-empty frame <= 128 chars")

    def _check_reach_to(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        arm = str(args.get("arm") or "right").strip().lower()
        if arm not in {"left", "right"}:
            errors.append(f"{path}.args.arm must be 'left' or 'right'")
        target_count = _embodied_target_count(args)
        if target_count != 1:
            errors.append(f"{path}.args must specify exactly one reach target")
        if all(key in args for key in ("x", "y", "z")):
            for key in ("x", "y", "z"):
                _required_float(args, key, path, errors)

    def _check_entity_target(
        self,
        args: dict[str, Any],
        path: str,
        errors: list[str],
        *,
        skill: str,
    ) -> None:
        target = _clean_label(
            args.get("entity")
            or args.get("entity_id")
            or args.get("target")
            or args.get("object")
            or args.get("person")
            or args.get("person_id")
            or args.get("frame")
        )
        if not target or len(target) > 128:
            errors.append(f"{path}.args target for {skill} must be non-empty and <= 128 chars")

    def _check_look_or_track(self, args: dict[str, Any], path: str, errors: list[str], *, skill: str) -> None:
        has_direction = bool(_clean_label(args.get("direction")))
        has_frame = bool(_clean_label(args.get("frame") or args.get("target_frame")))
        has_entity = any(
            _clean_label(args.get(key))
            for key in ("entity", "entity_id", "target", "object", "person", "person_id")
        )
        has_xyz = all(key in args for key in ("x", "y", "z"))
        if sum(bool(item) for item in (has_direction, has_frame, has_entity, has_xyz)) != 1:
            errors.append(f"{path}.args for {skill} must specify exactly one direction/frame/entity/x-y-z target")
        if has_direction and str(args.get("direction") or "").strip().lower() not in LOOK_AT_DIRECTIONS:
            errors.append(f"{path}.args.direction is not allowed: {args.get('direction')!r}")
        if has_xyz:
            for key in ("x", "y", "z"):
                _required_float(args, key, path, errors)

    def _check_relation(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        relation = _clean_label(args.get("relation") or args.get("predicate"))
        if not relation or len(relation) > 128:
            errors.append(f"{path}.args.relation must be non-empty and <= 128 chars")

    def _check_axis_skill(self, args: dict[str, Any], path: str, errors: list[str], *, skill: str) -> None:
        arm = str(args.get("arm") or "right").strip().lower()
        if arm not in {"left", "right"}:
            errors.append(f"{path}.args.arm must be 'left' or 'right'")
        axis = str(args.get("axis") or "").strip().lower()
        if axis not in {"x", "y", "z"}:
            errors.append(f"{path}.args.axis must be x, y, or z")
        if skill == "align_axis" and _embodied_target_count(args) != 1:
            errors.append(f"{path}.args for align_axis must specify exactly one target")
        if skill == "move_along_axis":
            distance = _optional_float(args, "distance_m", path, errors)
            if distance is None:
                errors.append(f"{path}.args.distance_m is required")

    def _check_arm_generic(self, args: dict[str, Any], path: str, errors: list[str], *, skill: str) -> None:
        arm = str(args.get("arm") or "right").strip().lower()
        if arm not in {"left", "right"}:
            errors.append(f"{path}.args.arm must be 'left' or 'right'")
        if skill not in {"retract"}:
            target_count = _embodied_target_count(args)
            if target_count != 1:
                errors.append(f"{path}.args for {skill} must specify exactly one target")
        for key in ("duration_sec", "min_duration_sec", "extent_m", "cycles"):
            _optional_float(args, key, path, errors)

    def _check_guide_entity_to_place(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        place = _clean_label(args.get("place") or args.get("place_name") or args.get("name"))
        if not place or len(place) > 64:
            errors.append(f"{path}.args.place must be non-empty and <= 64 chars")
        self._check_entity_target(args, path, errors, skill="guide_entity_to_place")

    def _check_wait_for_participant(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        participant = _clean_label(args.get("participant") or args.get("role") or args.get("entity"))
        if participant and len(participant) > 128:
            errors.append(f"{path}.args.participant must be <= 128 chars")
        condition = str(args.get("condition") or "present").strip().lower()
        if condition not in {"present", "hand_raised", "hand_offered", "facing_robot"}:
            errors.append(
                f"{path}.args.condition must be one of present/hand_raised/hand_offered/facing_robot"
            )

    def _check_remember_place(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        name = _clean_label(args.get("name") or args.get("place") or args.get("place_name"))
        if not name or len(name) > 64:
            errors.append(f"{path}.args.name must be non-empty and <= 64 chars")

    def _check_remember_person(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        self._check_entity_target(args, path, errors, skill="remember_person")
        name = _clean_label(args.get("name"))
        if not name or len(name) > 64:
            errors.append(f"{path}.args.name must be non-empty and <= 64 chars")

    def _check_remember_entity(self, args: dict[str, Any], path: str, errors: list[str]) -> None:
        self._check_entity_target(args, path, errors, skill="remember_entity")
        alias = _clean_label(args.get("alias"))
        if not alias or len(alias) > 64:
            errors.append(f"{path}.args.alias must be non-empty and <= 64 chars")

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
        # FOUND LIVE 2026-08-31 (real mission-journal audit): 83 of 84 planned missions
        # used goal type "human" / verification "implicit_conversation" -- "the mission
        # completed if execution didn't error" -- and most of those (66) contained a real
        # physical skill (go_to_place/approach_entity/play_animation/...). A skill's own
        # ROS-level SUCCEEDED is not the same claim as "the user's actual request was
        # satisfied"; for a mission that moves the robot's body or base, implicit
        # human/conversational success is not a real success criterion at all -- reject
        # the plan and force a structured predicate instead of silently accepting it.
        verification_mode = str((goal_spec.get("verification") or {}).get("mode") or "")
        if (
            stats.physical_skill_counts
            and (goal_spec.get("type") == "human" or verification_mode == "implicit_conversation")
        ):
            errors.append(
                "goal_spec uses implicit/human success verification but the plan contains "
                f"physical skill(s) {sorted(stats.physical_skill_counts)} -- a physical "
                "mission must declare a structured goal_spec.predicate, not rely on "
                "'execution did not error' as its success criterion"
            )
            return
        predicate = str(goal_spec.get("predicate") or "")
        if not predicate or not stats.physical_skill_counts:
            return
        expected = {
            "robot_at_place": ("go_to_place",),
            "entity_approached": ("approach_entity",),
            "robot_near_interaction_owner": ("come_to_me",),
            "relative_motion_completed": ("simple_move",),
            "animation_played": ("play_animation", "look_at", "point_at"),
            "reach_completed": ("reach_to",),
            "entity_following": ("follow_entity",),
            "entity_at_place": ("guide_entity_to_place",),
            "contact_detected": ("wait_for_contact", "detect_contact"),
            "axis_aligned": ("align_axis",),
            "axis_motion_completed": ("move_along_axis",),
            "distance_maintained": ("maintain_distance",),
            "pose_held": ("hold_pose",),
            "oscillation_completed": ("oscillate",),
            "retracted": ("retract",),
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


def _embodied_target_count(args: dict[str, Any]) -> int:
    has_xyz = all(key in args for key in ("x", "y", "z"))
    partial_xyz = any(key in args for key in ("x", "y", "z")) and not has_xyz
    if partial_xyz:
        return 0
    return sum(
        bool(_clean_label(args.get(key)))
        for key in ("entity", "entity_id", "object", "target", "person", "person_id", "place", "frame")
    ) + int(has_xyz)
