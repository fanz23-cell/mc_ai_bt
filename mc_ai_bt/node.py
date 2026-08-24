from __future__ import annotations

import threading
import json

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from mc_one.msg import AiBtIdentity, TaskEvent, TaskStatus, WorldEvent
from mc_one.srv import (
    CancelMission,
    ListMissions,
    PauseMission,
    QueryWorld,
    ResumeMission,
    SubmitTaskIntent,
)

from .context_builder import ContextBuilder
from .executor import BtExecutor
from .goal_check import GoalChecker, TriState
from .identity import Identity
from .mission import (
    Mission,
    MissionEvent,
    MissionManager,
    STATE_BLOCKED,
    STATE_CANCELED,
    STATE_FAILED,
    STATE_PLANNING,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)
from .mission_journal import MissionJournal
from .policy_guard import PolicyGuard
from .planner_factory import PlannerSettings, build_planner
from .planning_pipeline import PlanningPipeline
from .skill_adapters import RosSkillExecutor
from .skill_registry import SkillRegistry
from .task_projection import task_projection_updates
from .query_world import QueryWorldEngine
from .ros_identity import identity_to_msg
from .resource_client import ResourceLeaseClient
from .trigger_manager import (
    ACTION_BLOCK_ACTIVE,
    ACTION_REPLAN_ACTIVE,
    TriggerDecision,
    TriggerManager,
    trigger_decision_matches_mission,
)
from .validator import PlanValidator
from .world_state_client import WorldStateClient, WorldStateWriter


def _runner_key(identity: Identity) -> str:
    return identity.execution_id or identity.mission_id


class AiBtNode(Node):
    def __init__(self) -> None:
        super().__init__("manager", namespace="/mc_ai_bt")
        self._skill_registry = SkillRegistry()
        self._declare_parameters()
        self._mission_journal = self._build_mission_journal()
        recovered_missions = self._load_journal_missions()
        self._missions = MissionManager(recovered_missions)
        self._planner = self._build_planner()
        self._validator = PlanValidator(self._skill_registry)
        self._policy_guard = PolicyGuard(skill_registry=self._skill_registry)
        self._executor = BtExecutor()
        self._client_callback_group = ReentrantCallbackGroup()
        self._world_state = WorldStateClient(
            self,
            callback_group=self._client_callback_group,
        )
        self._world_writer = WorldStateWriter(
            self,
            callback_group=self._client_callback_group,
        )
        self._goal_checker = GoalChecker(self._world_state.snapshot_json)
        self._context_builder = ContextBuilder(
            snapshot_provider=self._world_state.snapshot_json,
            skill_registry=self._skill_registry,
        )
        self._planning = PlanningPipeline(
            planner=self._planner,
            context_builder=self._context_builder,
            validator=self._validator,
            policy_guard=self._policy_guard,
        )
        self._lease_client = ResourceLeaseClient(
            self,
            callback_group=self._client_callback_group,
        )
        self._skill_executor = RosSkillExecutor(
            self,
            leases=self._lease_client,
            world_state=self._world_writer,
        )
        self._trigger_manager = TriggerManager()
        self._query_world = QueryWorldEngine()
        self._mission_lock = threading.RLock()
        self._runner_threads: list[threading.Thread] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._mission_cancel_keys: dict[str, str] = {}
        self._status_pub = self.create_publisher(TaskStatus, "/mc_ai_bt/task_status", 10)
        self._event_pub = self.create_publisher(TaskEvent, "/mc_ai_bt/task_events", 10)
        self.create_service(SubmitTaskIntent, "/mc_ai_bt/submit_task", self._handle_submit)
        self.create_service(CancelMission, "/mc_ai_bt/cancel_mission", self._handle_cancel)
        self.create_service(PauseMission, "/mc_ai_bt/pause_mission", self._handle_pause)
        self.create_service(ResumeMission, "/mc_ai_bt/resume_mission", self._handle_resume)
        self.create_service(ListMissions, "/mc_ai_bt/list_missions", self._handle_list)
        self.create_service(QueryWorld, "/mc_ai_bt/query_world", self._handle_query_world)
        self.create_subscription(
            WorldEvent,
            "/mc_world_state/events",
            self._on_world_event,
            10,
        )
        self._startup_snapshot_publishes_remaining = 3 if recovered_missions else 0
        self._startup_snapshot_timer = None
        if self._startup_snapshot_publishes_remaining:
            self._startup_snapshot_timer = self.create_timer(
                1.0,
                self._publish_startup_snapshot,
            )

    def _declare_parameters(self) -> None:
        self.declare_parameter("planner_backend", "bootstrap")
        self.declare_parameter("planner_model_name", "")
        self.declare_parameter("planner_temperature", 0.0)
        self.declare_parameter("planner_timeout", 15.0)
        self.declare_parameter("mission_journal_path", "")

    def _build_mission_journal(self) -> MissionJournal | None:
        path = str(self.get_parameter("mission_journal_path").value or "").strip()
        if not path:
            return None
        return MissionJournal(path)

    def _load_journal_missions(self) -> tuple[Mission, ...]:
        if self._mission_journal is None:
            return ()
        try:
            missions = self._mission_journal.load_latest_missions(recover_nonterminal=True)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"mission journal load failed: {type(exc).__name__}: {exc}"
            )
            return ()
        if missions:
            self.get_logger().warning(
                f"loaded {len(missions)} mission(s) from journal; "
                "non-terminal missions are blocked for manual review"
            )
        return missions

    def _build_planner(self):
        settings = PlannerSettings(
            backend=str(self.get_parameter("planner_backend").value or "bootstrap"),
            model_name=str(self.get_parameter("planner_model_name").value or ""),
            temperature=float(self.get_parameter("planner_temperature").value or 0.0),
            timeout_sec=float(self.get_parameter("planner_timeout").value or 15.0),
        )
        try:
            return build_planner(
                settings,
                skill_registry=self._skill_registry,
                logger=self.get_logger(),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"planner backend {settings.backend!r} unavailable "
                f"({type(exc).__name__}: {exc}); falling back to bootstrap"
            )
            return build_planner(
                PlannerSettings(),
                skill_registry=self._skill_registry,
                logger=self.get_logger(),
            )

    def _handle_submit(
        self,
        request: SubmitTaskIntent.Request,
        response: SubmitTaskIntent.Response,
    ) -> SubmitTaskIntent.Response:
        with self._mission_lock:
            accepted, message, mission, event = self._missions.submit(
                intent_text=request.intent_text,
                source=request.source,
                operator_id=request.operator_id,
                parent_mission_id=request.parent_mission_id,
                priority=request.priority,
                allow_queue=request.allow_queue,
                context_json=request.context_json,
            )
        self._publish_event(event)
        response.accepted = accepted
        response.message = message
        response.identity = self._identity_msg(mission.identity)
        if not accepted:
            return response
        if mission.state == STATE_QUEUED:
            return response

        self._plan_and_start(mission)
        return response

    def _plan_and_start(self, mission: Mission) -> None:
        expected_identity = mission.identity
        with self._mission_lock:
            missions = self._missions.all()
        planning = self._planning.plan(mission, missions)
        if not planning.ok:
            with self._mission_lock:
                try:
                    failed = self._missions.mark_terminal(
                        mission.identity.mission_id,
                        state=STATE_FAILED,
                        message=f"{planning.stage}: {planning.message}",
                        expected_identity=expected_identity,
                    )
                except KeyError as exc:
                    self.get_logger().debug(f"stale planning failure ignored: {exc}")
                    return
            self._publish_event(failed)
            self._start_next_ready()
            return

        with self._mission_lock:
            try:
                planned = self._missions.set_plan(
                    mission.identity.mission_id,
                    bt_json=planning.bt_json,
                    goal_spec_json=planning.goal_spec_json,
                    expected_identity=expected_identity,
                )
            except KeyError as exc:
                self.get_logger().debug(f"stale plan ignored: {exc}")
                return
        self._publish_event(planned)
        self._start_runner(planned.mission)

    def _start_runner(self, mission: Mission) -> None:
        cancel_event = threading.Event()
        runner_key = _runner_key(mission.identity)
        with self._mission_lock:
            self._cancel_events[runner_key] = cancel_event
            self._mission_cancel_keys[mission.identity.mission_id] = runner_key
        thread = threading.Thread(
            target=self._run_mission,
            args=(mission, cancel_event, runner_key),
            name=f"mission-{mission.identity.mission_id[:8]}",
            daemon=True,
        )
        self._runner_threads.append(thread)
        thread.start()

    def _run_mission(
        self,
        mission: Mission,
        cancel_event: threading.Event,
        runner_key: str,
    ) -> None:
        try:
            with self._skill_executor.use_identity(mission.identity):
                execution = self._executor.execute_json(
                    mission.bt_json,
                    self._skill_executor,
                    cancel_event=cancel_event,
                    checks=self._goal_checker,
                    progress_callback=lambda active_node, progress: self._on_execution_progress(
                        mission,
                        active_node,
                        progress,
                    ),
                )
            state = STATE_FAILED
            message = execution.message
            if execution.blocked:
                state = STATE_BLOCKED
                message = execution.message
            elif execution.success:
                check = self._goal_checker.check_json(mission.goal_spec_json, execution)
                if check.state is TriState.TRUE:
                    state = STATE_SUCCEEDED
                    message = f"goal check TRUE: {check.message}"
                elif check.state is TriState.UNKNOWN:
                    state = STATE_BLOCKED
                    message = f"goal check UNKNOWN: {check.message}"
                else:
                    state = STATE_FAILED
                    message = f"goal check FALSE: {check.message}"

            with self._mission_lock:
                if not self._missions.accepts_async_result(mission.identity):
                    return
                terminal = self._missions.mark_terminal(
                    mission.identity.mission_id,
                    state=state,
                    message=message,
                )
            self._publish_event(terminal)
            self._start_next_ready()
        finally:
            with self._mission_lock:
                if self._mission_cancel_keys.get(mission.identity.mission_id) == runner_key:
                    self._mission_cancel_keys.pop(mission.identity.mission_id, None)
                self._cancel_events.pop(runner_key, None)

    def _start_next_ready(self) -> None:
        with self._mission_lock:
            ready = [
                mission
                for mission in self._missions.all()
                if mission.state == STATE_PLANNING and not mission.bt_json
            ]
        if ready:
            self._plan_and_start(ready[0])

    def _handle_cancel(
        self,
        request: CancelMission.Request,
        response: CancelMission.Response,
    ) -> CancelMission.Response:
        try:
            with self._mission_lock:
                target_id = self._missions.resolve_control_id(request.mission_id)
                runner_key = self._mission_cancel_keys.get(target_id, "")
                cancel_event = self._cancel_events.get(runner_key)
                event = self._missions.cancel(request.mission_id, request.reason)
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            return response
        if cancel_event is not None:
            cancel_event.set()
            self._skill_executor.cancel_current(request.reason or "mission canceled")
        self._publish_event(event)
        self._start_next_ready()
        response.success = True
        response.message = event.message
        return response

    def _handle_pause(
        self,
        request: PauseMission.Request,
        response: PauseMission.Response,
    ) -> PauseMission.Response:
        try:
            with self._mission_lock:
                target_id = self._missions.resolve_control_id(request.mission_id)
                runner_key = self._mission_cancel_keys.get(target_id, "")
                cancel_event = self._cancel_events.get(runner_key)
                event = self._missions.pause(request.mission_id, request.reason)
            if cancel_event is not None:
                cancel_event.set()
                self._skill_executor.cancel_current(request.reason or "mission paused")
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            return response
        self._publish_event(event)
        response.success = True
        response.message = event.message
        return response

    def _handle_resume(
        self,
        request: ResumeMission.Request,
        response: ResumeMission.Response,
    ) -> ResumeMission.Response:
        try:
            with self._mission_lock:
                event = self._missions.resume(request.mission_id, request.reason)
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            return response
        self._publish_event(event)
        if event.mission.state == STATE_RUNNING and event.mission.bt_json:
            self._start_runner(event.mission)
        elif event.mission.state == STATE_PLANNING and not event.mission.bt_json:
            self._plan_and_start(event.mission)
        response.success = True
        response.message = event.message
        return response

    def _handle_list(
        self,
        request: ListMissions.Request,
        response: ListMissions.Response,
    ) -> ListMissions.Response:
        try:
            with self._mission_lock:
                missions = self._select_missions(
                    request.mission_id,
                    include_terminal=bool(request.include_terminal),
                )
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            response.statuses = []
            return response
        response.success = True
        response.message = f"{len(missions)} mission(s)"
        response.statuses = [self._status_msg(mission) for mission in missions]
        return response

    def _handle_query_world(
        self,
        request: QueryWorld.Request,
        response: QueryWorld.Response,
    ) -> QueryWorld.Response:
        scopes = tuple(scope for scope in request.scopes if scope.strip())
        if not scopes:
            scopes = self._query_world.scopes_for_query(request.query_text)
        max_age_sec = float(request.max_age_sec if request.max_age_sec > 0 else 5.0)
        snapshot = self._world_state.snapshot(scopes, max_age_sec)
        if snapshot is None:
            response.success = False
            response.message = "world_state snapshot is unavailable"
            response.certainty = 0
            response.answer_text = "I cannot read the current world state right now."
            response.evidence_json = "{}"
            return response

        answer = self._query_world.answer_json(
            request.query_text,
            str(getattr(snapshot, "world_json", "") or ""),
        )
        response.success = answer.success
        response.message = answer.message
        response.certainty = answer.certainty
        response.answer_text = answer.answer_text
        response.evidence_json = answer.evidence_json
        response.snapshot = snapshot
        return response

    def _on_world_event(self, msg: WorldEvent) -> None:
        decision = self._trigger_manager.handle_world_event(
            event_type=msg.event_type,
            snapshot_id=msg.snapshot_id,
            payload_json=msg.payload_json,
        )
        if decision.is_no_action:
            return
        self._handle_trigger_decision(decision)

    def _handle_trigger_decision(self, decision: TriggerDecision) -> None:
        if decision.action == ACTION_BLOCK_ACTIVE:
            self._block_active_from_trigger(decision)
            return
        if decision.action == ACTION_REPLAN_ACTIVE:
            self._replan_active_from_trigger(decision)

    def _select_missions(
        self,
        mission_id: str,
        *,
        include_terminal: bool,
    ) -> tuple[Mission, ...]:
        if mission_id.strip():
            resolved = self._missions.resolve_control_id(mission_id)
            mission = self._missions.get(resolved)
            if mission is None:
                raise KeyError(f"unknown mission_id: {mission_id}")
            missions = (mission,)
        else:
            missions = self._missions.all()
        if include_terminal:
            return missions
        terminal = {STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELED, STATE_BLOCKED}
        return tuple(mission for mission in missions if mission.state not in terminal)

    def _publish_event(self, event: MissionEvent) -> None:
        self._append_journal(event)
        status = self._status_msg(event.mission)
        msg = TaskEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.event = event.event
        msg.status = status
        msg.message = event.message
        msg.payload_json = event.payload_json
        self._event_pub.publish(msg)
        self._status_pub.publish(status)
        self._publish_task_projection()

    def _publish_status(self, mission: Mission) -> None:
        self._status_pub.publish(self._status_msg(mission))

    def _publish_startup_snapshot(self) -> None:
        if self._startup_snapshot_publishes_remaining <= 0:
            self._cancel_startup_snapshot_timer()
            return
        with self._mission_lock:
            missions = self._missions.all()
        if not missions:
            self._startup_snapshot_publishes_remaining = 0
            self._cancel_startup_snapshot_timer()
            return
        for mission in missions:
            self._publish_status(mission)
        self._publish_task_projection()
        self._startup_snapshot_publishes_remaining -= 1
        if self._startup_snapshot_publishes_remaining <= 0:
            self._cancel_startup_snapshot_timer()

    def _cancel_startup_snapshot_timer(self) -> None:
        timer = self._startup_snapshot_timer
        if timer is None:
            return
        self._startup_snapshot_timer = None
        try:
            timer.cancel()
            self.destroy_timer(timer)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(
                f"startup snapshot timer cleanup skipped: {type(exc).__name__}: {exc}"
            )

    def _on_execution_progress(
        self,
        mission: Mission,
        active_node: str,
        progress: float,
    ) -> None:
        with self._mission_lock:
            try:
                updated = self._missions.update_status(
                    mission.identity.mission_id,
                    active_node=active_node,
                    status_text="running",
                    progress=progress,
                    expected_identity=mission.identity,
                )
            except KeyError:
                return
        self._publish_status(updated)

    def _append_journal(self, event: MissionEvent) -> None:
        if self._mission_journal is None:
            return
        try:
            self._mission_journal.append(event)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"mission journal append failed: {type(exc).__name__}: {exc}"
            )

    def _publish_task_projection(self) -> None:
        with self._mission_lock:
            missions = self._missions.all()
        for update in task_projection_updates(missions):
            ok, message = self._world_writer.update(update, timeout_sec=0.25)
            if not ok:
                self.get_logger().debug(f"task projection skipped: {message}")

    def _block_active_from_trigger(self, decision: TriggerDecision) -> None:
        try:
            with self._mission_lock:
                target_id = self._missions.resolve_control_id("")
                runner_key = self._mission_cancel_keys.get(target_id, "")
                cancel_event = self._cancel_events.get(runner_key)
                event = self._missions.mark_terminal(
                    target_id,
                    state=STATE_BLOCKED,
                    message=decision.message,
                )
        except KeyError:
            self.get_logger().debug(
                f"trigger ignored with no active mission: {decision.reason}"
            )
            return
        if cancel_event is not None:
            cancel_event.set()
            self._skill_executor.cancel_current(decision.message)
        self._publish_event(event)
        self._start_next_ready()

    def _replan_active_from_trigger(self, decision: TriggerDecision) -> None:
        try:
            with self._mission_lock:
                target_id = self._missions.resolve_control_id("")
                current = self._missions.get(target_id)
                if current is None or not trigger_decision_matches_mission(
                    decision,
                    mission_id=current.identity.mission_id,
                    plan_version=current.identity.plan_version,
                ):
                    self.get_logger().debug(
                        f"stale replan trigger ignored: {decision.reason}"
                    )
                    return
                runner_key = self._mission_cancel_keys.get(target_id, "")
                cancel_event = self._cancel_events.get(runner_key)
                event = self._missions.request_replan(
                    target_id,
                    decision.message,
                    payload_json=json.dumps(
                        {
                            "schema": "mc_ai_bt.replan_trigger.v1",
                            "reason": decision.reason,
                            "details": decision.details,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
        except KeyError:
            self.get_logger().debug(
                f"replan trigger ignored with no active mission: {decision.reason}"
            )
            return
        if cancel_event is not None:
            cancel_event.set()
            self._skill_executor.cancel_current(decision.message)
        self._publish_event(event)
        self._plan_and_start(event.mission)

    def _status_msg(self, mission: Mission) -> TaskStatus:
        msg = TaskStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.identity = self._identity_msg(mission.identity)
        msg.state = mission.state
        msg.priority = mission.priority
        msg.title = mission.title
        msg.active_node = mission.active_node
        msg.status_text = mission.status_text
        msg.progress = mission.progress
        msg.bt_json = mission.bt_json
        msg.goal_spec_json = mission.goal_spec_json
        msg.error_code = mission.error_code
        return msg

    @staticmethod
    def _identity_msg(identity: Identity) -> AiBtIdentity:
        return identity_to_msg(identity)


def main() -> None:
    rclpy.init()
    node = AiBtNode()
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
