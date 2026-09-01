from __future__ import annotations

import threading
import json

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from mc_one.msg import AiBtIdentity, PolicyState, TaskEvent, TaskStatus, WorldEvent
from mc_one.srv import (
    CancelMission,
    ListMissions,
    PauseMission,
    QueryWorld,
    ReprioritizeMission,
    ResumeMission,
    SetPolicyState,
    SubmitTaskIntent,
)

from .context_builder import ContextBuilder
from .executor import BtExecutor
from .goal_check import GoalChecker, TriState
from .identity import Identity
from .mission import (
    EVENT_PREEMPTED,
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
from .planning_pipeline import PlanningPipeline, PlanningResult
from .skill_adapters import RosSkillExecutor
from .skill_registry import SkillRegistry
from .task_projection import task_projection_updates
from .query_world import QueryWorldEngine
from .ros_identity import identity_to_msg
from .resource_client import ResourceLeaseClient
from .trigger_manager import (
    ACTION_BLOCKED,
    ACTION_ASK_CLARIFICATION,
    ACTION_LOCAL_HANDLED,
    ACTION_PLAN_NEW_MISSION,
    ACTION_REPLAN,
    TriggerDecision,
    TriggerManager,
    trigger_decision_matches_mission,
)
from .validator import PlanValidator
from .visual_client import MissionCheckExecutor, VisualCheckClient, VisualCheckSettings
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
        self._visual_client = VisualCheckClient(
            self,
            settings=VisualCheckSettings(
                action_name=str(self.get_parameter("visual_check_action").value or "/mc_multimodal/visual_check"),
                camera_source=str(self.get_parameter("visual_check_camera_source").value or "head"),
                max_age_sec=float(self.get_parameter("visual_check_max_age_sec").value or 2.0),
                timeout_sec=float(self.get_parameter("visual_check_timeout_sec").value or 15.0),
            ),
            callback_group=self._client_callback_group,
        )
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
        self._policy_state = self._default_policy_state()
        self._mission_lock = threading.RLock()
        self._runner_threads: list[threading.Thread] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._mission_cancel_keys: dict[str, str] = {}
        self._status_pub = self.create_publisher(TaskStatus, "/mc_ai_bt/task_status", 10)
        self._event_pub = self.create_publisher(TaskEvent, "/mc_ai_bt/task_events", 10)
        self._policy_pub = self.create_publisher(
            PolicyState,
            "/mc_ai_bt/policy_state",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_service(SubmitTaskIntent, "/mc_ai_bt/submit_task", self._handle_submit)
        self.create_service(CancelMission, "/mc_ai_bt/cancel_mission", self._handle_cancel)
        self.create_service(PauseMission, "/mc_ai_bt/pause_mission", self._handle_pause)
        self.create_service(ResumeMission, "/mc_ai_bt/resume_mission", self._handle_resume)
        self.create_service(ReprioritizeMission, "/mc_ai_bt/reprioritize_mission", self._handle_reprioritize)
        self.create_service(ListMissions, "/mc_ai_bt/list_missions", self._handle_list)
        self.create_service(QueryWorld, "/mc_ai_bt/query_world", self._handle_query_world)
        self.create_service(SetPolicyState, "/mc_ai_bt/set_policy_state", self._handle_set_policy_state)
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
        self._publish_policy_state()

    def _declare_parameters(self) -> None:
        self.declare_parameter("planner_backend", "bootstrap")
        self.declare_parameter("planner_model_name", "")
        self.declare_parameter("planner_temperature", 0.0)
        self.declare_parameter("planner_timeout", 15.0)
        self.declare_parameter("mission_journal_path", "")
        self.declare_parameter("visual_check_action", "/mc_multimodal/visual_check")
        self.declare_parameter("visual_check_camera_source", "head")
        self.declare_parameter("visual_check_max_age_sec", 2.0)
        self.declare_parameter("visual_check_timeout_sec", 15.0)

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

        self._plan_and_start_async(mission)
        return response

    def _plan_and_start_async(self, mission: Mission) -> None:
        """Run _plan_and_start on its own thread so the caller returns
        immediately instead of blocking for the full LLM planning latency.

        FOUND LIVE 2026-08-31: SubmitTaskIntent.srv's own comment documents
        the mission as running asynchronously, but _handle_submit called
        _plan_and_start directly -- the service callback itself blocked until
        planning finished, failed, or _plan_with_hard_timeout's backstop
        fired. _plan_with_hard_timeout already runs the LLM call on its own
        thread, but that only bounds ONE call's latency; it does not stop the
        calling thread from blocking on it. This is the actual async
        boundary: the same expected_identity staleness checks
        _plan_and_start already uses internally (mark_terminal/set_plan's
        KeyError-on-stale-identity path) are what make it safe to let more
        than one of these run/land at different times.

        Every call site that used to call _plan_and_start directly now goes
        through here, not just the three external service handlers (submit/
        resume/reprioritize) this was originally written for: _start_next_ready
        runs on a mission's own runner thread (a planning hang there would
        have delayed the NEXT mission picking up its slot, the same class of
        problem this exists to prevent, just one thread removed from a live
        caller instead of on it), and _replan_active_from_trigger/
        _plan_new_mission_from_trigger run on whatever callback processes
        WorldEvent-driven triggers, where a synchronous block risks starving
        an entire ROS callback group, not just one caller's response.
        """
        threading.Thread(
            target=self._plan_and_start,
            args=(mission,),
            name=f"plan-and-start-{mission.identity.mission_id[:8]}",
            daemon=True,
        ).start()

    def _plan_and_start(self, mission: Mission) -> None:
        expected_identity = mission.identity
        with self._mission_lock:
            missions = self._missions.all()
        planning = self._plan_with_hard_timeout(mission, missions)
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

    def _plan_with_hard_timeout(
        self, mission: Mission, missions: tuple[Mission, ...]
    ) -> PlanningResult:
        """Enforce the configured planner_timeout as a hard wall-clock deadline
        around self._planning.plan(), independent of whatever timeout the
        injected LLM client itself claims to honor.

        FOUND LIVE 2026-08-31: the langchain_openai ChatOpenAI client backing
        the planner is already constructed with timeout=planner_timeout
        (planner_factory.py), but under this session's observed network
        conditions that configured timeout did not fire at all: a mission sat
        in "planning" for 4+ minutes with no exception ever raised anywhere,
        holding the single active-mission slot the whole time (submit_task_intent
        calls self._plan_and_start synchronously, so the caller's own request
        -- the Robot Gateway Bridge's /submit_mission call -- was left hanging
        too, past its own separate timeout, and reported a confusing failure
        while the mission kept sitting there regardless).
        PlanningPipeline.plan already turns a raised exception into a graceful
        PlanningResult(False, "planner", ...) -- see its own try/except -- this
        gives that same graceful outcome an upper bound even when the
        underlying call never raises anything at all. Running it on its own
        thread and giving up waiting after the deadline, regardless of whether
        that thread ever returns, is a hard backstop, not a replacement for a
        working client-side timeout -- the orphaned thread's eventual result,
        if any, is simply discarded.
        """
        timeout_sec = float(self.get_parameter("planner_timeout").value or 15.0)
        result_box: dict[str, object] = {}
        done = threading.Event()

        def _run() -> None:
            try:
                result_box["value"] = self._planning.plan(mission, missions)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=_run,
            name=f"planning-{mission.identity.mission_id[:8]}",
            daemon=True,
        ).start()
        # A little slack past timeout_sec: the client's own timeout is meant to
        # fire first and produce a real error message; this deadline is only
        # the backstop for when it does not.
        if not done.wait(timeout=max(0.0, timeout_sec) + 1.0):
            return PlanningResult(
                False,
                "planner",
                f"planner timed out after {timeout_sec:g}s with no response from the LLM backend",
            )
        if "error" in result_box:
            exc = result_box["error"]
            return PlanningResult(False, "planner", f"planner failed: {type(exc).__name__}: {exc}")
        return result_box["value"]  # type: ignore[return-value]

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
                checks = MissionCheckExecutor(
                    goal_checker=self._goal_checker,
                    visual_client=self._visual_client,
                    identity=mission.identity,
                    cancel_event=cancel_event,
                )
                execution = self._executor.execute_json(
                    mission.bt_json,
                    self._skill_executor,
                    cancel_event=cancel_event,
                    checks=checks,
                    progress_callback=lambda active_node, progress: self._on_execution_progress(
                        mission,
                        active_node,
                        progress,
                    ),
                )
            state = STATE_FAILED
            message = execution.message
            # `escalate`, not STATE_BLOCKED: a genuine evidentiary gap (a
            # Condition/GoalCheck/VisualCheck node, or the final goal check
            # itself, came back UNKNOWN) is resumable once new evidence
            # arrives -- pause() rather than a terminal mark, per
            # OMEGACLAW_AI_BT_INTEGRATION.md §3.5. A mechanical dead-end
            # (execution.blocked but NOT execution.needs_decision -- a
            # Timeout expired, a Parallel child never reported, a checker
            # isn't wired up) still terminates as BLOCKED: replanning won't
            # fix a config error or an expired timeout the way it can
            # resolve missing evidence, and there is no decision for Omega
            # to actually make there.
            escalate = False
            if execution.blocked:
                if execution.needs_decision:
                    escalate = True
                    message = f"awaiting Omega decision: {execution.message}"
                else:
                    state = STATE_BLOCKED
                    message = execution.message
            elif execution.success:
                check = checks.check_json(mission.goal_spec_json, execution)
                if check.state is TriState.TRUE:
                    state = STATE_SUCCEEDED
                    message = f"goal check TRUE: {check.message}"
                elif check.state is TriState.UNKNOWN:
                    escalate = True
                    message = f"awaiting Omega decision: goal check UNKNOWN: {check.message}"
                else:
                    state = STATE_FAILED
                    message = f"goal check FALSE: {check.message}"

            with self._mission_lock:
                if not self._missions.accepts_async_result(mission.identity):
                    return
                if escalate:
                    outcome = self._missions.pause(mission.identity.mission_id, message)
                else:
                    outcome = self._missions.mark_terminal(
                        mission.identity.mission_id,
                        state=state,
                        message=message,
                    )
            self._publish_event(outcome)
            if not escalate:
                # A paused mission keeps holding _active_id (pause() does not
                # promote the next queued mission, unlike mark_terminal) --
                # nothing else should start in its place while it is only
                # waiting on a decision, not actually done.
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
            self._plan_and_start_async(ready[0])

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
            self._plan_and_start_async(event.mission)
        response.success = True
        response.message = event.message
        return response

    def _handle_reprioritize(
        self,
        request: ReprioritizeMission.Request,
        response: ReprioritizeMission.Response,
    ) -> ReprioritizeMission.Response:
        preempted_cancel_event = None
        preempt_reason = request.reason or "reprioritized"
        try:
            with self._mission_lock:
                event = self._missions.reprioritize(
                    request.mission_id,
                    priority=request.priority,
                    preempt_if_needed=bool(request.preempt_if_needed),
                    reason=preempt_reason,
                )
                preempted_id = ""
                if event.event == EVENT_PREEMPTED and event.payload_json:
                    try:
                        payload = json.loads(event.payload_json)
                        preempted_id = str(payload.get("preempted_mission_id") or "")
                    except (TypeError, ValueError):
                        preempted_id = ""
                if preempted_id:
                    runner_key = self._mission_cancel_keys.get(preempted_id, "")
                    preempted_cancel_event = self._cancel_events.get(runner_key)
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            return response
        if preempted_cancel_event is not None:
            preempted_cancel_event.set()
            self._skill_executor.cancel_current(preempt_reason)
        self._publish_event(event)
        if event.mission.state == STATE_PLANNING and not event.mission.bt_json:
            self._plan_and_start_async(event.mission)
        response.success = True
        response.message = event.message
        response.status = self._status_msg(event.mission)
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

    def _handle_set_policy_state(
        self,
        request: SetPolicyState.Request,
        response: SetPolicyState.Response,
    ) -> SetPolicyState.Response:
        try:
            policy = self._policy_from_request(request)
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            response.current_policy = self._policy_state
            return response
        self._policy_state = policy
        self._publish_policy_state()
        self._world_writer.update_fact(
            source=request.source or "mc_ai_bt.policy",
            scope="policy",
            key="state",
            value=_policy_state_dict(policy),
            timeout_sec=0.25,
        )
        response.success = True
        response.message = "policy updated"
        response.current_policy = policy
        return response

    def _on_world_event(self, msg: WorldEvent) -> None:
        has_active_mission = self._has_active_mission()
        decision = self._trigger_manager.handle_world_event(
            event_type=msg.event_type,
            snapshot_id=msg.snapshot_id,
            payload_json=msg.payload_json,
            has_active_mission=has_active_mission,
        )
        if decision.is_no_action:
            return
        self._handle_trigger_decision(decision)

    def _handle_trigger_decision(self, decision: TriggerDecision) -> None:
        if decision.action == ACTION_LOCAL_HANDLED:
            self.get_logger().debug(f"trigger handled locally: {decision.reason}")
            return
        if decision.action == ACTION_BLOCKED:
            self._block_active_from_trigger(decision)
            return
        if decision.action == ACTION_REPLAN:
            self._replan_active_from_trigger(decision)
            return
        if decision.action == ACTION_ASK_CLARIFICATION:
            self._block_active_from_trigger(decision)
            return
        if decision.action == ACTION_PLAN_NEW_MISSION:
            self._plan_new_mission_from_trigger(decision)
            return
        self.get_logger().debug(
            f"unsupported trigger decision ignored: {decision.action}"
        )

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

    def _publish_policy_state(self) -> None:
        policy = self._copy_policy_state(self._policy_state)
        policy.header.stamp = self.get_clock().now().to_msg()
        self._policy_pub.publish(policy)

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

    def _has_active_mission(self) -> bool:
        terminal = {STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELED, STATE_BLOCKED}
        with self._mission_lock:
            try:
                target_id = self._missions.resolve_control_id("")
            except KeyError:
                return False
            mission = self._missions.get(target_id)
        return mission is not None and mission.state not in terminal

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
        self._plan_and_start_async(event.mission)

    def _plan_new_mission_from_trigger(self, decision: TriggerDecision) -> None:
        context_json = json.dumps(
            {
                "schema": "mc_ai_bt.trigger_context.v1",
                "decision": {
                    "action": decision.action,
                    "reason": decision.reason,
                    "message": decision.message,
                    "details": decision.details,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        intent_text = _intent_for_trigger_new_mission(decision)
        try:
            with self._mission_lock:
                if self._has_active_mission_unlocked():
                    self.get_logger().debug(
                        f"new mission trigger ignored because a mission is active: {decision.reason}"
                    )
                    return
                accepted, _message, mission, event = self._missions.submit(
                    intent_text=intent_text,
                    source="world_event",
                    operator_id="trigger_manager",
                    parent_mission_id="",
                    priority=20,
                    allow_queue=False,
                    context_json=context_json,
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"new mission trigger failed: {type(exc).__name__}: {exc}"
            )
            return
        self._publish_event(event)
        if accepted and mission.state != STATE_QUEUED:
            self._plan_and_start_async(mission)

    def _has_active_mission_unlocked(self) -> bool:
        terminal = {STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELED, STATE_BLOCKED}
        try:
            target_id = self._missions.resolve_control_id("")
        except KeyError:
            return False
        mission = self._missions.get(target_id)
        return mission is not None and mission.state not in terminal

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

    @staticmethod
    def _default_policy_state() -> PolicyState:
        policy = PolicyState()
        policy.autonomy_level = PolicyState.AUTONOMY_INTERACTIVE
        policy.privacy_mode = False
        policy.allow_active_vision = True
        policy.allow_following = True
        policy.allow_guiding = True
        policy.allow_approach_unknown_person = False
        policy.hard_safety_stop_active = False
        policy.consent_owner_id = ""
        policy.policy_json = "{}"
        return policy

    @staticmethod
    def _copy_policy_state(src: PolicyState) -> PolicyState:
        policy = PolicyState()
        policy.autonomy_level = int(src.autonomy_level)
        policy.privacy_mode = bool(src.privacy_mode)
        policy.allow_active_vision = bool(src.allow_active_vision)
        policy.allow_following = bool(src.allow_following)
        policy.allow_guiding = bool(src.allow_guiding)
        policy.allow_approach_unknown_person = bool(src.allow_approach_unknown_person)
        policy.hard_safety_stop_active = bool(src.hard_safety_stop_active)
        policy.consent_owner_id = str(src.consent_owner_id or "")
        policy.policy_json = str(src.policy_json or "{}")
        return policy

    def _policy_from_request(self, request: SetPolicyState.Request) -> PolicyState:
        policy = self._copy_policy_state(self._policy_state if request.merge else request.policy)
        patch_text = str(request.patch_json or "").strip()
        if not patch_text:
            return policy
        try:
            patch = json.loads(patch_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid policy patch_json: {exc}") from exc
        if not isinstance(patch, dict):
            raise ValueError("policy patch_json must be an object")
        _apply_policy_patch(policy, patch)
        return policy


def _log_provenance(entry_point: str) -> None:
    """Print which copy of this node's code is actually running, and whether
    it matches the real ros2-run entry point — before rclpy.init(), with a
    plain print (not the ROS logger) so it survives regardless of logging
    config. See [[feedback_dual_entrypoint_hotpatch]]: a site-packages-only
    hot patch can silently leave the real entry point stale for an unknown
    period with no error at all. This makes "which code is actually running"
    checkable in the container logs instead of assumed.
    """
    import hashlib
    import os

    def _hash(path: str) -> tuple[str, str]:
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()[:12]
            return digest, f"{os.path.getmtime(path):.0f}"
        except OSError as exc:
            return f"MISSING({exc})", "-"

    self_path = os.path.abspath(__file__)
    self_sha, self_mtime = _hash(self_path)
    entry_sha, entry_mtime = _hash(entry_point)
    print(
        f"PROVENANCE self={self_path} sha256={self_sha} mtime={self_mtime} "
        f"| entry={entry_point} sha256={entry_sha} mtime={entry_mtime}",
        flush=True,
    )


def main() -> None:
    _log_provenance("/ros2_ws/install/mc_ai_bt/lib/mc_ai_bt/ai_bt")
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


def _intent_for_trigger_new_mission(decision: TriggerDecision) -> str:
    if decision.reason == "PERSON_HAND_OFFERED":
        return "A person is offering an interaction. Decide whether to start a safe local interaction or take no action."
    if decision.reason == "PERSON_APPROACHED":
        return "A person approached the robot. Decide whether to greet or take no action."
    return f"Handle world event {decision.reason} if useful, otherwise take no action."


def _policy_state_dict(policy: PolicyState) -> dict[str, object]:
    return {
        "schema": "mc_ai_bt.policy_state.v1",
        "autonomy_level": int(policy.autonomy_level),
        "privacy_mode": bool(policy.privacy_mode),
        "allow_active_vision": bool(policy.allow_active_vision),
        "allow_following": bool(policy.allow_following),
        "allow_guiding": bool(policy.allow_guiding),
        "allow_approach_unknown_person": bool(policy.allow_approach_unknown_person),
        "hard_safety_stop_active": bool(policy.hard_safety_stop_active),
        "consent_owner_id": str(policy.consent_owner_id or ""),
        "policy_json": str(policy.policy_json or "{}"),
    }


def _apply_policy_patch(policy: PolicyState, patch: dict[str, object]) -> None:
    allowed = {
        "autonomy_level",
        "privacy_mode",
        "allow_active_vision",
        "allow_following",
        "allow_guiding",
        "allow_approach_unknown_person",
        "hard_safety_stop_active",
        "consent_owner_id",
        "policy_json",
    }
    for key, value in patch.items():
        if key not in allowed:
            raise ValueError(f"unsupported policy field: {key}")
        if key == "autonomy_level":
            level = int(value)
            if level < PolicyState.AUTONOMY_LOCKED_DOWN or level > PolicyState.AUTONOMY_SUPERVISED:
                raise ValueError(f"invalid autonomy_level: {level}")
            policy.autonomy_level = level
        elif key == "consent_owner_id":
            policy.consent_owner_id = str(value or "")
        elif key == "policy_json":
            text = str(value or "{}")
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid policy_json: {exc}") from exc
            policy.policy_json = text
        else:
            setattr(policy, key, bool(value))


if __name__ == "__main__":
    main()
