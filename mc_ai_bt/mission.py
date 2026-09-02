from __future__ import annotations

from dataclasses import dataclass, replace

from .identity import Identity


STATE_UNKNOWN = 0
STATE_QUEUED = 1
STATE_PLANNING = 2
STATE_WAITING_FOR_PERMISSION = 3
STATE_RUNNING = 4
STATE_PAUSED = 5
STATE_SUCCEEDED = 6
STATE_FAILED = 7
STATE_CANCELED = 8
STATE_BLOCKED = 9
_TERMINAL_STATES = {STATE_CANCELED, STATE_FAILED, STATE_SUCCEEDED, STATE_BLOCKED}

EVENT_SUBMITTED = 1
EVENT_ACCEPTED = 2
EVENT_REJECTED = 3
EVENT_PLANNED = 4
EVENT_STARTED = 5
EVENT_PAUSED = 8
EVENT_RESUMED = 9
EVENT_PREEMPTED = 10
EVENT_CANCELED = 11
EVENT_SUCCEEDED = 12
EVENT_FAILED = 13
EVENT_BLOCKED = 14


@dataclass(frozen=True)
class Mission:
    identity: Identity
    intent_text: str
    source: str
    operator_id: str
    priority: int
    allow_queue: bool
    context_json: str
    state: int = STATE_QUEUED
    title: str = ""
    active_node: str = ""
    status_text: str = ""
    progress: float = 0.0
    bt_json: str = ""
    goal_spec_json: str = ""
    error_code: str = ""


@dataclass(frozen=True)
class MissionEvent:
    event: int
    mission: Mission
    message: str = ""
    payload_json: str = ""


class MissionManager:
    def __init__(self, missions: tuple[Mission, ...] = ()) -> None:
        self._missions: dict[str, Mission] = {
            mission.identity.mission_id: mission for mission in missions
        }
        self._active_id: str | None = self._select_active_id()
        # mission_id -> the pause reason it was resumed FROM (set by resume(), cleared
        # once the mission moves on to any state other than an identical re-pause).
        # See pause()'s own comment for why this exists.
        self._resumed_from_reason: dict[str, str] = {}

    def submit(
        self,
        *,
        intent_text: str,
        source: str,
        operator_id: str,
        parent_mission_id: str,
        priority: int,
        allow_queue: bool,
        context_json: str,
    ) -> tuple[bool, str, Mission, MissionEvent]:
        if not intent_text.strip():
            mission = self._synthetic_rejected(source, operator_id, parent_mission_id)
            return False, "intent_text is required", mission, MissionEvent(EVENT_REJECTED, mission)

        slot_busy = self._slot_busy()
        if slot_busy and not allow_queue:
            mission = self._synthetic_rejected(source, operator_id, parent_mission_id)
            return False, "another mission is active", mission, MissionEvent(EVENT_REJECTED, mission)

        identity = Identity.new(
            parent_mission_id=parent_mission_id,
            source=source,
            operator_id=operator_id,
        )
        state = STATE_QUEUED if slot_busy else STATE_PLANNING
        mission = Mission(
            identity=identity,
            intent_text=intent_text,
            source=source,
            operator_id=operator_id,
            priority=max(0, min(int(priority), 255)),
            allow_queue=allow_queue,
            context_json=context_json,
            state=state,
            title=intent_text.strip()[:80],
            status_text="queued" if state == STATE_QUEUED else "planning",
        )
        self._missions[mission.identity.mission_id] = mission
        if not slot_busy:
            self._active_id = mission.identity.mission_id
        return True, "accepted", mission, MissionEvent(EVENT_ACCEPTED, mission, "accepted")

    def _slot_busy(self) -> bool:
        # FOUND LIVE 2026-09-01 (D v1 correctness review): a PAUSED mission holds no
        # real resources (see pause()'s own docstring -- mc_resource_authority never
        # sees a lease held across a pause) and stays independently resumable
        # regardless of _active_id (resume()'s own slot_free check already re-queues
        # ITSELF if something else claims the slot meanwhile -- see resume()'s
        # comment). _release_slot_for_pause only promotes a mission that was ALREADY
        # queued at the moment of pausing; nothing re-checks the queue afterward, so
        # treating a paused mission as still "active" here let it starve every
        # mission submitted AFTER it paused (as opposed to the case already handled:
        # one queued before it paused) -- forever, since only resuming/canceling
        # THAT SAME mission was ever going to look at the queue again.
        if self._active_id is None:
            return False
        active = self._missions.get(self._active_id)
        return active is not None and active.state != STATE_PAUSED

    def set_plan(
        self,
        mission_id: str,
        bt_json: str,
        goal_spec_json: str,
        *,
        expected_identity: Identity | None = None,
    ) -> MissionEvent:
        mission = self._require(mission_id)
        self._check_can_update(mission, expected_identity)
        identity = mission.identity
        if mission.bt_json or identity.execution_id or identity.plan_version == 0:
            identity = identity.next_plan()
        planned = replace(
            mission,
            identity=identity.for_execution(),
            state=STATE_RUNNING,
            bt_json=bt_json,
            goal_spec_json=goal_spec_json,
            status_text="running",
        )
        self._missions[mission_id] = planned
        self._active_id = mission_id
        return MissionEvent(EVENT_PLANNED, planned, "planned")

    def request_replan(
        self,
        mission_id: str,
        reason: str,
        *,
        payload_json: str = "",
    ) -> MissionEvent:
        mission_id = self._resolve_control_id(mission_id)
        mission = self._require(mission_id)
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission_id}")
        replanning = replace(
            mission,
            identity=mission.identity.next_plan(),
            state=STATE_PLANNING,
            active_node="",
            status_text=reason or "replanning",
            bt_json="",
            goal_spec_json="",
        )
        self._missions[mission_id] = replanning
        self._active_id = mission_id
        return MissionEvent(
            EVENT_PREEMPTED,
            replanning,
            replanning.status_text,
            payload_json,
        )

    def pause(self, mission_id: str, reason: str) -> MissionEvent:
        # PAUSED is inert -- accepts_async_result() already refuses execution
        # results for a paused mission, and mc_resource_authority never sees a
        # lease held across a pause (each dedicated skill's lease is scoped to
        # one BT-node execution, released long before a Condition/GoalCheck
        # node can raise the escalation that leads here). So unlike cancel/
        # mark_terminal, this used to leave `_active_id` pointed at the paused
        # mission indefinitely -- the sole "active" slot this manager enforces
        # stayed occupied by a mission doing nothing, and every other queued
        # mission starved until a human explicitly cancelled it.
        #
        # FOUND LIVE 2026-08-30: a mission paused on an unanswered "awaiting
        # Omega decision" escalation (§10.7/C2) blocked a second, later
        # mission at STATE_QUEUED forever -- it never even reached planning.
        # Omega itself never noticed: it only tracks "is a robot_event still
        # owed to me", and a queued mission that never starts never emits one,
        # so Omega just replied conversationally to the user's next request
        # without submitting anything new. Confirmed via mc_resource_authority
        # returning zero active leases at the time -- this is the manager's
        # own single-active-mission bookkeeping, not a resource conflict.
        mission_id = self._resolve_control_id(mission_id)
        mission = self._require(mission_id)
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission_id}")
        if mission.state == STATE_PAUSED:
            return MissionEvent(EVENT_PAUSED, mission, mission.status_text)
        # FOUND LIVE 2026-08-30: resume() re-runs the exact BT node that raised the
        # escalation, with no way to carry new evidence into it (e.g. Omega separately
        # confirming something by camera doesn't reach the Condition it resumed) -- so
        # a resume based on anything other than the world-state fact the check itself
        # reads just re-produces the identical "awaiting Omega decision" reason,
        # immediately. Observed live: Omega resumed such a mission, got the identical
        # pause text back within the same second, and then never noticed -- its own
        # summary kept saying "resumed, awaiting outcome" for 6+ minutes with no
        # further action, while the mission sat paused the whole time. Silently
        # re-pausing on the identical reason right after a resume is indistinguishable
        # from progress to whoever is watching for a robot_event, so instead of pausing
        # again, this now escalates straight to BLOCKED (terminal -- releases the
        # slot exactly like any other terminal transition, and emits a real event
        # through the same channel §16.1 fixed) with a message that says plainly that
        # resuming will not help, rather than leaving the mission (and whoever is
        # waiting on it) in the same silent limbo a second time.
        if self._resumed_from_reason.get(mission_id) == reason:
            self._resumed_from_reason.pop(mission_id, None)
            return self.mark_terminal(
                mission_id,
                state=STATE_BLOCKED,
                message=(
                    f"resumed, but immediately re-paused on the identical reason: {reason!r} -- "
                    "resuming again will not resolve this; a different approach is needed"
                ),
            )
        paused = replace(mission, state=STATE_PAUSED, status_text=reason or "paused")
        self._missions[mission_id] = paused
        if self._active_id == mission_id:
            self._release_slot_for_pause(mission_id)
        return MissionEvent(EVENT_PAUSED, paused, paused.status_text)

    def resume(self, mission_id: str, reason: str) -> MissionEvent:
        mission_id = self._resolve_control_id(mission_id)
        mission = self._require(mission_id)
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission_id}")
        if mission.state != STATE_PAUSED:
            raise KeyError(f"mission is not paused: {mission_id}")
        # Mirror pause()'s own slot handoff: pause() promotes a queued mission
        # into `_active_id` when one is waiting, so resuming must not
        # unconditionally reclaim the slot -- something else may genuinely be
        # active now. Claim it only if it is free or still this mission's own
        # (the common case: nothing was queued when this mission paused, so
        # `_active_id` never moved); otherwise re-enter the normal queue, the
        # same way any other mission waits its turn.
        slot_free = self._active_id is None or self._active_id == mission_id
        next_state = (
            (STATE_RUNNING if mission.bt_json else STATE_PLANNING)
            if slot_free else STATE_QUEUED
        )
        resumed = replace(
            mission,
            identity=mission.identity.for_execution() if (mission.bt_json and slot_free) else mission.identity,
            state=next_state,
            status_text=reason or {
                STATE_RUNNING: "running",
                STATE_PLANNING: "planning",
                STATE_QUEUED: "queued",
            }[next_state],
        )
        self._missions[mission_id] = resumed
        if slot_free:
            self._active_id = mission_id
        # Only relevant when resuming straight back into the SAME BT (STATE_RUNNING):
        # that is the one case where the exact node that raised this pause is about to
        # re-execute verbatim, with no way for whatever resolved the caller's mind
        # (resume's own `reason` here, or evidence gathered outside this mission
        # entirely) to reach that node. A resume that goes through STATE_PLANNING
        # instead (slot was busy, or there was no bt_json yet) gets a fresh plan, which
        # may check something else entirely -- tracking the old reason there would
        # compare apples to oranges, so only record it for the STATE_RUNNING case.
        if next_state == STATE_RUNNING:
            self._resumed_from_reason[mission_id] = mission.status_text
        else:
            self._resumed_from_reason.pop(mission_id, None)
        return MissionEvent(EVENT_RESUMED, resumed, resumed.status_text)

    def reprioritize(
        self,
        mission_id: str,
        *,
        priority: int,
        preempt_if_needed: bool,
        reason: str,
    ) -> MissionEvent:
        mission_id = self._resolve_control_id(mission_id)
        mission = self._require(mission_id)
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission_id}")
        new_priority = max(0, min(int(priority), 255))
        updated = replace(
            mission,
            priority=new_priority,
            status_text=reason or mission.status_text,
        )
        self._missions[mission_id] = updated

        active = self._missions.get(self._active_id or "")
        if (
            preempt_if_needed
            and updated.state == STATE_QUEUED
            and active is not None
            and active.identity.mission_id != mission_id
            and active.state not in _TERMINAL_STATES
            and updated.priority > active.priority
        ):
            demoted = replace(
                active,
                identity=active.identity.next_plan(),
                state=STATE_QUEUED,
                active_node="",
                status_text=f"preempted by higher priority mission {mission_id}",
                bt_json="",
                goal_spec_json="",
            )
            promoted = replace(
                updated,
                state=STATE_PLANNING,
                active_node="",
                status_text=reason or "planning",
                bt_json="",
                goal_spec_json="",
            )
            self._missions[demoted.identity.mission_id] = demoted
            self._missions[mission_id] = promoted
            self._active_id = mission_id
            return MissionEvent(
                EVENT_PREEMPTED,
                promoted,
                promoted.status_text,
                payload_json=f'{{"preempted_mission_id":"{demoted.identity.mission_id}"}}',
            )

        return MissionEvent(EVENT_ACCEPTED, updated, reason or "priority updated")

    def cancel(self, mission_id: str, reason: str) -> MissionEvent:
        mission_id = self._resolve_control_id(mission_id)
        mission = self._require(mission_id)
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission_id}")
        canceled = replace(
            mission,
            state=STATE_CANCELED,
            status_text=reason or "canceled",
            progress=mission.progress,
        )
        self._missions[mission_id] = canceled
        if self._active_id == mission_id:
            self._promote_next_queued()
        self._resumed_from_reason.pop(mission_id, None)
        return MissionEvent(EVENT_CANCELED, canceled, canceled.status_text)

    def mark_terminal(
        self,
        mission_id: str,
        *,
        state: int,
        message: str,
        expected_identity: Identity | None = None,
    ) -> MissionEvent:
        if state not in {STATE_SUCCEEDED, STATE_FAILED, STATE_BLOCKED}:
            raise ValueError("terminal state must be succeeded, failed or blocked")
        mission = self._require(mission_id)
        self._check_can_update(mission, expected_identity)
        done = replace(
            mission,
            state=state,
            status_text=message,
            progress=1.0 if state == STATE_SUCCEEDED else mission.progress,
        )
        self._missions[mission_id] = done
        if self._active_id == mission_id:
            self._promote_next_queued()
        self._resumed_from_reason.pop(mission_id, None)
        event = {
            STATE_SUCCEEDED: EVENT_SUCCEEDED,
            STATE_FAILED: EVENT_FAILED,
            STATE_BLOCKED: EVENT_BLOCKED,
        }[state]
        return MissionEvent(event, done, message)

    def accepts_async_result(self, identity: Identity) -> bool:
        mission = self._missions.get(identity.mission_id)
        if mission is None:
            return False
        if mission.state == STATE_PAUSED or mission.state in _TERMINAL_STATES:
            return False
        if mission.identity.plan_version != identity.plan_version:
            return False
        if identity.execution_id and mission.identity.execution_id:
            return mission.identity.execution_id == identity.execution_id
        return True

    def update_status(
        self,
        mission_id: str,
        *,
        active_node: str = "",
        status_text: str = "",
        progress: float | None = None,
        expected_identity: Identity | None = None,
    ) -> Mission:
        mission = self._require(mission_id)
        self._check_can_update(mission, expected_identity)
        updated = replace(
            mission,
            active_node=active_node if active_node != "" else mission.active_node,
            status_text=status_text if status_text != "" else mission.status_text,
            progress=_clamp_progress(progress) if progress is not None else mission.progress,
        )
        self._missions[mission_id] = updated
        return updated

    def get(self, mission_id: str) -> Mission | None:
        return self._missions.get(mission_id)

    def all(self) -> tuple[Mission, ...]:
        return tuple(self._missions.values())

    def resolve_control_id(self, mission_id: str) -> str:
        return self._resolve_control_id(mission_id)

    def _require(self, mission_id: str) -> Mission:
        mission = self._missions.get(mission_id)
        if mission is None:
            raise KeyError(f"unknown mission_id: {mission_id}")
        return mission

    def _check_can_update(
        self,
        mission: Mission,
        expected_identity: Identity | None,
    ) -> None:
        if mission.state in _TERMINAL_STATES:
            raise KeyError(f"mission is terminal: {mission.identity.mission_id}")
        if expected_identity is None:
            return
        if mission.identity != expected_identity:
            raise KeyError(f"mission identity changed: {mission.identity.mission_id}")

    def _resolve_control_id(self, mission_id: str) -> str:
        key = (mission_id or "").strip().lower()
        if key not in {"", "active", "current"}:
            return mission_id
        if self._active_id is None:
            raise KeyError("no active mission")
        return self._active_id

    def _next_queued_id(self) -> str | None:
        queued = [
            mission
            for mission in self._missions.values()
            if mission.state == STATE_QUEUED
        ]
        if not queued:
            return None
        queued.sort(key=lambda mission: (-mission.priority, mission.identity.mission_id))
        return queued[0].identity.mission_id

    def _select_active_id(self) -> str | None:
        active = [
            mission
            for mission in self._missions.values()
            if mission.state not in _TERMINAL_STATES
        ]
        if not active:
            return None
        active.sort(key=lambda mission: (-mission.priority, mission.identity.mission_id))
        return active[0].identity.mission_id

    def _promote(self, next_id: str) -> None:
        self._active_id = next_id
        queued = self._missions[next_id]
        self._missions[next_id] = replace(
            queued,
            state=STATE_PLANNING,
            status_text="planning",
        )

    def _promote_next_queued(self) -> None:
        # Used after a mission goes TERMINAL: it can never be a valid
        # "" / "active" / "current" alias target again (_check_can_update
        # already rejects terminal missions), so orphaning `_active_id` to
        # None when nothing is queued is correct here -- there genuinely is
        # no mission left to call "current".
        next_id = self._next_queued_id()
        if next_id is None:
            self._active_id = None
            return
        self._promote(next_id)

    def _release_slot_for_pause(self, mission_id: str) -> None:
        # Same handoff as _promote_next_queued, but for a mission going
        # PAUSED, not terminal: unlike a terminal mission, a paused one
        # remains a perfectly valid resume()/cancel() target, including via
        # the ""/"active"/"current" alias (see
        # test_empty_control_id_targets_current_active_mission) -- so when
        # there is nothing queued to hand the slot to, `_active_id` is left
        # pointing at `mission_id` rather than cleared to None.
        next_id = self._next_queued_id()
        if next_id is None:
            return
        self._promote(next_id)

    @staticmethod
    def _synthetic_rejected(
        source: str,
        operator_id: str,
        parent_mission_id: str,
    ) -> Mission:
        identity = Identity.new(
            parent_mission_id=parent_mission_id,
            source=source,
            operator_id=operator_id,
        )
        return Mission(
            identity=identity,
            intent_text="",
            source=source,
            operator_id=operator_id,
            priority=0,
            allow_queue=False,
            context_json="",
            state=STATE_FAILED,
            status_text="rejected",
        )


def _clamp_progress(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
