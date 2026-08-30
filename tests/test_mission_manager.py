from mc_ai_bt.identity import Identity
from mc_ai_bt.mission import (
    EVENT_PREEMPTED,
    MissionManager,
    STATE_CANCELED,
    STATE_PAUSED,
    STATE_QUEUED,
    STATE_PLANNING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)


def test_submit_sets_first_mission_active_planning():
    manager = MissionManager()

    accepted, message, mission, _event = manager.submit(
        intent_text="go to the test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )

    assert accepted
    assert message == "accepted"
    assert mission.state == STATE_PLANNING
    assert mission.identity.mission_id


def test_set_plan_increments_plan_version_and_runs():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="wave",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )

    event = manager.set_plan(mission.identity.mission_id, "{}", "{}")

    assert event.mission.state == STATE_RUNNING
    assert event.mission.identity.plan_version == 1
    assert event.mission.identity.execution_id


def test_replan_invalidates_old_execution_before_new_plan_runs():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(mission.identity.mission_id, "old bt", "old goal").mission

    event = manager.request_replan(
        running.identity.mission_id,
        "goal evidence changed",
        payload_json='{"event":"GOAL_EVIDENCE_CHANGED"}',
    )

    assert event.event == EVENT_PREEMPTED
    assert event.payload_json == '{"event":"GOAL_EVIDENCE_CHANGED"}'
    assert event.mission.state == STATE_PLANNING
    assert event.mission.identity.plan_version == running.identity.plan_version + 1
    assert event.mission.identity.execution_id == ""
    assert event.mission.bt_json == ""
    assert event.mission.goal_spec_json == ""
    assert not manager.accepts_async_result(running.identity)


def test_set_plan_after_replan_uses_reserved_plan_version():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(mission.identity.mission_id, "old bt", "old goal").mission
    replanning = manager.request_replan(running.identity.mission_id, "replan").mission

    planned = manager.set_plan(replanning.identity.mission_id, "new bt", "new goal").mission

    assert planned.state == STATE_RUNNING
    assert planned.identity.plan_version == replanning.identity.plan_version
    assert planned.identity.execution_id
    assert planned.identity.execution_id != running.identity.execution_id
    assert not manager.accepts_async_result(running.identity)
    assert manager.accepts_async_result(planned.identity)


def test_set_plan_rejects_stale_planning_identity_after_replan():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    original_identity = mission.identity
    manager.request_replan(mission.identity.mission_id, "new evidence")

    try:
        manager.set_plan(
            mission.identity.mission_id,
            "stale bt",
            "stale goal",
            expected_identity=original_identity,
        )
    except KeyError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("stale planner output should not be accepted")


def test_planning_failure_rejects_stale_identity_after_replan():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to the test_place",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    original_identity = mission.identity
    manager.request_replan(mission.identity.mission_id, "new evidence")

    try:
        manager.mark_terminal(
            mission.identity.mission_id,
            state=STATE_SUCCEEDED,
            message="stale success",
            expected_identity=original_identity,
        )
    except KeyError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("stale terminal update should not be accepted")


def test_late_result_from_old_plan_is_rejected():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="wave",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    planned = manager.set_plan(mission.identity.mission_id, "{}", "{}").mission

    assert manager.accepts_async_result(planned.identity)
    assert not manager.accepts_async_result(
        Identity(
            mission_id=planned.identity.mission_id,
            plan_version=0,
            execution_id="old",
        )
    )


def test_update_status_sets_active_node_and_progress():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="wave",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    planned = manager.set_plan(mission.identity.mission_id, "{}", "{}").mission

    updated = manager.update_status(
        planned.identity.mission_id,
        active_node="root.children[0]:Action:say",
        progress=0.25,
        expected_identity=planned.identity,
    )

    assert updated.active_node == "root.children[0]:Action:say"
    assert updated.progress == 0.25


def test_update_status_rejects_stale_identity():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="wave",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    planned = manager.set_plan(mission.identity.mission_id, "{}", "{}").mission
    manager.request_replan(planned.identity.mission_id, "new evidence")

    try:
        manager.update_status(
            planned.identity.mission_id,
            active_node="stale",
            expected_identity=planned.identity,
        )
    except KeyError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("stale status update should not be accepted")


def test_cancel_promotes_next_queued_mission():
    manager = MissionManager()
    _accepted, _message, first, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    manager.submit(
        intent_text="second",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=5,
        allow_queue=True,
        context_json="{}",
    )

    canceled = manager.cancel(first.identity.mission_id, "new command")
    promoted = [mission for mission in manager.all() if mission.state == STATE_PLANNING]

    assert canceled.mission.state == STATE_CANCELED
    assert len(promoted) == 1
    assert promoted[0].intent_text == "second"


def test_terminal_promotes_next_queued_mission():
    manager = MissionManager()
    _accepted, _message, first, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    manager.submit(
        intent_text="second",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=5,
        allow_queue=True,
        context_json="{}",
    )

    manager.mark_terminal(
        first.identity.mission_id,
        state=STATE_SUCCEEDED,
        message="done",
    )
    promoted = [mission for mission in manager.all() if mission.state == STATE_PLANNING]

    assert len(promoted) == 1
    assert promoted[0].intent_text == "second"


def test_reprioritize_updates_queued_priority():
    manager = MissionManager()
    manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    _accepted, _message, queued, _event = manager.submit(
        intent_text="second",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=2,
        allow_queue=True,
        context_json="{}",
    )

    event = manager.reprioritize(
        queued.identity.mission_id,
        priority=42,
        preempt_if_needed=False,
        reason="operator priority",
    )

    assert event.mission.priority == 42
    assert event.mission.status_text == "operator priority"


def test_reprioritize_can_preempt_active_with_higher_priority_queued_mission():
    manager = MissionManager()
    _accepted, _message, first, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(first.identity.mission_id, "bt", "goal").mission
    _accepted, _message, queued, _event = manager.submit(
        intent_text="second",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=2,
        allow_queue=True,
        context_json="{}",
    )

    event = manager.reprioritize(
        queued.identity.mission_id,
        priority=50,
        preempt_if_needed=True,
        reason="urgent",
    )

    assert event.event == EVENT_PREEMPTED
    assert event.mission.state == STATE_PLANNING
    assert event.mission.intent_text == "second"
    assert not manager.accepts_async_result(running.identity)
    demoted = manager.get(first.identity.mission_id)
    assert demoted is not None
    assert demoted.state == STATE_QUEUED


def test_empty_control_id_targets_current_active_mission():
    manager = MissionManager()
    _accepted, _message, first, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(first.identity.mission_id, "{}", "{}").mission

    paused = manager.pause("", "hold")
    resumed = manager.resume("current", "continue")
    canceled = manager.cancel("active", "stop")

    assert paused.mission.identity.mission_id == running.identity.mission_id
    assert paused.mission.state == STATE_PAUSED
    assert resumed.mission.state == STATE_RUNNING
    assert resumed.mission.identity.execution_id != paused.mission.identity.execution_id
    assert canceled.mission.state == STATE_CANCELED


def test_paused_mission_rejects_late_async_result():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(mission.identity.mission_id, "{}", "{}").mission

    manager.pause(running.identity.mission_id, "hold")

    assert not manager.accepts_async_result(running.identity)


def test_resume_generates_new_execution_id_for_same_plan():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    running = manager.set_plan(mission.identity.mission_id, "{}", "{}").mission
    manager.pause(running.identity.mission_id, "hold")

    resumed = manager.resume(running.identity.mission_id, "continue").mission

    assert resumed.state == STATE_RUNNING
    assert resumed.identity.plan_version == running.identity.plan_version
    assert resumed.identity.execution_id != running.identity.execution_id
    assert not manager.accepts_async_result(running.identity)
    assert manager.accepts_async_result(resumed.identity)


def test_resume_requires_paused_mission():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )

    try:
        manager.resume(mission.identity.mission_id, "continue")
    except KeyError as exc:
        assert "not paused" in str(exc)
    else:
        raise AssertionError("resume should reject non-paused mission")


def test_terminal_mission_cannot_be_resumed_or_canceled_again():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="first",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=1,
        allow_queue=True,
        context_json="{}",
    )
    manager.cancel(mission.identity.mission_id, "stop")

    for operation in (manager.resume, manager.cancel, manager.pause):
        try:
            operation(mission.identity.mission_id, "again")
        except KeyError as exc:
            assert "terminal" in str(exc) or "no active mission" in str(exc)
        else:
            raise AssertionError("terminal mission operation should fail")


def _submit(manager, intent_text, *, priority=1):
    _accepted, _message, mission, _event = manager.submit(
        intent_text=intent_text,
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=priority,
        allow_queue=True,
        context_json="{}",
    )
    return mission


def test_pause_promotes_next_queued_mission():
    # FOUND LIVE 2026-08-30: pause() used to leave `_active_id` pointed at the
    # paused mission indefinitely -- since only cancel/mark_terminal ever
    # promoted a queued mission, a mission paused on an unanswered escalation
    # (e.g. §10.7/C2's "awaiting Omega decision") blocked every later mission
    # at STATE_QUEUED forever, with no resource conflict involved at all.
    manager = MissionManager()
    first = _submit(manager, "first")
    manager.set_plan(first.identity.mission_id, "bt", "goal")
    second = _submit(manager, "second")
    assert manager.get(second.identity.mission_id).state == STATE_QUEUED

    manager.pause(first.identity.mission_id, "awaiting Omega decision: is a woman visible?")

    promoted = manager.get(second.identity.mission_id)
    assert promoted.state == STATE_PLANNING, (
        "a queued mission must start as soon as the active one pauses, not "
        "starve until someone cancels the paused one"
    )
    assert manager.get(first.identity.mission_id).state == STATE_PAUSED


def test_pause_with_nothing_queued_keeps_current_alias_on_the_paused_mission():
    # The other half of the same fix: when nothing is waiting, pause() must
    # NOT orphan `_active_id` to None (unlike a terminal transition, a paused
    # mission is still a legitimate ""/"active"/"current" resume target --
    # see test_empty_control_id_targets_current_active_mission).
    manager = MissionManager()
    first = _submit(manager, "first")
    manager.set_plan(first.identity.mission_id, "bt", "goal")

    manager.pause("current", "hold")

    resumed = manager.resume("current", "continue")
    assert resumed.mission.identity.mission_id == first.identity.mission_id
    assert resumed.mission.state == STATE_RUNNING


def test_resume_requeues_instead_of_stealing_an_occupied_slot():
    # The flip side of test_pause_promotes_next_queued_mission: once pause()
    # has handed the slot to a second mission, resuming the first one must
    # not unconditionally reclaim it back -- that would silently corrupt
    # `_active_id` bookkeeping (two missions both effectively "active") and
    # starve whatever the second mission itself later queues behind it.
    manager = MissionManager()
    first = _submit(manager, "first")
    manager.set_plan(first.identity.mission_id, "bt", "goal")
    second = _submit(manager, "second")
    manager.pause(first.identity.mission_id, "hold")
    assert manager.get(second.identity.mission_id).state == STATE_PLANNING

    resumed = manager.resume(first.identity.mission_id, "continue")

    assert resumed.mission.state == STATE_QUEUED, (
        "the slot belongs to 'second' now -- 'first' must wait its turn, "
        "not barge back into STATE_RUNNING/PLANNING"
    )
    assert manager.get(second.identity.mission_id).state == STATE_PLANNING, (
        "resuming 'first' must not disturb the mission that is actually active"
    )

    manager.cancel(second.identity.mission_id, "done")

    assert manager.get(first.identity.mission_id).state == STATE_PLANNING, (
        "once the slot frees up, the re-queued (resumed) mission takes its turn "
        "like any other queued mission"
    )
