import json

from mc_ai_bt.identity import Identity
from mc_ai_bt.mission import (
    EVENT_BLOCKED,
    EVENT_CANCELED,
    EVENT_FAILED,
    EVENT_PAUSED,
    EVENT_PREEMPTED,
    EVENT_SUCCEEDED,
    MISSION_OUTCOME_SCHEMA,
    Mission,
    MissionManager,
    STATE_BLOCKED,
    STATE_CANCELED,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_QUEUED,
    STATE_PLANNING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
    _build_mission_outcome_envelope,
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


def test_mission_submitted_after_another_already_paused_still_starts():
    # FOUND LIVE 2026-09-01 (D v1 correctness review): _release_slot_for_pause only
    # promotes a mission that was ALREADY queued at the moment pause() ran -- nothing
    # re-checks the queue afterward. Before this fix, a mission paused with nothing
    # queued left _active_id pointed at it forever (deliberately, so it stays a valid
    # ""/"active"/"current" resume target -- see the test above), which meant a
    # mission submitted LATER got stuck QUEUED with no promotion trigger ever coming:
    # only resuming/canceling the SAME paused mission would have looked at the queue
    # again. A paused mission holds no real resources (pause()'s own docstring), so
    # it must not block admission of new work the way a genuinely running one does.
    manager = MissionManager()
    first = _submit(manager, "first")
    manager.set_plan(first.identity.mission_id, "bt", "goal")
    manager.pause(first.identity.mission_id, "awaiting Omega decision")
    assert manager.all()[0].state == STATE_PAUSED

    second = _submit(manager, "second")

    assert second.state == STATE_PLANNING  # not STATE_QUEUED -- this is the whole bug
    accepted, _message, _mission, event = manager.submit(
        intent_text="third", source="voice", operator_id="user",
        parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
    )
    assert accepted
    # The paused mission is still independently resumable, just no longer "active":
    # resume() re-queues itself since the slot now genuinely belongs to `second`.
    resumed = manager.resume(first.identity.mission_id, "late evidence")
    assert resumed.mission.state == STATE_QUEUED


def test_resume_then_identical_repause_escalates_to_blocked_not_silent_limbo():
    # Found live 2026-08-30: resume() re-runs the exact BT node that raised the
    # escalation, with no way to carry new evidence into it. Omega separately
    # confirmed (by camera) the thing the paused Condition needed, resumed the
    # mission on that basis, and the mission re-paused on the IDENTICAL reason
    # within the same second -- Omega never noticed and just kept believing
    # the mission was "resumed, awaiting outcome" for 6+ minutes while it sat
    # paused the whole time, with no robot_event to tell anyone otherwise.
    manager = MissionManager()
    mission = _submit(manager, "find the woman")
    manager.set_plan(mission.identity.mission_id, "bt", "goal")
    reason = "awaiting Omega decision: condition UNKNOWN: no verification evidence for entity_located"

    manager.pause(mission.identity.mission_id, reason)
    manager.resume(mission.identity.mission_id, "camera confirmed it")
    event = manager.pause(mission.identity.mission_id, reason)

    assert event.event == EVENT_BLOCKED
    assert event.mission.state == STATE_BLOCKED
    assert "resuming again will not resolve this" in event.mission.status_text
    assert reason in event.mission.status_text


def test_resume_then_different_repause_reason_is_a_normal_pause():
    # The other half: a re-pause for a genuinely DIFFERENT reason (the mission
    # progressed to a new check, or the escalation text itself changed) is not
    # the stuck-loop pattern above and must not be forced to BLOCKED.
    manager = MissionManager()
    mission = _submit(manager, "find the woman")
    manager.set_plan(mission.identity.mission_id, "bt", "goal")

    manager.pause(mission.identity.mission_id, "awaiting Omega decision: is the door open?")
    manager.resume(mission.identity.mission_id, "it is open")
    event = manager.pause(mission.identity.mission_id, "awaiting Omega decision: is anyone home?")

    assert event.event == EVENT_PAUSED
    assert event.mission.state == STATE_PAUSED


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


# --- P0.1 (2026-09-06, GPT-approved): MissionOutcomeEnvelope construction ---
# in MissionEvent.payload_json for terminal events. See mission.py's own
# _build_mission_outcome_envelope/_mission_semantic_entity_ids/
# _mission_grounding_refs docstrings for the design rationale (mission-time
# data only, never a terminal-time re-query; honest [] when nothing matches;
# must never block the terminal event itself even on malformed legacy data).

def _mission(
    *, state, goal_spec_json="", bt_json="", context_json="", error_code="",
    execution_id="exec1",
) -> Mission:
    identity = Identity(mission_id="m1", execution_id=execution_id)
    return Mission(
        identity=identity, intent_text="test", source="voice", operator_id="user",
        priority=10, allow_queue=True, context_json=context_json, state=state,
        goal_spec_json=goal_spec_json, bt_json=bt_json, error_code=error_code,
    )


def _grounded_entities_context(entries: list[dict]) -> str:
    return json.dumps({"caller_context": {"grounded_entities": entries}})


def test_envelope_includes_semantic_entity_id_when_goal_targets_a_grounded_entity():
    context_json = _grounded_entities_context([
        {"alias": "66", "entity_id": "person_x1", "semantic_entity_id": "person_sem_1"},
    ])
    goal_spec_json = json.dumps({"predicate": "entity_approached", "args": {"entity_id": "person_x1"}})
    mission = _mission(
        state=STATE_SUCCEEDED, goal_spec_json=goal_spec_json, context_json=context_json)

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert envelope["schema"] == MISSION_OUTCOME_SCHEMA
    assert envelope["mission_id"] == "m1"
    assert envelope["execution_id"] == "exec1"
    assert envelope["terminal_state"] == "SUCCEEDED"
    assert envelope["goal_predicate"] == "entity_approached"
    assert envelope["semantic_entity_ids"] == ["person_sem_1"]
    assert envelope["grounding_refs"] == []
    assert "error_code" not in envelope


def test_envelope_semantic_entity_ids_empty_when_no_grounded_entity_matches():
    # Freshly-minted naming target (B-v1 TURN1 shape): goal_spec references
    # an entity_id that WorldState only bound to a semantic_entity_id DURING
    # execution -- this mission's own (planning-time) context_json has no
    # entry for it yet. Honest [], not a guess, not an error.
    goal_spec_json = json.dumps({"predicate": "person_named", "args": {"entity_id": "person_x1", "name": "66"}})
    mission = _mission(
        state=STATE_SUCCEEDED, goal_spec_json=goal_spec_json,
        context_json=_grounded_entities_context([]))

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert envelope["semantic_entity_ids"] == []


def test_envelope_omits_goal_predicate_and_subjects_for_implicit_human_goal_spec():
    goal_spec_json = json.dumps({"type": "human", "verification": {"mode": "implicit_conversation"}})
    mission = _mission(state=STATE_SUCCEEDED, goal_spec_json=goal_spec_json)

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert "goal_predicate" not in envelope
    assert envelope["semantic_entity_ids"] == []


def test_envelope_includes_grounding_ref_when_bt_json_carries_one():
    bt_json = json.dumps({
        "type": "Action", "skill": "remember_person",
        "args": {"entity_id": "person_x1", "name": "66", "_planning_grounding_ref": "gnd_abc123"},
    })
    mission = _mission(state=STATE_SUCCEEDED, bt_json=bt_json)

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert envelope["grounding_refs"] == ["gnd_abc123"]


def test_envelope_grounding_refs_empty_for_legacy_plan_without_the_field():
    bt_json = json.dumps({"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}})
    mission = _mission(state=STATE_SUCCEEDED, bt_json=bt_json)

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert envelope["grounding_refs"] == []


def test_envelope_includes_error_code_only_when_present():
    with_code = _mission(state=STATE_FAILED, error_code="NAV_TIMEOUT")
    without_code = _mission(state=STATE_FAILED)

    assert json.loads(_build_mission_outcome_envelope(with_code))["error_code"] == "NAV_TIMEOUT"
    assert "error_code" not in json.loads(_build_mission_outcome_envelope(without_code))


def test_envelope_omits_execution_id_when_legitimately_absent():
    mission = _mission(state=STATE_BLOCKED, execution_id="")

    envelope = json.loads(_build_mission_outcome_envelope(mission))

    assert "execution_id" not in envelope


def test_envelope_covers_succeeded_failed_blocked_canceled():
    for state, name in (
        (STATE_SUCCEEDED, "SUCCEEDED"), (STATE_FAILED, "FAILED"),
        (STATE_BLOCKED, "BLOCKED"), (STATE_CANCELED, "CANCELED"),
    ):
        envelope = json.loads(_build_mission_outcome_envelope(_mission(state=state)))
        assert envelope["terminal_state"] == name


def test_envelope_is_empty_string_for_non_terminal_state():
    # PAUSED is explicitly NOT in V1's terminal-state map (mid-mission,
    # not a mission outcome) -- confirms it is never blindly included.
    assert _build_mission_outcome_envelope(_mission(state=STATE_PAUSED)) == ""
    assert _build_mission_outcome_envelope(_mission(state=STATE_RUNNING)) == ""


def test_envelope_construction_never_raises_on_malformed_legacy_data():
    # Envelope generation must never be allowed to block the terminal event
    # itself -- garbage in every JSON-bearing field must still yield a
    # valid minimal envelope, never an exception.
    mission = _mission(
        state=STATE_SUCCEEDED, goal_spec_json="not json{{{",
        bt_json="also not json", context_json="{broken",
    )

    result = _build_mission_outcome_envelope(mission)

    envelope = json.loads(result)
    assert envelope["schema"] == MISSION_OUTCOME_SCHEMA
    assert envelope["mission_id"] == "m1"
    assert envelope["terminal_state"] == "SUCCEEDED"
    assert envelope["semantic_entity_ids"] == []
    assert envelope["grounding_refs"] == []


def test_mark_terminal_attaches_envelope_to_the_real_event_payload_json():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to 66", source="voice", operator_id="user",
        parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
    )
    manager.set_plan(
        mission.identity.mission_id,
        json.dumps({"type": "Action", "skill": "approach_entity", "args": {"entity_id": "person_x1"}}),
        json.dumps({"predicate": "entity_approached", "args": {"entity_id": "person_x1"}}),
    )

    event = manager.mark_terminal(
        mission.identity.mission_id, state=STATE_SUCCEEDED, message="goal check TRUE")

    assert event.event == EVENT_SUCCEEDED
    envelope = json.loads(event.payload_json)
    assert envelope["schema"] == MISSION_OUTCOME_SCHEMA
    assert envelope["mission_id"] == mission.identity.mission_id
    assert envelope["terminal_state"] == "SUCCEEDED"
    assert envelope["goal_predicate"] == "entity_approached"


def test_cancel_attaches_envelope_to_the_real_event_payload_json():
    manager = MissionManager()
    _accepted, _message, mission, _event = manager.submit(
        intent_text="go to 66", source="voice", operator_id="user",
        parent_mission_id="", priority=10, allow_queue=True, context_json="{}",
    )

    event = manager.cancel(mission.identity.mission_id, "user said stop")

    assert event.event == EVENT_CANCELED
    envelope = json.loads(event.payload_json)
    assert envelope["terminal_state"] == "CANCELED"


def test_preempted_payload_json_is_unaffected_by_the_new_envelope():
    # Regression guard: EVENT_PREEMPTED's own hand-built payload_json
    # ('{"preempted_mission_id": "..."}') must remain byte-for-byte what it
    # was before this change -- it is a different event type, never routed
    # through _build_mission_outcome_envelope at all.
    manager = MissionManager()
    _accepted, _message, first, _event = manager.submit(
        intent_text="first", source="voice", operator_id="user",
        parent_mission_id="", priority=1, allow_queue=True, context_json="{}",
    )
    manager.set_plan(first.identity.mission_id, "bt", "goal")
    _accepted, _message, second, _event = manager.submit(
        intent_text="second", source="voice", operator_id="user",
        parent_mission_id="", priority=2, allow_queue=True, context_json="{}",
    )

    event = manager.reprioritize(
        second.identity.mission_id, priority=50, preempt_if_needed=True, reason="urgent")

    assert event.event == EVENT_PREEMPTED
    assert json.loads(event.payload_json) == {"preempted_mission_id": first.identity.mission_id}
