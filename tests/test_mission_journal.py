from mc_ai_bt.mission import (
    EVENT_ACCEPTED,
    MissionEvent,
    MissionManager,
    STATE_BLOCKED,
    STATE_PLANNING,
    STATE_RUNNING,
)
from mc_ai_bt.mission_journal import MissionJournal


def _mission():
    manager = MissionManager()
    _accepted, _message, mission, event = manager.submit(
        intent_text="go to kitchen",
        source="voice",
        operator_id="user",
        parent_mission_id="",
        priority=10,
        allow_queue=True,
        context_json="{}",
    )
    return manager, mission, event


def test_mission_journal_appends_and_loads_latest_mission(tmp_path):
    path = tmp_path / "missions.jsonl"
    journal = MissionJournal(path)
    manager, mission, accepted = _mission()
    planned = manager.set_plan(mission.identity.mission_id, "bt", "goal")

    journal.append(accepted)
    journal.append(planned)

    loaded = journal.load_latest_missions(recover_nonterminal=False)

    assert len(loaded) == 1
    assert loaded[0].identity.mission_id == mission.identity.mission_id
    assert loaded[0].state == STATE_RUNNING
    assert loaded[0].bt_json == "bt"


def test_mission_journal_recovers_nonterminal_missions_as_blocked(tmp_path):
    path = tmp_path / "missions.jsonl"
    journal = MissionJournal(path)
    manager, mission, _accepted = _mission()
    planned = manager.set_plan(mission.identity.mission_id, "bt", "goal")

    journal.append(planned)

    recovered = journal.load_latest_missions(recover_nonterminal=True)[0]

    assert recovered.state == STATE_BLOCKED
    assert recovered.identity.execution_id == ""
    assert recovered.error_code == "recovered_after_restart"


def test_mission_journal_ignores_malformed_lines(tmp_path):
    path = tmp_path / "missions.jsonl"
    path.write_text("not json\n{}\n", encoding="utf-8")

    loaded = MissionJournal(path).load_latest_missions()

    assert loaded == ()


def test_mission_manager_can_start_from_recovered_missions():
    _manager, mission, _event = _mission()
    recovered = MissionEvent(
        EVENT_ACCEPTED,
        mission,
        "accepted",
    ).mission
    manager = MissionManager((recovered,))

    assert manager.get(mission.identity.mission_id).state == STATE_PLANNING
    assert manager.resolve_control_id("active") == mission.identity.mission_id
