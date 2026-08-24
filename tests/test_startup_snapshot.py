from __future__ import annotations

import threading

import pytest

from mc_ai_bt.identity import Identity
from mc_ai_bt.mission import Mission, STATE_BLOCKED

pytest.importorskip("rclpy")

from mc_ai_bt.node import AiBtNode


class _Missions:
    def __init__(self, missions):
        self._missions = tuple(missions)

    def all(self):
        return self._missions


class _Timer:
    def __init__(self):
        self.canceled = False

    def cancel(self):
        self.canceled = True


class _FakeNode:
    def __init__(self, missions, remaining=2):
        self._mission_lock = threading.RLock()
        self._missions = _Missions(missions)
        self._startup_snapshot_publishes_remaining = remaining
        self._startup_snapshot_timer = _Timer()
        self.statuses = []
        self.projections = 0
        self.destroyed_timers = []

    def _publish_status(self, mission):
        self.statuses.append(mission.identity.mission_id)

    def _publish_task_projection(self):
        self.projections += 1

    def destroy_timer(self, timer):
        self.destroyed_timers.append(timer)

    def get_logger(self):
        class _Log:
            def debug(self, _message):
                pass

        return _Log()


def _mission(mission_id="recovered"):
    return Mission(
        identity=Identity(mission_id=mission_id, plan_version=2),
        intent_text="go to kitchen",
        source="voice",
        operator_id="user",
        priority=10,
        allow_queue=True,
        context_json="{}",
        state=STATE_BLOCKED,
        status_text="recovered after node restart; manual review required",
        error_code="recovered_after_restart",
    )


def test_startup_snapshot_republishes_recovered_mission_statuses():
    fake = _FakeNode([_mission()], remaining=2)

    AiBtNode._publish_startup_snapshot(fake)

    assert fake.statuses == ["recovered"]
    assert fake.projections == 1
    assert fake._startup_snapshot_publishes_remaining == 1
    assert fake._startup_snapshot_timer is not None


def test_startup_snapshot_cleans_timer_after_last_publish():
    fake = _FakeNode([_mission()], remaining=1)

    timer = fake._startup_snapshot_timer
    AiBtNode._publish_startup_snapshot(fake)

    assert fake.statuses == ["recovered"]
    assert fake.projections == 1
    assert fake._startup_snapshot_publishes_remaining == 0
    assert fake._startup_snapshot_timer is None
    assert fake.destroyed_timers == [timer]
