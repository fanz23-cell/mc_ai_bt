import threading

import pytest

pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

from action_msgs.msg import GoalStatus

from mc_ai_bt.identity import Identity
from mc_ai_bt.resource_client import LeaseResult
from mc_ai_bt.skill_adapters import ActionBinding, RosSkillExecutor, _binding_with_node_timeout


class FakeLeaseClient:
    def __init__(self, result: LeaseResult):
        self.result = result
        self.acquired = []
        self.released = []

    def acquire(self, *, resources, reason, timeout_sec, identity=None):
        self.acquired.append(
            {
                "resources": resources,
                "reason": reason,
                "timeout_sec": timeout_sec,
                "identity": identity,
            }
        )
        return self.result

    def release(self, lease_id: str, *, reason: str = "done", identity=None):
        self.released.append({"lease_id": lease_id, "reason": reason, "identity": identity})


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeWorldState:
    def __init__(self):
        self.updates = []

    def update(self, update):
        self.updates.append(update)
        return True, "ok"


class _FakeFuture:
    """Minimal rclpy.Future double: add_done_callback fires synchronously,
    matching what _wait_future (skill_adapters.py) expects."""

    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _FakeGoalHandle:
    def __init__(self, *, accepted: bool, success: bool, message: str = "done"):
        self.accepted = accepted
        self.is_cancel_requested = False
        self._success = success
        self._message = message

    def get_result_async(self):
        wrapped = type("Wrapped", (), {})()
        wrapped.status = GoalStatus.STATUS_SUCCEEDED if self._success else GoalStatus.STATUS_ABORTED
        result = type("Result", (), {})()
        result.success = self._success
        result.message = self._message
        wrapped.result = result
        return _FakeFuture(wrapped)

    def cancel_goal_async(self):
        return _FakeFuture(None)


class FakeActionClient:
    def __init__(self, *, accepted: bool = True, success: bool = True):
        self.sent_goals = []
        self._goal_handle = _FakeGoalHandle(accepted=accepted, success=success)

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        self.sent_goals.append(goal)
        return _FakeFuture(self._goal_handle)


def _executor(*, lease_result: LeaseResult):
    executor = RosSkillExecutor.__new__(RosSkillExecutor)
    executor._identity_context = threading.local()
    executor._leases = FakeLeaseClient(lease_result)
    executor._speak_pub = FakePublisher()
    executor._world_state = FakeWorldState()
    executor._current_lock = threading.Lock()
    executor._current_goal_handle = None
    return executor


def test_say_acquires_voice_and_face_lease_before_publish():
    executor = _executor(lease_result=LeaseResult(True, "ok", "lease-1"))

    result = executor._say({"text": "hello"})

    assert result.success
    assert executor._leases.acquired == [
        {
            "resources": ("voice", "face"),
            "reason": "mc_ai_bt skill: say",
            "timeout_sec": 1.0,
            "identity": executor._leases.acquired[0]["identity"],
        }
    ]
    assert executor._leases.released == [
        {
            "lease_id": "lease-1",
            "reason": "say submitted",
            "identity": executor._leases.released[0]["identity"],
        }
    ]
    assert executor._speak_pub.messages[0].utterance == "hello"
    assert executor._world_state.updates[0].key == "last_utterance"


def test_say_does_not_publish_when_lease_is_denied():
    executor = _executor(lease_result=LeaseResult(False, "busy"))

    result = executor._say({"text": "hello"})

    assert not result.success
    assert "resource lease denied" in result.message
    assert executor._speak_pub.messages == []
    assert executor._leases.released == []


def test_say_passes_current_mission_identity_to_resource_lease():
    executor = _executor(lease_result=LeaseResult(True, "ok", "lease-1"))
    identity = Identity(
        mission_id="mission-1",
        plan_version=3,
        execution_id="exec-1",
        parent_mission_id="parent-1",
        source="voice",
        operator_id="operator-1",
    )

    with executor.use_identity(identity):
        result = executor._say({"text": "hello"})

    assert result.success
    acquired_identity = executor._leases.acquired[0]["identity"]
    released_identity = executor._leases.released[0]["identity"]
    assert acquired_identity.mission_id == "mission-1"
    assert acquired_identity.plan_version == 3
    assert acquired_identity.execution_id == "exec-1"
    assert acquired_identity.parent_mission_id == "parent-1"
    assert acquired_identity.source == "voice"
    assert acquired_identity.operator_id == "operator-1"
    assert released_identity.mission_id == "mission-1"
    assert released_identity.plan_version == 3
    assert released_identity.execution_id == "exec-1"


def test_run_action_acquires_and_releases_the_binding_s_resources():
    # Found live 2026-08-29 (§C5): dedicated-dispatch skills (go_to_place/
    # come_to_me/simple_move/play_animation/look_at/point_at) never acquired a
    # resource lease at all -- only `say` did -- so two of them could race on
    # the same base/body resource with no arbitration. _run_action is the
    # single dispatch point all of them share via ActionBinding.
    executor = _executor(lease_result=LeaseResult(True, "ok", "lease-1"))
    client = FakeActionClient(success=True)
    binding = ActionBinding(client, object, ("base",), 5.0, {"robot_at_place": "kitchen"})

    result = executor._run_action("go_to_place", binding, None, object(), None)

    assert result.success
    assert executor._leases.acquired == [
        {
            "resources": ("base",),
            "reason": "mc_ai_bt skill: go_to_place",
            "timeout_sec": 1.0,
            "identity": executor._leases.acquired[0]["identity"],
        }
    ]
    assert executor._leases.released == [
        {
            "lease_id": "lease-1",
            "reason": "go_to_place finished",
            "identity": executor._leases.released[0]["identity"],
        }
    ]
    assert len(client.sent_goals) == 1


def test_run_action_denied_lease_never_sends_the_goal():
    executor = _executor(lease_result=LeaseResult(False, "busy"))
    client = FakeActionClient(success=True)
    binding = ActionBinding(client, object, ("base",), 5.0, {})

    result = executor._run_action("simple_move", binding, None, object(), None)

    assert not result.success
    assert "resource lease denied" in result.message
    assert client.sent_goals == []
    assert executor._leases.released == []


def test_run_action_releases_the_lease_even_when_the_action_fails():
    executor = _executor(lease_result=LeaseResult(True, "ok", "lease-1"))
    client = FakeActionClient(success=False)
    binding = ActionBinding(client, object, ("body",), 5.0, {})

    result = executor._run_action("play_animation", binding, None, object(), None)

    assert not result.success
    assert executor._leases.released == [
        {
            "lease_id": "lease-1",
            "reason": "play_animation finished",
            "identity": executor._leases.released[0]["identity"],
        }
    ]


def test_node_timeout_can_only_shorten_action_binding_timeout():
    binding = ActionBinding(
        client=object(),
        goal_type=object,
        resources=("base",),
        timeout_sec=300.0,
        success_facts={},
    )

    shorter = _binding_with_node_timeout(binding, 12.0)
    longer = _binding_with_node_timeout(binding, 900.0)
    absent = _binding_with_node_timeout(binding, None)

    assert shorter.timeout_sec == 12.0
    assert longer.timeout_sec == 300.0
    assert absent is binding
