import threading

import pytest

pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

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


def _executor(*, lease_result: LeaseResult):
    executor = RosSkillExecutor.__new__(RosSkillExecutor)
    executor._identity_context = threading.local()
    executor._leases = FakeLeaseClient(lease_result)
    executor._speak_pub = FakePublisher()
    executor._world_state = FakeWorldState()
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
