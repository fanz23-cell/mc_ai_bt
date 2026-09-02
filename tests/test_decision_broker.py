"""DecisionBroker -- the real DecisionResolver BtExecutor calls into for D v1.

Constructs instances via __new__ (bypassing __init__, which builds real ROS
clients) and sets only the attributes each method actually touches -- the same
unbound-instance pattern used throughout this test suite (e.g.
test_skill_adapters.py's _executor helper) for anything that would otherwise
need a live rclpy node.
"""
import threading

import pytest

pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

from mc_ai_bt.decision_broker import (  # noqa: E402
    PHYSICAL_PREDICATES,
    SEMANTIC_PREDICATES,
    DecisionBroker,
    LiveObjectLocator,
    _APPROACH_DISTANCE_TOLERANCE_M,
)
from mc_ai_bt.executor import DecisionOutcome, ExecutionResult  # noqa: E402
from mc_one.action import RequestHumanConfirmation  # noqa: E402


def _broker(**overrides) -> DecisionBroker:
    broker = DecisionBroker.__new__(DecisionBroker)
    broker._node = None
    broker._checks = overrides.get("checks")
    broker._object_locator = overrides.get("object_locator")
    broker._world_writer = overrides.get("world_writer")
    broker._fast_window_sec = overrides.get("fast_window_sec", 1.0)
    broker._poll_interval_sec = overrides.get("poll_interval_sec", 0.05)
    broker._confirmation_timeout_sec = overrides.get("confirmation_timeout_sec", 1.0)
    broker._confirmation_client = overrides.get("confirmation_client")
    return broker


class _SequenceChecks:
    """Returns each state in order on successive .check() calls, holding the last
    one once exhausted."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def check(self, goal_spec, execution):
        idx = min(self.calls, len(self.states) - 1)
        self.calls += 1
        return type("Result", (), {"state": self.states[idx], "message": f"state={self.states[idx]}"})()


class _RecordingWorldWriter:
    def __init__(self):
        self.calls = []

    def update_fact(self, *, source, scope, key, value, merge=False, timeout_sec=0.5):
        self.calls.append({"source": source, "scope": scope, "key": key, "value": value})
        return True, "ok"


class _FakeObjectLocator:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def locate(self, object_name, *, timeout_sec=2.0):
        self.calls.append(object_name)
        return self._result


def test_resolve_returns_unknown_immediately_for_a_predicate_with_no_strategy():
    checks = _SequenceChecks(["UNKNOWN"])
    broker = _broker(checks=checks)

    outcome = broker.resolve(
        predicate="robot_at_place", args={}, reason="no strategy", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"
    assert checks.calls == 0  # never even asked the checker -- straight fallback


def test_predicate_classification_matches_the_documented_sets():
    assert PHYSICAL_PREDICATES == {"entity_approached", "object_visible", "person_visible"}
    assert SEMANTIC_PREDICATES == set()  # reserved, nothing registered yet


# --- poll_until_known: TRUE/FALSE return immediately, only UNKNOWN keeps polling ----


def test_poll_until_known_returns_true_immediately_without_polling_again():
    checks = _SequenceChecks(["TRUE"])
    broker = _broker(checks=checks, fast_window_sec=5.0)
    refreshed = []

    outcome = broker._poll_until_known({"predicate": "x"}, lambda: refreshed.append(1), None)

    assert outcome.state == "TRUE"
    assert checks.calls == 1
    assert refreshed == [1]


def test_poll_until_known_returns_false_immediately_not_continue_waiting():
    # The exact semantic _execute_wait_for_event gets wrong for this use case (FALSE
    # there means "keep waiting for it to become true") -- here FALSE is a real,
    # immediate answer.
    checks = _SequenceChecks(["FALSE"])
    broker = _broker(checks=checks, fast_window_sec=5.0)

    outcome = broker._poll_until_known({"predicate": "x"}, lambda: None, None)

    assert outcome.state == "FALSE"
    assert checks.calls == 1


def test_poll_until_known_keeps_polling_through_unknown_then_returns_true():
    checks = _SequenceChecks(["UNKNOWN", "UNKNOWN", "TRUE"])
    broker = _broker(checks=checks, fast_window_sec=5.0, poll_interval_sec=0.01)

    outcome = broker._poll_until_known({"predicate": "x"}, lambda: None, None)

    assert outcome.state == "TRUE"
    assert checks.calls == 3


def test_poll_until_known_gives_up_as_unknown_after_the_fast_window():
    checks = _SequenceChecks(["UNKNOWN"])
    broker = _broker(checks=checks, fast_window_sec=0.05, poll_interval_sec=0.02)

    outcome = broker._poll_until_known({"predicate": "x"}, lambda: None, None)

    assert outcome.state == "UNKNOWN"
    assert checks.calls > 1  # actually polled more than once before giving up


def test_poll_until_known_stops_promptly_when_canceled():
    checks = _SequenceChecks(["UNKNOWN"])
    broker = _broker(checks=checks, fast_window_sec=5.0, poll_interval_sec=0.01)
    cancel_event = threading.Event()
    cancel_event.set()

    outcome = broker._poll_until_known({"predicate": "x"}, lambda: None, cancel_event)

    assert outcome.state == "UNKNOWN"
    assert "canceled" in outcome.message


# --- entity_approached: fresh live locate + distance, written to WorldState ---------


def test_refresh_entity_approached_writes_matched_true_within_tolerance():
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator(({"x": 0.9, "y": 0.0, "z": 0.0, "score": 0.8}, "found"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_entity_approached({"target": "alice"})

    assert locator.calls == ["alice"]
    assert len(writer.calls) == 1
    call = writer.calls[0]
    assert call["scope"] == "objects" and call["key"] == "entity_approached"
    assert call["value"]["matched"] is True
    assert call["value"]["distance_after_arrival_m"] == pytest.approx(0.9)


def test_refresh_entity_approached_writes_matched_false_when_still_far():
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator(({"x": 3.0, "y": 4.0, "z": 0.0, "score": 0.8}, "found"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_entity_approached({"target": "alice"})

    assert writer.calls[0]["value"]["matched"] is False
    assert writer.calls[0]["value"]["distance_after_arrival_m"] == pytest.approx(5.0)


def test_refresh_entity_approached_writes_nothing_when_target_not_found():
    # A single missed fresh locate is not proof the approach failed -- must stay
    # UNKNOWN (no write at all), not become a confirmed matched=False. See the
    # matching comment in decision_broker.py's _refresh_entity_approached, added
    # after this exact scenario broke test_d_v1_inline_decision_resolution.py's
    # fallback-to-pause test.
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator((None, "not found"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_entity_approached({"target": "alice"})

    assert writer.calls == []


def test_refresh_entity_approached_does_nothing_without_a_target():
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator((None, "n/a"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_entity_approached({})

    assert locator.calls == []
    assert writer.calls == []


def test_approach_distance_tolerance_matches_mc_embodied_skills():
    # Mirrors mc_embodied_skills/node.py's _APPROACH_DISTANCE_TOLERANCE_M exactly --
    # no shared import path between the two images, so this pins the value so a
    # future change to one side doesn't silently drift from the other.
    assert _APPROACH_DISTANCE_TOLERANCE_M == 1.5


# --- object_visible: fresh live locate, written under the real object_fact_key -----


def test_refresh_object_visible_writes_visible_true_with_score():
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator(({"x": 1.0, "y": 1.0, "z": 0.0, "score": 0.73}, "found"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_object_visible({"object": "red chair"})

    assert locator.calls == ["red chair"]
    call = writer.calls[0]
    assert call["scope"] == "objects"
    assert call["key"] == "object:red chair"  # normalise_entity_name: lower + collapse
    assert call["value"] == {"object_name": "red chair", "visible": True, "score": 0.73}


def test_refresh_object_visible_writes_visible_false_when_not_found():
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator((None, "not found"))
    broker = _broker(object_locator=locator, world_writer=writer)

    broker._refresh_object_visible({"name": "medicine"})

    assert writer.calls[0]["value"] == {"object_name": "medicine", "visible": False}


# --- semantic/policy: reuses RequestHumanConfirmation, mission-local outcome only ---


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _FakeGoalHandle:
    def __init__(self, *, accepted: bool, decision: int, reason: str = ""):
        self.accepted = accepted
        self._decision = decision
        self._reason = reason

    def get_result_async(self):
        wrapped = type("Wrapped", (), {})()
        result = type("Result", (), {"decision": self._decision, "reason": self._reason})()
        wrapped.result = result
        return _FakeFuture(wrapped)

    def cancel_goal_async(self):
        return _FakeFuture(None)


class _FakeConfirmationClient:
    def __init__(self, *, ready: bool = True, accepted: bool = True, decision=None, reason: str = ""):
        self._ready = ready
        self._goal_handle = _FakeGoalHandle(
            accepted=accepted,
            decision=decision if decision is not None else RequestHumanConfirmation.Goal.DECISION_APPROVED,
            reason=reason,
        )
        self.sent_goals = []

    def wait_for_server(self, timeout_sec=None):
        return self._ready

    def send_goal_async(self, goal):
        self.sent_goals.append(goal)
        return _FakeFuture(self._goal_handle)


def test_resolve_semantic_approved_returns_true():
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_APPROVED, reason="fine")
    broker = _broker(confirmation_client=client)

    outcome = broker._resolve_semantic("may_interrupt", {"person": "bob"}, "ambiguous", None)

    assert outcome.state == "TRUE"
    assert outcome.message == "fine"
    assert len(client.sent_goals) == 1
    assert client.sent_goals[0].request_id  # a real, non-empty id was generated


def test_resolve_semantic_denied_returns_false():
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_DENIED, reason="not now")
    broker = _broker(confirmation_client=client)

    outcome = broker._resolve_semantic("may_interrupt", {}, "ambiguous", None)

    assert outcome.state == "FALSE"
    assert outcome.message == "not now"


def test_resolve_semantic_timeout_returns_unknown():
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_TIMEOUT)
    broker = _broker(confirmation_client=client)

    outcome = broker._resolve_semantic("may_interrupt", {}, "ambiguous", None)

    assert outcome.state == "UNKNOWN"


def test_resolve_semantic_unavailable_channel_returns_unknown_without_sending_a_goal():
    client = _FakeConfirmationClient(ready=False)
    broker = _broker(confirmation_client=client)

    outcome = broker._resolve_semantic("may_interrupt", {}, "ambiguous", None)

    assert outcome.state == "UNKNOWN"
    assert client.sent_goals == []


def test_resolve_semantic_rejected_goal_returns_unknown():
    client = _FakeConfirmationClient(accepted=False)
    broker = _broker(confirmation_client=client)

    outcome = broker._resolve_semantic("may_interrupt", {}, "ambiguous", None)

    assert outcome.state == "UNKNOWN"


def test_resolve_semantic_each_call_gets_a_fresh_request_id():
    client = _FakeConfirmationClient()
    broker = _broker(confirmation_client=client)

    broker._resolve_semantic("may_interrupt", {}, "first", None)
    broker._resolve_semantic("may_interrupt", {}, "second", None)

    ids = [goal.request_id for goal in client.sent_goals]
    assert len(ids) == 2
    assert ids[0] != ids[1]


# --- resolve() dispatch wiring, end to end (no ROS beyond the fakes above) ---------


def test_resolve_dispatches_physical_predicates_through_poll_until_known():
    checks = _SequenceChecks(["TRUE"])
    writer = _RecordingWorldWriter()
    locator = _FakeObjectLocator(({"x": 0.1, "y": 0.0, "z": 0.0, "score": 0.9}, "found"))
    broker = _broker(checks=checks, object_locator=locator, world_writer=writer, fast_window_sec=1.0)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "TRUE"
    assert locator.calls == ["alice"]  # the physical refresh actually ran
