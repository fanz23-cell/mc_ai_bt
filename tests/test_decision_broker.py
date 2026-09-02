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
    _APPROACH_DISTANCE_TOLERANCE_M,
    _D_V1_SEMANTIC_SMOKE_TEST_PREDICATE,
    LocateResult,
    MissionBoundDecisionResolver,
    PHYSICAL_PREDICATES,
    SEMANTIC_PREDICATES,
    DecisionBroker,
    _match_available_classes,
)
from mc_ai_bt.identity import Identity  # noqa: E402
from mc_one.action import RequestHumanConfirmation  # noqa: E402
from mc_one.msg import AiBtIdentity  # noqa: E402


def _broker(**overrides) -> DecisionBroker:
    broker = DecisionBroker.__new__(DecisionBroker)
    broker._node = None
    broker._object_locator = overrides.get("object_locator")
    broker._person_visible_checker = overrides.get("person_visible_checker")
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


class _FixedLocator:
    def __init__(self, result: LocateResult):
        self._result = result
        self.calls = []

    def locate(self, target, *, timeout_sec=2.0):
        self.calls.append(target)
        return self._result


class _SequenceLocator:
    """Returns each LocateResult in order on successive .locate() calls."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def locate(self, target, *, timeout_sec=2.0):
        self.calls.append(target)
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx]


def test_resolve_returns_unknown_immediately_for_a_predicate_with_no_strategy():
    broker = _broker()

    outcome = broker.resolve(
        predicate="robot_at_place", args={}, reason="no strategy", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"


def test_predicate_classification_matches_the_documented_sets():
    assert PHYSICAL_PREDICATES == {"entity_approached", "object_visible", "person_visible"}
    # Not empty: the semantic-smoke-test predicate exists purely to exercise the
    # public resolve() -> confirmation -> DecisionOutcome path end to end, since no
    # real PREDICATE_REGISTRY entry needs semantic resolution today.
    assert SEMANTIC_PREDICATES == {_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE}


def test_approach_distance_tolerance_matches_mc_embodied_skills():
    assert _APPROACH_DISTANCE_TOLERANCE_M == 1.5


# --- entity_approached: direct, check-local, no WorldState write at all ------------


def test_entity_approached_true_when_fresh_locate_is_within_tolerance():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 0.9, "y": 0.0, "z": 0.0, "score": 0.8}))
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "TRUE"
    assert locator.calls == ["alice"]


def test_entity_approached_false_when_fresh_locate_is_still_far():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 3.0, "y": 4.0, "z": 0.0, "score": 0.8}))
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "FALSE"


def test_entity_approached_stays_unknown_when_multiple_same_class_matches_are_visible():
    # A2 ambiguity guard, ahead of C's real entity-identity layer: today's
    # class-based lookup has no notion of WHICH instance was actually
    # approached. LocalizeObject.srv sorts nearest-first, so without this
    # guard a SECOND same-class instance now closer than the one actually
    # navigated to (e.g. two chairs, robot ends up between them) would
    # silently confirm "approached" against the wrong one. Must never guess.
    locator = _FixedLocator(
        LocateResult("FOUND", fact={"x": 0.1, "y": 0.0, "z": 0.0, "score": 0.8, "match_count": 2})
    )
    broker = _broker(object_locator=locator, fast_window_sec=0.1, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "chair"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "UNKNOWN"
    assert "2 matching instances" in outcome.message


def test_entity_approached_true_when_exactly_one_match_even_with_match_count_field_present():
    # Regression: match_count=1 (the normal/common case) must not accidentally
    # trip the ambiguity guard.
    locator = _FixedLocator(
        LocateResult("FOUND", fact={"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.8, "match_count": 1})
    )
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "chair"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "TRUE"


def test_object_visible_ignores_match_count_ambiguity_by_design():
    # object_visible only asks "is ANY instance of this class visible" --
    # identity/which-instance is irrelevant to that question, unlike
    # entity_approached's "did I successfully reach THIS ONE". The ambiguity
    # guard must be scoped to entity_approached only.
    locator = _FixedLocator(
        LocateResult("FOUND", fact={"x": 0.5, "y": 0.0, "z": 0.0, "score": 0.8, "match_count": 3})
    )
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="object_visible", args={"name": "chair"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "TRUE"


def test_entity_approached_stays_unknown_when_target_not_found():
    # A single missed fresh locate is not proof the approach failed.
    locator = _FixedLocator(LocateResult("NOT_FOUND", reason="not found"))
    broker = _broker(object_locator=locator, fast_window_sec=0.1, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "UNKNOWN"


def test_entity_approached_stays_unknown_when_localizer_service_is_down():
    # FOUND LIVE 2026-09-01 (4th-party review): a perception OUTAGE must never read
    # the same as a real, completed "not visible" query.
    locator = _FixedLocator(LocateResult("INCONCLUSIVE", reason="object localization service unavailable"))
    broker = _broker(object_locator=locator, fast_window_sec=0.1, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="unknown",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "UNKNOWN"


def test_entity_approached_does_not_confuse_two_different_targets():
    # FOUND LIVE 2026-09-01 (4th-party review): the first cut wrote a single GLOBAL
    # objects.entity_approached key with no target scoping -- confirming Alice was
    # approached could satisfy an unrelated later check for Bob reading the same
    # stale key. Now there is no shared key at all: each resolve() call is entirely
    # check-local, driven only by ITS OWN fresh locate of ITS OWN target.
    locator = _SequenceLocator([
        LocateResult("FOUND", fact={"x": 0.1, "y": 0.0, "z": 0.0, "score": 0.9}),   # alice: close
        LocateResult("FOUND", fact={"x": 9.0, "y": 0.0, "z": 0.0, "score": 0.9}),   # bob: far
    ])
    broker = _broker(object_locator=locator)

    alice = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="u", facts={}, cancel_event=None)
    bob = broker.resolve(
        predicate="entity_approached", args={"target": "bob"}, reason="u", facts={}, cancel_event=None)

    assert alice.state == "TRUE"
    assert bob.state == "FALSE"  # not contaminated by alice's TRUE
    assert locator.calls == ["alice", "bob"]


def test_entity_approached_retries_within_the_fast_window_and_can_still_succeed():
    locator = _SequenceLocator([
        LocateResult("NOT_FOUND", reason="not found"),
        LocateResult("FOUND", fact={"x": 0.2, "y": 0.0, "z": 0.0, "score": 0.9}),
    ])
    broker = _broker(object_locator=locator, fast_window_sec=2.0, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="entity_approached", args={"target": "alice"}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "TRUE"
    assert len(locator.calls) >= 2


# --- object_visible: direct, check-local, no synthetic fact that could shadow a real
# multi-instance perception fact -----------------------------------------------------


def test_object_visible_true_when_found_above_min_score():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 1.0, "y": 1.0, "z": 0.0, "score": 0.73}))
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="object_visible", args={"name": "chair"}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "TRUE"


def test_object_visible_false_below_min_score():
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 1.0, "y": 1.0, "z": 0.0, "score": 0.2}))
    broker = _broker(object_locator=locator)

    outcome = broker.resolve(
        predicate="object_visible", args={"name": "chair", "min_score": 0.5}, reason="u",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "FALSE"


def test_object_visible_false_when_a_completed_query_finds_nothing():
    # A completed, fresh query that conclusively found nothing IS a real answer for
    # "is X visible right now" -- unlike entity_approached's past-tense claim.
    locator = _FixedLocator(LocateResult("NOT_FOUND", reason="not found"))
    broker = _broker(object_locator=locator, fast_window_sec=0.1, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="object_visible", args={"name": "medicine"}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "FALSE"


def test_object_visible_stays_unknown_on_service_outage_not_false():
    # FOUND LIVE 2026-09-01 (4th-party review), the most dangerous bug in the first
    # cut: service unavailable / timeout / exception must never read as a confirmed
    # "not visible" -- that would make a plain perception outage look like real
    # negative evidence.
    for reason in ("object localization service unavailable", "localize_object call failed: timeout"):
        locator = _FixedLocator(LocateResult("INCONCLUSIVE", reason=reason))
        broker = _broker(object_locator=locator, fast_window_sec=0.1, poll_interval_sec=0.02)

        outcome = broker.resolve(
            predicate="object_visible", args={"name": "chair"}, reason="u", facts={}, cancel_event=None)

        assert outcome.state == "UNKNOWN", reason


def test_object_visible_never_touches_world_state():
    # There is no world_writer on this broker at all -- if _check_object_visible
    # tried to write anything, this would raise AttributeError instead of quietly
    # succeeding, which is exactly the point: no synthetic "object:<name>" fact can
    # ever again shadow mc_world_state's real per-instance "object:<name>:<cell>" keys.
    locator = _FixedLocator(LocateResult("FOUND", fact={"x": 0.1, "y": 0.1, "z": 0.0, "score": 0.9}))
    broker = _broker(object_locator=locator)
    assert not hasattr(broker, "_world_writer")

    outcome = broker.resolve(
        predicate="object_visible", args={"name": "chair"}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "TRUE"


# --- LiveObjectLocator's own class-name matching (mirrors ObjectLocalizerClient) ---


def test_match_available_classes_bare_noun_matches_multiword_class():
    assert _match_available_classes("plant", ["chair", "potted plant"]) == ["potted plant"]


def test_match_available_classes_exact_match():
    assert _match_available_classes("chair", ["chair", "potted plant"]) == ["chair"]


def test_match_available_classes_no_plausible_match_is_empty():
    assert _match_available_classes("basketball", ["chair", "potted plant"]) == []


# --- person_visible: the one predicate that legitimately still polls a checker ----


def test_person_visible_returns_true_immediately():
    checks = _SequenceChecks(["TRUE"])
    broker = _broker(person_visible_checker=checks, fast_window_sec=5.0)

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "TRUE"
    assert checks.calls == 1


def test_person_visible_returns_false_immediately_not_continue_waiting():
    # The exact semantic _execute_wait_for_event gets wrong for this use case (FALSE
    # there means "keep waiting for it to become true") -- here FALSE is a real,
    # immediate answer.
    checks = _SequenceChecks(["FALSE"])
    broker = _broker(person_visible_checker=checks, fast_window_sec=5.0)

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "FALSE"
    assert checks.calls == 1


def test_person_visible_keeps_polling_through_unknown_then_returns_true():
    checks = _SequenceChecks(["UNKNOWN", "UNKNOWN", "TRUE"])
    broker = _broker(person_visible_checker=checks, fast_window_sec=5.0, poll_interval_sec=0.01)

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "TRUE"
    assert checks.calls == 3


def test_person_visible_gives_up_as_unknown_after_the_fast_window():
    checks = _SequenceChecks(["UNKNOWN"])
    broker = _broker(person_visible_checker=checks, fast_window_sec=0.05, poll_interval_sec=0.02)

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"
    assert checks.calls > 1


def test_person_visible_stops_promptly_when_canceled():
    checks = _SequenceChecks(["UNKNOWN"])
    broker = _broker(person_visible_checker=checks, fast_window_sec=5.0, poll_interval_sec=0.01)
    cancel_event = threading.Event()
    cancel_event.set()

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=cancel_event)

    assert outcome.state == "UNKNOWN"


def test_person_visible_unconfigured_checker_returns_unknown():
    broker = _broker(person_visible_checker=None, fast_window_sec=0.1)

    outcome = broker.resolve(
        predicate="person_visible", args={}, reason="u", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"


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


def test_resolve_semantic_approved_returns_true_via_public_resolve():
    # Exercises the PUBLIC resolve(), not the private method directly -- proves the
    # smoke-test predicate is actually registered and dispatches correctly, not just
    # that _resolve_semantic itself works in isolation.
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_APPROVED, reason="fine")
    broker = _broker(confirmation_client=client)

    outcome = broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={"person": "bob"}, reason="ambiguous",
        facts={}, cancel_event=None,
    )

    assert outcome.state == "TRUE"
    assert outcome.message == "fine"
    assert len(client.sent_goals) == 1
    assert client.sent_goals[0].request_id  # a real, non-empty id was generated


def test_resolve_semantic_denied_returns_false():
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_DENIED, reason="not now")
    broker = _broker(confirmation_client=client)

    outcome = broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="ambiguous", facts={}, cancel_event=None)

    assert outcome.state == "FALSE"
    assert outcome.message == "not now"


def test_resolve_semantic_timeout_returns_unknown():
    client = _FakeConfirmationClient(decision=RequestHumanConfirmation.Goal.DECISION_TIMEOUT)
    broker = _broker(confirmation_client=client)

    outcome = broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="ambiguous", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"


def test_resolve_semantic_unavailable_channel_returns_unknown_without_sending_a_goal():
    client = _FakeConfirmationClient(ready=False)
    broker = _broker(confirmation_client=client)

    outcome = broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="ambiguous", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"
    assert client.sent_goals == []


def test_resolve_semantic_rejected_goal_returns_unknown():
    client = _FakeConfirmationClient(accepted=False)
    broker = _broker(confirmation_client=client)

    outcome = broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="ambiguous", facts={}, cancel_event=None)

    assert outcome.state == "UNKNOWN"


def test_resolve_semantic_each_call_gets_a_fresh_request_id():
    client = _FakeConfirmationClient()
    broker = _broker(confirmation_client=client)

    broker.resolve(predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="first", facts={}, cancel_event=None)
    broker.resolve(predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="second", facts={}, cancel_event=None)

    ids = [goal.request_id for goal in client.sent_goals]
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_resolve_semantic_converts_the_bound_mission_identity_to_the_real_ros_type():
    # FOUND LIVE 2026-09-01 (4th-party review): the earlier version of this test used
    # identity = object() and asserted `is identity` -- which passed even though the
    # real code path skipped identity_to_msg() entirely and would have handed a bare
    # internal Identity dataclass straight to a ROS action Goal field expecting a
    # real AiBtIdentity message (a boundary every OTHER action adapter in this
    # package already converts at). This test now uses an actual Identity and checks
    # the real converted type/fields, not object identity of an opaque stand-in.
    client = _FakeConfirmationClient()
    broker = _broker(confirmation_client=client)
    identity = Identity(
        mission_id="mission-1", plan_version=3, execution_id="exec-1",
        parent_mission_id="parent-1", source="voice", operator_id="operator-1",
    )

    broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="u", facts={},
        cancel_event=None, identity=identity,
    )

    sent_identity = client.sent_goals[0].identity
    assert isinstance(sent_identity, AiBtIdentity)
    assert sent_identity.mission_id == "mission-1"
    assert sent_identity.plan_version == 3
    assert sent_identity.execution_id == "exec-1"
    assert sent_identity.parent_mission_id == "parent-1"
    assert sent_identity.source == "voice"
    assert sent_identity.operator_id == "operator-1"


def test_resolve_semantic_with_no_identity_sends_an_empty_ros_identity():
    client = _FakeConfirmationClient()
    broker = _broker(confirmation_client=client)

    broker.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="u", facts={},
        cancel_event=None, identity=None,
    )

    assert isinstance(client.sent_goals[0].identity, AiBtIdentity)


def test_mission_bound_resolver_forwards_the_real_identity_through_to_ros():
    # node.py never calls DecisionBroker.resolve() directly -- it always goes through
    # MissionBoundDecisionResolver.for_mission(mission.identity), which was the actual
    # path that skipped identity_to_msg(). This exercises THAT wrapper, not just the
    # broker's own resolve() in isolation.
    client = _FakeConfirmationClient()
    broker = _broker(confirmation_client=client)
    identity = Identity(mission_id="mission-2", plan_version=1, execution_id="exec-2")
    resolver = MissionBoundDecisionResolver(broker, identity)

    resolver.resolve(
        predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="u", facts={}, cancel_event=None)

    sent_identity = client.sent_goals[0].identity
    assert isinstance(sent_identity, AiBtIdentity)
    assert sent_identity.mission_id == "mission-2"
    assert sent_identity.execution_id == "exec-2"


class _SlowGoalHandle(_FakeGoalHandle):
    """get_result_async() never completes on its own -- only cancellation moves it,
    simulating a real in-flight confirmation wait."""

    def __init__(self):
        super().__init__(accepted=True, decision=RequestHumanConfirmation.Goal.DECISION_CANCELED)
        self._canceled = threading.Event()

    def get_result_async(self):
        return _NeverDoneFuture(self)

    def cancel_goal_async(self):
        self._canceled.set()
        return _FakeFuture(None)


class _NeverDoneFuture:
    def __init__(self, goal_handle):
        self._goal_handle = goal_handle

    def add_done_callback(self, callback):
        pass  # never fires -- this is the whole point

    def result(self):
        raise AssertionError("should never be called")


def test_resolve_semantic_returns_promptly_on_cancel_not_the_full_timeout():
    # FOUND LIVE 2026-09-01 (4th-party review): the earlier _wait_future called
    # on_cancel() when cancel_event fired but then kept waiting for the ORIGINAL
    # deadline anyway. Now a cancel gets its own short grace period.
    goal_handle = _SlowGoalHandle()
    client = _FakeConfirmationClient(accepted=True)
    client._goal_handle = goal_handle
    broker = _broker(confirmation_client=client, confirmation_timeout_sec=30.0)
    cancel_event = threading.Event()
    cancel_event.set()  # already canceled before the wait even starts

    started = threading.Event()

    def _run():
        started.set()
        outcome_box["outcome"] = broker.resolve(
            predicate=_D_V1_SEMANTIC_SMOKE_TEST_PREDICATE, args={}, reason="u",
            facts={}, cancel_event=cancel_event,
        )
        finished.set()

    outcome_box = {}
    finished = threading.Event()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    # The ORIGINAL confirmation_timeout_sec is 30s -- if cancel didn't return
    # promptly, this would still be running well within a couple of seconds.
    assert finished.wait(timeout=3.0), "resolve() did not return promptly after cancel"
    assert outcome_box["outcome"].state == "UNKNOWN"
    thread.join(timeout=1.0)
