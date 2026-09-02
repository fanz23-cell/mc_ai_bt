"""A2: an Action node's own post-execution verification can come back
genuinely unknown (mc_embodied_skills' approach_entity: navigation
succeeded, but the fresh post-arrival re-locate couldn't confirm the
target). Before this, that always ended the Sequence right there --
DecisionBroker never got a chance, because _execute_action had no
decision_resolver parameter at all (unlike _execute_condition/
_execute_goal_check).

Critical correctness constraint this suite exists to prove: the skill
(the real physical action) must run AT MOST ONCE, no matter what the
resolver decides -- and if the resolver ALSO can't resolve it, the result
must be blocked=True, needs_decision=False (-> node.py's _run_mission maps
this to a terminal STATE_BLOCKED / mark_terminal), never needs_decision=True
(-> mission.py's pause()/resume(), which replays the BT from its root and
would re-run the action a second time).
"""

from __future__ import annotations

from mc_ai_bt.executor import ACTION_POSTCONDITION_POLICY, BtExecutor, DecisionOutcome, ExecutionResult


class _RecordingSkills:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.calls = 0

    def execute_skill(self, name, args, cancel_event=None, *, timeout_sec=None):
        self.calls += 1
        return self._result


class _FakeResolver:
    def __init__(self, outcome: DecisionOutcome) -> None:
        self._outcome = outcome
        self.calls: list[dict] = []

    def resolve(self, *, predicate, args, reason, facts, cancel_event):
        self.calls.append({"predicate": predicate, "args": args})
        return self._outcome


_BLOCKED_WITH_REQUEST = ExecutionResult(
    False, "approach_entity: reached waypoint but lost track of alice on arrival", {
        "_resolution_request": {
            "kind": "postcondition_unknown", "predicate": "entity_approached", "args": {"target": "alice"},
        },
    },
    blocked=True,
)

_ACTION_NODE = {"type": "Action", "skill": "approach_entity", "args": {"target": "alice"}}


def test_policy_allowlist_covers_exactly_approach_entity_entity_approached():
    assert ACTION_POSTCONDITION_POLICY == {"approach_entity": {"entity_approached"}}


def test_resolver_true_succeeds_the_action_and_calls_the_skill_exactly_once():
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "alice is 0.80m away (fresh check)"))

    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=resolver)

    assert result.success is True
    assert result.blocked is False
    assert "resolved" in result.message
    assert skills.calls == 1
    assert resolver.calls == [{"predicate": "entity_approached", "args": {"target": "alice"}}]


def test_resolver_false_fails_the_action_and_calls_the_skill_exactly_once():
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    resolver = _FakeResolver(DecisionOutcome("FALSE", "alice is 4.20m away, not within 1.5m"))

    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=resolver)

    assert result.success is False
    assert result.blocked is False
    assert "failed (resolved)" in result.message
    assert skills.calls == 1


def test_resolver_still_unknown_is_blocked_terminal_never_needs_decision():
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    resolver = _FakeResolver(DecisionOutcome("UNKNOWN", "fast resolution window elapsed"))

    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=resolver)

    assert result.success is False
    assert result.blocked is True
    # The one invariant this whole test module exists to protect: this must
    # NEVER be True. needs_decision=True routes through mission.py's
    # pause()/resume(), which replays the BT from its root -- re-running the
    # already-executed physical navigation a second time.
    assert result.needs_decision is False
    assert "POSTCONDITION_UNRESOLVED_AFTER_SIDE_EFFECT" in result.message
    assert skills.calls == 1


def test_ordinary_blocked_result_without_resolution_request_is_untouched():
    # Regression: a skill's plain "input validation failed" BLOCKED (e.g.
    # remember_person requiring a target) must not be reinterpreted as a
    # postcondition-unknown case just because it happens to be blocked=True.
    plain_blocked = ExecutionResult(False, "remember_person requires target", {}, blocked=True)
    skills = _RecordingSkills(plain_blocked)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "should never be reached"))

    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=resolver)

    assert result is plain_blocked
    assert resolver.calls == []


def test_resolution_request_for_a_skill_not_in_the_policy_allowlist_is_ignored():
    unapproved = ExecutionResult(
        False, "some other skill blocked", {
            "_resolution_request": {"kind": "postcondition_unknown", "predicate": "entity_approached", "args": {}},
        },
        blocked=True,
    )
    skills = _RecordingSkills(unapproved)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "should never be reached"))
    node = {"type": "Action", "skill": "some_other_skill", "args": {}}

    result = BtExecutor().execute(node, skills, decision_resolver=resolver)

    assert result is unapproved
    assert resolver.calls == []


def test_resolution_request_predicate_not_approved_for_this_skill_is_ignored():
    wrong_predicate = ExecutionResult(
        False, "approach_entity blocked", {
            "_resolution_request": {"kind": "postcondition_unknown", "predicate": "object_visible", "args": {}},
        },
        blocked=True,
    )
    skills = _RecordingSkills(wrong_predicate)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "should never be reached"))

    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=resolver)

    assert result is wrong_predicate
    assert resolver.calls == []


def test_no_resolver_configured_leaves_the_blocked_result_untouched():
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    result = BtExecutor().execute(_ACTION_NODE, skills, decision_resolver=None)
    assert result is _BLOCKED_WITH_REQUEST


def test_inline_resolution_disallowed_inside_retry_never_calls_the_resolver():
    # Same scoping rule D v1 already established for Condition/GoalCheck:
    # Parallel/Retry/Timeout force _allow_inline_resolution=False on their
    # children, and Action nodes must respect that identically.
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "should never be reached"))
    root = {"type": "Retry", "max_attempts": 1, "child": _ACTION_NODE}

    result = BtExecutor().execute(root, skills, decision_resolver=resolver)

    assert resolver.calls == []
    assert skills.calls == 1
    assert result.blocked is True
    assert result.needs_decision is False


def test_sequence_continues_past_a_resolved_true_action_to_the_next_child():
    skills = _RecordingSkills(_BLOCKED_WITH_REQUEST)
    resolver = _FakeResolver(DecisionOutcome("TRUE", "alice is 0.80m away (fresh check)"))
    root = {"type": "Sequence", "children": [_ACTION_NODE, {"type": "Action", "skill": "say", "args": {}}]}

    class _SaySkills(_RecordingSkills):
        def execute_skill(self, name, args, cancel_event=None, *, timeout_sec=None):
            if name == "approach_entity":
                return super().execute_skill(name, args, cancel_event, timeout_sec=timeout_sec)
            self.calls += 1
            return ExecutionResult(True, "said")

    skills = _SaySkills(_BLOCKED_WITH_REQUEST)
    result = BtExecutor().execute(root, skills, decision_resolver=resolver)

    assert result.success is True
    assert skills.calls == 2  # approach_entity once, say once -- no replay of either
