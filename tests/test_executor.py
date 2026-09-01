import json
import threading
import time

from mc_ai_bt.executor import BtExecutor, ExecutionResult
from mc_ai_bt.goal_check import CheckResult, TriState


class FakeSkills:
    def __init__(self, fail_first: int = 0):
        self.said = []
        self.fail_first = fail_first
        self.timeouts = []

    def execute_skill(
        self,
        name: str,
        args: dict,
        cancel_event=None,
        *,
        timeout_sec=None,
    ) -> ExecutionResult:
        self.timeouts.append(timeout_sec)
        if name != "say":
            return ExecutionResult(False, f"unsupported skill: {name}")
        if len(self.said) < self.fail_first:
            self.said.append(args["text"])
            return ExecutionResult(False, "temporary failure")
        self.said.append(args["text"])
        return ExecutionResult(True, "said", {"said": args["text"]})


class FakeChecks:
    def __init__(self, state: TriState = TriState.TRUE):
        self.state = state
        self.requests = []

    def check(self, goal_spec: dict, execution: ExecutionResult):
        self.requests.append((goal_spec, execution))
        return CheckResult(self.state, f"{self.state.value.lower()} check")


class FakeChecksWithVisual(FakeChecks):
    def __init__(
        self,
        state: TriState = TriState.TRUE,
        visual_state: TriState = TriState.TRUE,
    ):
        super().__init__(state)
        self.visual_state = visual_state
        self.visual_requests = []

    def visual_check(self, check: dict, facts: dict):
        self.visual_requests.append((check, facts))
        return CheckResult(self.visual_state, f"{self.visual_state.value.lower()} visual")


class FakeChecksSequence:
    """Returns a different TriState on each successive call -- used to prove
    WaitForEvent actually re-polls instead of only checking once."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def check(self, goal_spec: dict, execution: ExecutionResult):
        self.calls += 1
        state = self.states[min(self.calls - 1, len(self.states) - 1)]
        return CheckResult(state, f"{state.value.lower()} check {self.calls}")


class FactSkills:
    def __init__(self):
        self.calls = []

    def execute_skill(
        self,
        name: str,
        args: dict,
        cancel_event=None,
        *,
        timeout_sec=None,
    ) -> ExecutionResult:
        self.calls.append((name, args, timeout_sec))
        if name == "fact":
            return ExecutionResult(True, "fact", {str(args["key"]): args["value"]})
        return ExecutionResult(False, f"unsupported skill: {name}")


def test_executor_runs_say_action():
    skills = FakeSkills()
    root = {"type": "Action", "skill": "say", "args": {"text": "hello"}}

    result = BtExecutor().execute(root, skills)

    assert result.success
    assert skills.said == ["hello"]
    assert result.facts == {"said": "hello"}
    assert skills.timeouts == [None]


def test_executor_passes_action_timeout_to_skill_adapter():
    skills = FakeSkills()
    root = {
        "type": "Action",
        "skill": "say",
        "args": {"text": "hello"},
        "timeout_sec": 12.5,
    }

    result = BtExecutor().execute(root, skills)

    assert result.success
    assert skills.timeouts == [12.5]


def test_executor_rejects_invalid_action_timeout_defensively():
    skills = FakeSkills()
    root = {
        "type": "Action",
        "skill": "say",
        "args": {"text": "hello"},
        "timeout_sec": 0,
    }

    result = BtExecutor().execute(root, skills)

    assert not result.success
    assert "timeout_sec" in result.message
    assert skills.said == []


def test_executor_stops_sequence_on_failure():
    skills = FakeSkills()
    root = {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "first"}},
            {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
            {"type": "Action", "skill": "say", "args": {"text": "third"}},
        ],
    }

    result = BtExecutor().execute(root, skills)

    assert not result.success
    assert skills.said == ["first"]


def test_executor_accepts_json_root():
    skills = FakeSkills()
    root_json = json.dumps({"type": "Action", "skill": "say", "args": {"text": "hi"}})

    result = BtExecutor().execute_json(root_json, skills)

    assert result.success
    assert skills.said == ["hi"]


def test_executor_stops_before_node_when_canceled():
    skills = FakeSkills()
    cancel_event = threading.Event()
    cancel_event.set()

    result = BtExecutor().execute(
        {"type": "Action", "skill": "say", "args": {"text": "hi"}},
        skills,
        cancel_event=cancel_event,
    )

    assert not result.success
    assert "canceled" in result.message
    assert skills.said == []


def test_executor_runs_retry_until_child_succeeds():
    skills = FakeSkills(fail_first=1)
    root = {
        "type": "Retry",
        "max_attempts": 2,
        "child": {"type": "Action", "skill": "say", "args": {"text": "again"}},
    }

    result = BtExecutor().execute(root, skills)

    assert result.success
    assert result.message == "retry succeeded on attempt 2"
    assert skills.said == ["again", "again"]
    assert result.facts == {"said": "again"}


def test_executor_wait_can_be_canceled():
    skills = FakeSkills()
    cancel_event = threading.Event()
    cancel_event.set()

    result = BtExecutor().execute(
        {"type": "Wait", "duration_sec": 1.0},
        skills,
        cancel_event=cancel_event,
    )

    assert not result.success
    assert result.message == "mission canceled"


def test_executor_noaction_succeeds_without_skill_call():
    skills = FakeSkills()

    result = BtExecutor().execute(
        {"type": "NoAction", "reason": "already satisfied"},
        skills,
    )

    assert result.success
    assert result.message == "already satisfied"
    assert skills.said == []


def test_executor_parallel_runs_all_children_and_merges_facts():
    skills = FactSkills()

    result = BtExecutor().execute(
        {
            "type": "Parallel",
            "children": [
                {"type": "Action", "skill": "fact", "args": {"key": "left", "value": 1}},
                {"type": "Action", "skill": "fact", "args": {"key": "right", "value": 2}},
            ],
        },
        skills,
    )

    assert result.success
    assert result.facts == {"left": 1, "right": 2}
    assert sorted(call[1]["key"] for call in skills.calls) == ["left", "right"]


def test_executor_timeout_returns_child_result_when_child_finishes():
    skills = FakeSkills()

    result = BtExecutor().execute(
        {
            "type": "Timeout",
            "timeout_sec": 1.0,
            "child": {"type": "Action", "skill": "say", "args": {"text": "quick"}},
        },
        skills,
    )

    assert result.success
    assert result.facts == {"said": "quick"}


def test_executor_timeout_blocks_and_cancels_slow_child():
    skills = FakeSkills()
    started = time.monotonic()

    result = BtExecutor().execute(
        {
            "type": "Timeout",
            "timeout_sec": 0.01,
            "child": {"type": "Wait", "duration_sec": 1.0},
        },
        skills,
    )

    assert not result.success
    assert result.blocked
    assert "timeout after" in result.message
    assert time.monotonic() - started < 0.5
    # A mechanical dead-end, not an evidentiary gap: replanning the same
    # timeout won't be fixed by Omega resolving some ambiguity, so this must
    # stay a terminal BLOCKED rather than becoming an escalation pause (see
    # OMEGACLAW_AI_BT_INTEGRATION.md §3.5 / node.py's _run_mission).
    assert not result.needs_decision


def test_executor_condition_true_continues_sequence():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)
    root = {
        "type": "Sequence",
        "children": [
            {
                "type": "Condition",
                "predicate": "robot_at_place",
                "args": {"name": "test_place"},
            },
            {"type": "Action", "skill": "say", "args": {"text": "done"}},
        ],
    }

    result = BtExecutor().execute(root, skills, checks=checks)

    assert result.success
    assert skills.said == ["done"]
    assert checks.requests[0][0]["predicate"] == "robot_at_place"


def test_executor_condition_unknown_blocks_instead_of_failing_false():
    skills = FakeSkills()
    checks = FakeChecks(TriState.UNKNOWN)

    result = BtExecutor().execute(
        {
            "type": "Condition",
            "predicate": "robot_at_place",
            "args": {"name": "test_place"},
        },
        skills,
        checks=checks,
    )

    assert not result.success
    assert result.blocked
    assert "UNKNOWN" in result.message
    # A genuine evidentiary gap: more/better evidence (a re-observe, asking
    # the person) could resolve this, so it's the one case that should
    # become a resumable escalation pause rather than a terminal BLOCKED.
    assert result.needs_decision


def test_wait_for_event_succeeds_immediately_when_true():
    checks = FakeChecks(TriState.TRUE)
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 5.0, "poll_interval_sec": 0.05}

    result = BtExecutor().execute(node, FakeSkills(), checks=checks)

    assert result.success
    assert len(checks.requests) == 1


def test_wait_for_event_polls_until_true():
    checks = FakeChecksSequence([TriState.FALSE, TriState.FALSE, TriState.TRUE])
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 5.0, "poll_interval_sec": 0.02}

    result = BtExecutor().execute(node, FakeSkills(), checks=checks)

    assert result.success
    assert checks.calls == 3


def test_wait_for_event_times_out_as_unknown_not_false():
    checks = FakeChecks(TriState.FALSE)
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 0.12, "poll_interval_sec": 0.05}

    result = BtExecutor().execute(node, FakeSkills(), checks=checks)

    assert not result.success
    assert result.blocked
    assert result.needs_decision
    assert "timed out" in result.message


def test_wait_for_event_respects_cancellation():
    checks = FakeChecks(TriState.FALSE)
    cancel_event = threading.Event()

    def _cancel_soon():
        time.sleep(0.05)
        cancel_event.set()

    threading.Thread(target=_cancel_soon).start()
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 5.0, "poll_interval_sec": 0.5}
    start = time.monotonic()

    result = BtExecutor().execute(node, FakeSkills(), cancel_event=cancel_event, checks=checks)

    assert time.monotonic() - start < 1.0
    assert not result.success
    assert "canceled" in result.message


def test_wait_for_event_requires_numeric_timeout():
    checks = FakeChecks(TriState.TRUE)
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "poll_interval_sec": 0.05}

    result = BtExecutor().execute(node, FakeSkills(), checks=checks)

    assert not result.success
    assert "timeout_sec must be numeric" in result.message


def test_wait_for_event_blocked_without_checker():
    node = {"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 1.0}

    result = BtExecutor().execute(node, FakeSkills(), checks=None)

    assert not result.success
    assert result.blocked


def test_goal_check_sees_facts_from_previous_action():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)
    root = {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "hi"}},
            {
                "type": "GoalCheck",
                "check": {
                    "predicate": "said",
                    "args": {"text": "hi"},
                },
            },
        ],
    }

    result = BtExecutor().execute(root, skills, checks=checks)

    assert result.success
    assert checks.requests[0][1].facts == {"said": "hi"}


def test_visual_check_query_runs_as_structured_object_check():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)

    result = BtExecutor().execute(
        {"type": "VisualCheck", "check": {"query": "do you see the test_object?"}},
        skills,
        checks=checks,
    )

    assert result.success
    assert checks.requests[0][0]["predicate"] == "object_visible"
    assert checks.requests[0][0]["args"] == {"name": "test_object"}


def test_visual_check_people_query_runs_as_structured_person_check():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)

    result = BtExecutor().execute(
        {"type": "VisualCheck", "check": {"query": "do you see anyone?"}},
        skills,
        checks=checks,
    )

    assert result.success
    assert checks.requests[0][0]["predicate"] == "person_visible"
    assert checks.requests[0][0]["args"] == {"min_count": 1}


def test_visual_check_unknown_query_blocks_without_calling_checker():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)

    result = BtExecutor().execute(
        {"type": "VisualCheck", "check": {"query": "is the room tidy?"}},
        skills,
        checks=checks,
    )

    assert not result.success
    assert result.blocked
    assert checks.requests == []
    # A wiring gap (no visual_check configured), not an evidentiary one --
    # Omega being asked to decide wouldn't help here either, so this must
    # NOT become an escalation pause.
    assert not result.needs_decision


def test_visual_check_unknown_query_uses_visual_service_when_configured():
    skills = FakeSkills()
    checks = FakeChecksWithVisual()

    result = BtExecutor().execute(
        {"type": "VisualCheck", "check": {"query": "is the room tidy?"}},
        skills,
        checks=checks,
    )

    assert result.success
    assert checks.requests == []
    assert checks.visual_requests == [({"query": "is the room tidy?"}, {})]


def test_visual_check_structured_unknown_falls_back_to_visual_service():
    skills = FakeSkills()
    checks = FakeChecksWithVisual(state=TriState.UNKNOWN, visual_state=TriState.TRUE)

    result = BtExecutor().execute(
        {"type": "VisualCheck", "check": {"query": "do you see the test_object?"}},
        skills,
        checks=checks,
    )

    assert result.success
    assert checks.requests[0][0]["predicate"] == "object_visible"
    assert checks.visual_requests == [({"query": "do you see the test_object?"}, {})]


def test_needs_decision_propagates_through_sequence():
    skills = FakeSkills()
    checks = FakeChecks(TriState.UNKNOWN)
    root = {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "first"}},
            {"type": "Condition", "predicate": "person_awake", "args": {}},
            {"type": "Action", "skill": "say", "args": {"text": "unreached"}},
        ],
    }

    result = BtExecutor().execute(root, skills, checks=checks)

    assert not result.success
    assert result.blocked
    assert result.needs_decision
    assert skills.said == ["first"]


def test_reexecuting_after_a_pause_replays_the_earlier_sequence_prefix():
    # Diagnostic experiment B (3rd-party review, 2026-09-01): confirms in code (not just by
    # reading mission.py's docstring/comments) that there is no checkpoint/continuation
    # anywhere in this executor. node.py's _run_mission calls BtExecutor.execute_json on the
    # mission's full bt_json on EVERY call, resume included -- there is no saved node
    # position, call stack, or blackboard carried between the pausing call and the resuming
    # one, just the same JSON tree and a fresh ExecutionResult. This test proves the
    # observable consequence directly: an Action before a paused Condition genuinely re-runs
    # when the same tree is executed again, it does not resume from the Condition.
    skills = FakeSkills()
    root = {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "first"}},
            {"type": "Condition", "predicate": "person_awake", "args": {}},
        ],
    }

    paused = BtExecutor().execute(root, skills, checks=FakeChecks(TriState.UNKNOWN))
    assert paused.blocked and paused.needs_decision
    assert skills.said == ["first"]

    # "Resume": the exact same tree, run again from the root, on the same skills provider --
    # this is what mission.bt_json + a fresh execute_json call actually does. No mechanism
    # anywhere skips straight to the Condition.
    resumed = BtExecutor().execute(root, skills, checks=FakeChecks(TriState.TRUE))
    assert resumed.success
    assert skills.said == ["first", "first"]


def test_needs_decision_propagates_through_fallback():
    skills = FakeSkills()
    checks = FakeChecks(TriState.UNKNOWN)
    root = {
        "type": "Fallback",
        "children": [
            {"type": "Condition", "predicate": "person_awake", "args": {}},
            {"type": "Action", "skill": "say", "args": {"text": "never reached"}},
        ],
    }

    result = BtExecutor().execute(root, skills, checks=checks)

    assert not result.success
    assert result.blocked
    assert result.needs_decision
    # Fallback stops at an UNKNOWN branch rather than trying the next one:
    # a genuine ambiguity needs resolving, not routing around.
    assert skills.said == []


def test_needs_decision_propagates_through_retry():
    skills = FakeSkills()
    checks = FakeChecks(TriState.UNKNOWN)

    result = BtExecutor().execute(
        {
            "type": "Retry",
            "max_attempts": 3,
            "child": {"type": "Condition", "predicate": "person_awake", "args": {}},
        },
        skills,
        checks=checks,
    )

    assert not result.success
    assert result.blocked
    assert result.needs_decision
    # Retrying the identical check can't manufacture new evidence -- exactly
    # why blocked short-circuits the remaining attempts.
    assert len(checks.requests) == 1


def test_a_plain_failure_does_not_set_needs_decision():
    """FALSE is a plain failure, not a gap -- it must not also pause."""
    skills = FakeSkills()
    checks = FakeChecks(TriState.FALSE)

    result = BtExecutor().execute(
        {"type": "Condition", "predicate": "person_awake", "args": {}},
        skills,
        checks=checks,
    )

    assert not result.success
    assert not result.blocked
    assert not result.needs_decision


def test_executor_reports_progress_for_sequence_leaves():
    skills = FakeSkills()
    events = []
    root = {
        "type": "Sequence",
        "children": [
            {"type": "Action", "skill": "say", "args": {"text": "one"}},
            {"type": "Action", "skill": "say", "args": {"text": "two"}},
        ],
    }

    result = BtExecutor().execute(
        root,
        skills,
        progress_callback=lambda label, progress: events.append((label, progress)),
    )

    assert result.success
    assert events == [
        ("root.children[0]:Action:say", 0.0),
        ("root.children[0]:Action:say", 0.5),
        ("root.children[1]:Action:say", 0.5),
        ("root.children[1]:Action:say", 1.0),
    ]


def test_executor_progress_counts_retry_attempt_budget():
    skills = FakeSkills(fail_first=1)
    events = []

    result = BtExecutor().execute(
        {
            "type": "Retry",
            "max_attempts": 2,
            "child": {"type": "Action", "skill": "say", "args": {"text": "again"}},
        },
        skills,
        progress_callback=lambda label, progress: events.append((label, progress)),
    )

    assert result.success
    assert events == [
        ("root.child[1]:Action:say", 0.0),
        ("root.child[1]:Action:say", 0.5),
        ("root.child[2]:Action:say", 0.5),
        ("root.child[2]:Action:say", 1.0),
    ]
