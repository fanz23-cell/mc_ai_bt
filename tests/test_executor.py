import json
import threading

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
            {"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}},
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


def test_executor_condition_true_continues_sequence():
    skills = FakeSkills()
    checks = FakeChecks(TriState.TRUE)
    root = {
        "type": "Sequence",
        "children": [
            {
                "type": "Condition",
                "predicate": "robot_at_place",
                "args": {"name": "kitchen"},
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
            "args": {"name": "kitchen"},
        },
        skills,
        checks=checks,
    )

    assert not result.success
    assert result.blocked
    assert "UNKNOWN" in result.message


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
        {"type": "VisualCheck", "check": {"query": "do you see the cup?"}},
        skills,
        checks=checks,
    )

    assert result.success
    assert checks.requests[0][0]["predicate"] == "object_visible"
    assert checks.requests[0][0]["args"] == {"name": "cup"}


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
