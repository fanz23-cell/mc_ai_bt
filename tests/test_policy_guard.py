from mc_ai_bt.policy_guard import PolicyGuard


def _plan(root, goal_spec=None):
    return {
        "schema": "mc_ai_bt.plan.v1",
        "root": root,
        "goal_spec": goal_spec or {
            "type": "human",
            "verification": {"mode": "implicit_conversation"},
        },
    }


def test_policy_accepts_bootstrap_nav_shape():
    plan = _plan(
        {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "say", "args": {"text": "Going."}},
                {"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}},
            ],
        },
        {
            "type": "structured",
            "predicate": "robot_at_place",
            "args": {"name": "kitchen"},
            "verification": {"mode": "world_state_or_nav_result"},
        },
    )

    result = PolicyGuard().check(plan)

    assert result.ok


def test_policy_accepts_bounded_multi_step_embodied_plan():
    plan = _plan(
        {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "say", "args": {"text": "Starting."}},
                {"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}},
                {"type": "VisualCheck", "check": {"query": "do you see the cup?"}},
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
            ],
        },
        {
            "type": "structured",
            "predicate": "animation_played",
            "args": {"animation": "wave"},
            "verification": {"mode": "action_result"},
        },
    )

    result = PolicyGuard().check(plan)

    assert result.ok


def test_policy_rejects_too_many_physical_actions():
    plan = _plan(
        {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
                {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "left", "value": 90}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "right", "value": 90}},
            ],
        }
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("physical" in error for error in result.errors)


def test_policy_rejects_too_many_base_actions():
    plan = _plan(
        {
            "type": "Sequence",
            "children": [
                {"type": "Action", "skill": "simple_move", "args": {"action": "left", "value": 45}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "right", "value": 45}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "left", "value": 45}},
                {"type": "Action", "skill": "simple_move", "args": {"action": "right", "value": 45}},
            ],
        }
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("base actions" in error for error in result.errors)


def test_policy_rejects_large_simple_move():
    plan = _plan({"type": "Action", "skill": "simple_move", "args": {"action": "forward", "value": 2.0}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("1m" in error for error in result.errors)


def test_policy_rejects_wait_and_retry_budget_explosion():
    plan = _plan(
        {
            "type": "Retry",
            "max_attempts": 5,
            "child": {
                "type": "Sequence",
                "children": [
                    {"type": "Wait", "duration_sec": 10},
                    {"type": "Action", "skill": "say", "args": {"text": "still trying"}},
                ],
            },
        }
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("wait budget" in error for error in result.errors)


def test_policy_rejects_too_many_visual_checks():
    plan = _plan(
        {
            "type": "Sequence",
            "children": [
                {"type": "VisualCheck", "check": {"query": "do you see the cup?"}},
                {"type": "VisualCheck", "check": {"query": "do you see the bottle?"}},
                {"type": "VisualCheck", "check": {"query": "do you see the book?"}},
                {"type": "VisualCheck", "check": {"query": "do you see the pen?"}},
                {"type": "VisualCheck", "check": {"query": "do you see the phone?"}},
            ],
        }
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("visual checks" in error for error in result.errors)


def test_policy_rejects_goal_action_mismatch():
    plan = _plan(
        {"type": "Action", "skill": "play_animation", "args": {"animation": "wave"}},
        {
            "type": "structured",
            "predicate": "robot_at_place",
            "args": {"name": "kitchen"},
            "verification": {"mode": "world_state_or_nav_result"},
        },
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("does not match" in error for error in result.errors)


def test_policy_accepts_look_at_and_point_at_skills():
    look = _plan(
        {"type": "Action", "skill": "look_at", "args": {"direction": "left_up", "hold": 1.0}},
        {
            "type": "structured",
            "predicate": "animation_played",
            "args": {"animation": "look_at"},
            "verification": {"mode": "action_result"},
        },
    )
    point = _plan(
        {"type": "Action", "skill": "point_at", "args": {"object": "cup", "arm": "right", "hold": 2.0}},
        {
            "type": "structured",
            "predicate": "animation_played",
            "args": {"animation": "point_at"},
            "verification": {"mode": "action_result"},
        },
    )

    assert PolicyGuard().check(look).ok
    assert PolicyGuard().check(point).ok


def test_policy_rejects_invalid_look_at_direction():
    plan = _plan({"type": "Action", "skill": "look_at", "args": {"direction": "sideways"}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("direction" in error for error in result.errors)


def test_policy_rejects_invalid_point_at_target():
    missing = _plan({"type": "Action", "skill": "point_at", "args": {"arm": "right"}})
    huge = _plan({"type": "Action", "skill": "point_at", "args": {"x": 10, "y": 0, "z": 0.5}})

    missing_result = PolicyGuard().check(missing)
    huge_result = PolicyGuard().check(huge)

    assert not missing_result.ok
    assert any("exactly one target" in error for error in missing_result.errors)
    assert not huge_result.ok
    assert any("x exceeds" in error for error in huge_result.errors)


def test_policy_rejects_unsafe_visual_check_predicate():
    plan = _plan(
        {
            "type": "VisualCheck",
            "check": {"predicate": "__import__", "args": {}},
        }
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("policy-safe" in error for error in result.errors)
