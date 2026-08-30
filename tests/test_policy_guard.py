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
                {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
            ],
        },
        {
            "type": "structured",
            "predicate": "robot_at_place",
            "args": {"name": "test_place"},
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
                {"type": "Action", "skill": "go_to_place", "args": {"name": "test_place"}},
                {"type": "VisualCheck", "check": {"query": "do you see the test_object?"}},
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
                {"type": "VisualCheck", "check": {"query": "do you see the test_object?"}},
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
            "args": {"name": "test_place"},
            "verification": {"mode": "world_state_or_nav_result"},
        },
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("does not match" in error for error in result.errors)


def test_policy_accepts_bounded_human_confirmation():
    plan = _plan(
        {
            "type": "Action",
            "skill": "request_human_confirmation",
            "args": {
                "prompt": "Approve opening the demo interaction?",
                "required_role": "operator",
                "timeout_sec": 30,
            },
        },
    )

    result = PolicyGuard().check(plan)

    assert result.ok


def test_policy_rejects_unbounded_human_confirmation():
    plan = _plan(
        {
            "type": "Action",
            "skill": "request_human_confirmation",
            "args": {
                "prompt": "x" * 300,
                "required_role": "stranger",
                "timeout_sec": 1000,
            },
        },
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("prompt exceeds" in error for error in result.errors)
    assert any("required_role" in error for error in result.errors)
    assert any("timeout_sec" in error for error in result.errors)


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
        {"type": "Action", "skill": "point_at", "args": {"object": "test_object", "arm": "right", "hold": 2.0}},
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


def test_policy_accepts_bounded_embodied_skills():
    reach = _plan({"type": "Action", "skill": "reach_to", "args": {"object": "test_object", "arm": "right"}})
    contact = _plan({"type": "Action", "skill": "wait_for_contact", "args": {"target": "generic_target"}})
    follow = _plan({"type": "Action", "skill": "follow_entity", "args": {"entity": "person:subject"}})
    guide = _plan(
        {
            "type": "Action",
            "skill": "guide_entity_to_place",
            "args": {"entity": "person:subject", "place": "target_place"},
        }
    )

    assert PolicyGuard().check(reach).ok
    assert PolicyGuard().check(contact).ok
    assert PolicyGuard().check(follow).ok
    assert PolicyGuard().check(guide).ok


def test_policy_rejects_embodied_skill_without_required_target():
    plan = _plan({"type": "Action", "skill": "reach_to", "args": {"arm": "right"}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("reach target" in error for error in result.errors)


def test_policy_accepts_approach_entity_with_target():
    plan = _plan({"type": "Action", "skill": "approach_entity", "args": {"target": "person"}})

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_approach_entity_without_target():
    plan = _plan({"type": "Action", "skill": "approach_entity", "args": {}})

    result = PolicyGuard().check(plan)

    assert not result.ok


def test_policy_accepts_face_entity_with_target():
    plan = _plan({"type": "Action", "skill": "face_entity", "args": {"target": "person"}})

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_face_entity_without_target():
    plan = _plan({"type": "Action", "skill": "face_entity", "args": {}})

    result = PolicyGuard().check(plan)

    assert not result.ok


def test_policy_accepts_wait_for_participant_default_condition():
    plan = _plan({"type": "Action", "skill": "wait_for_participant", "args": {"role": "customer"}})

    assert PolicyGuard().check(plan).ok


def test_policy_accepts_wait_for_participant_explicit_condition():
    plan = _plan(
        {"type": "Action", "skill": "wait_for_participant", "args": {"role": "customer", "condition": "hand_raised"}}
    )

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_wait_for_participant_bad_condition():
    plan = _plan(
        {"type": "Action", "skill": "wait_for_participant", "args": {"condition": "waving"}}
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("condition" in error for error in result.errors)


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


def test_policy_accepts_noaction_node():
    plan = _plan({"type": "NoAction", "reason": "already handled"})

    result = PolicyGuard().check(plan)

    assert result.ok


def test_policy_counts_actions_inside_parallel_and_timeout():
    plan = _plan(
        {
            "type": "Parallel",
            "children": [
                {
                    "type": "Timeout",
                    "timeout_sec": 5.0,
                    "child": {"type": "Action", "skill": "say", "args": {"text": "one"}},
                },
                {
                    "type": "Sequence",
                    "children": [
                        {"type": "Action", "skill": "say", "args": {"text": "two"}},
                        {"type": "Action", "skill": "say", "args": {"text": "three"}},
                    ],
                },
            ],
        },
    )

    result = PolicyGuard().check(plan)

    assert result.ok


def test_policy_rejects_wait_budget_inside_timeout():
    plan = _plan(
        {
            "type": "Timeout",
            "timeout_sec": 60.0,
            "child": {"type": "Wait", "duration_sec": 31.0},
        },
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("wait budget" in error for error in result.errors)
