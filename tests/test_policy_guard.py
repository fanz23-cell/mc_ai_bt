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
    # 2026-09-02: goal alignment is now derived from
    # DEFAULT_SKILLS[*].result_predicates (see policy_guard.py's
    # PREDICATE_TO_SKILLS) instead of a hand-written table that used to
    # loosely accept the generic "animation_played" for these two skills as
    # well as play_animation -- look_at/point_at each declare their OWN more
    # specific predicate (look_at_static_completed / point_at), which is what
    # a real plan must use now; that tightening is intentional (a plan
    # pairing look_at with predicate=point_at, or vice versa, is now
    # correctly rejected instead of silently passing under the old shared
    # "animation_played" bucket).
    look = _plan(
        {"type": "Action", "skill": "look_at", "args": {"direction": "left_up", "hold": 1.0}},
        {
            "type": "structured",
            "predicate": "look_at_static_completed",
            "args": {"direction": "left_up"},
            "verification": {"mode": "action_result"},
        },
    )
    point = _plan(
        {"type": "Action", "skill": "point_at", "args": {"object": "test_object", "arm": "right", "hold": 2.0}},
        {
            "type": "structured",
            "predicate": "point_at",
            "args": {"target": "test_object"},
            "verification": {"mode": "action_result"},
        },
    )

    assert PolicyGuard().check(look).ok
    assert PolicyGuard().check(point).ok


def test_policy_rejects_look_at_paired_with_reach_completed_predicate():
    # The specific class of bug the derive-from-registry fix catches: a
    # non-voice skill's result predicate must actually be one of ITS OWN
    # declared predicates, not another physical skill's.
    #
    # UPDATED 2026-09-13 (see 62_CHANGE_APPROVAL_PLANNER_GOALSPEC_ENABLING_
    # SEQUENCE.md): this used to pair look_at with "point_at" -- point_at's
    # own result_predicates has since been trimmed to just
    # ("animation_played",) (the bare "point_at" predicate had zero real
    # evaluator anywhere in this codebase, confirmed by direct grep, and
    # was silently blocking the deterministic goal_spec fill from ever
    # firing for a bare point_at plan). "point_at" is therefore no longer
    # claimed by ANY skill's result_predicates, so PREDICATE_TO_SKILLS no
    # longer has anything to align it against -- this specific test's own
    # mechanism (a predicate claimed by a DIFFERENT physical skill) needs a
    # predicate that is still actually claimed by some other skill;
    # "reach_completed" (reach_to's own, untouched) preserves the exact
    # same test intent.
    mismatched = _plan(
        {"type": "Action", "skill": "look_at", "args": {"direction": "left_up"}},
        {
            "type": "structured",
            "predicate": "reach_completed",
            "args": {"target": "test_object"},
            "verification": {"mode": "action_result"},
        },
    )
    result = PolicyGuard().check(mismatched)
    assert not result.ok
    assert any("reach_completed" in error and "does not match" in error for error in result.errors)


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
    # Physical skills need a structured goal_spec, not the _plan() default implicit/human
    # one -- see test_policy_rejects_physical_mission_with_implicit_success below.
    # wait_for_contact is NOT in this list -- it's status="blocked", see
    # test_policy_rejects_blocked_skills below.
    reach = _plan(
        {"type": "Action", "skill": "reach_to", "args": {"object": "test_object", "arm": "right"}},
        {"predicate": "reach_completed"},
    )
    follow = _plan(
        {"type": "Action", "skill": "follow_entity", "args": {"entity": "person:subject"}},
        {"predicate": "entity_following"},
    )
    guide = _plan(
        {
            "type": "Action",
            "skill": "guide_entity_to_place",
            "args": {"entity": "person:subject", "place": "target_place"},
        },
        {"predicate": "entity_at_place"},
    )

    assert PolicyGuard().check(reach).ok
    assert PolicyGuard().check(follow).ok
    assert PolicyGuard().check(guide).ok


# --- status="blocked" skills must be structurally rejected, not discovered by trying them
# (FOUND LIVE 2026-08-31) --------------------------------------------------------------


def test_policy_rejects_blocked_skills():
    for skill, args in [
        ("align_axis", {"axis": "x", "target": "test_object"}),
        ("move_along_axis", {"axis": "x", "distance_m": 0.1}),
        ("maintain_distance", {"target": "person:subject", "distance_m": 1.0}),
        ("wait_for_contact", {"target": "generic_target"}),
        ("detect_contact", {"target": "generic_target"}),
    ]:
        plan = _plan({"type": "Action", "skill": skill, "args": args}, {"predicate": "contact_detected"})
        result = PolicyGuard().check(plan)
        assert not result.ok, f"{skill} should be rejected (status=blocked)"
        assert any("not available" in error for error in result.errors), (skill, result.errors)


def test_policy_rejects_embodied_skill_without_required_target():
    plan = _plan({"type": "Action", "skill": "reach_to", "args": {"arm": "right"}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("reach target" in error for error in result.errors)


def test_policy_accepts_approach_entity_with_target():
    plan = _plan(
        {"type": "Action", "skill": "approach_entity", "args": {"target": "person"}},
        {"predicate": "entity_approached"},
    )

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_approach_entity_without_target():
    plan = _plan({"type": "Action", "skill": "approach_entity", "args": {}})

    result = PolicyGuard().check(plan)

    assert not result.ok


def test_policy_accepts_face_entity_with_target():
    plan = _plan(
        {"type": "Action", "skill": "face_entity", "args": {"target": "person"}},
        {"predicate": "entity_faced"},
    )

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_face_entity_without_target():
    plan = _plan({"type": "Action", "skill": "face_entity", "args": {}})

    result = PolicyGuard().check(plan)

    assert not result.ok


# --- physical missions must not use implicit/human success (FOUND LIVE 2026-08-31,
# real mission-journal audit: 66 of 143 real missions were exactly this shape) --------


def test_policy_rejects_physical_mission_with_implicit_success():
    # _plan()'s default goal_spec IS the implicit/human shape -- this is the exact
    # thing 66 real missions did.
    plan = _plan({"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("implicit" in error and "physical" in error for error in result.errors)


def test_policy_accepts_non_physical_mission_with_implicit_success():
    # say/request_human_confirmation are conversational, not physical -- implicit success
    # is a legitimate criterion for them and must not be rejected by the new rule.
    plan = _plan({"type": "Action", "skill": "say", "args": {"text": "hello"}})

    assert PolicyGuard().check(plan).ok


def test_policy_accepts_physical_mission_with_structured_success():
    plan = _plan(
        {"type": "Action", "skill": "go_to_place", "args": {"name": "kitchen"}},
        {"predicate": "robot_at_place"},
    )

    assert PolicyGuard().check(plan).ok


def test_policy_accepts_remember_place_with_name():
    plan = _plan({"type": "Action", "skill": "remember_place", "args": {"name": "front_desk"}})

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_remember_place_without_name():
    plan = _plan({"type": "Action", "skill": "remember_place", "args": {}})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("name" in error for error in result.errors)


def test_policy_accepts_remember_person_with_target_and_name():
    plan = _plan(
        {"type": "Action", "skill": "remember_person", "args": {"target": "person on the left", "name": "Alice"}},
        {"predicate": "person_named"},
    )

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_remember_person_without_name():
    plan = _plan(
        {"type": "Action", "skill": "remember_person", "args": {"target": "person on the left"}},
        {"predicate": "person_named"},
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("name" in error for error in result.errors)


def test_policy_rejects_remember_person_without_target():
    plan = _plan(
        {"type": "Action", "skill": "remember_person", "args": {"name": "Alice"}},
        {"predicate": "person_named"},
    )

    result = PolicyGuard().check(plan)

    assert not result.ok


def test_policy_accepts_remember_entity_with_target_and_alias():
    # Same physical-skill-needs-a-structured-goal rule remember_person's own
    # test above already works around (resources_for_skill declares "base").
    # 2026-09-02: entity_alias_bound is now a real, checked alignment entry
    # (PREDICATE_TO_SKILLS, derived from remember_entity's own
    # result_predicates) -- this must pass BECAUSE the predicate correctly
    # names remember_entity's own result, not merely because the check used
    # to be skipped for an unknown predicate name (see the rejection test
    # right below, which is the regression this fix closes).
    plan = _plan(
        {"type": "Action", "skill": "remember_entity", "args": {"target": "chair", "alias": "my chair"}},
        {"predicate": "entity_alias_bound"},
    )

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_remember_entity_paired_with_an_unrelated_predicate():
    # FOUND 2026-09-02 (GPT review): before this fix, PolicyGuard's
    # hand-written alignment table had no entry at all for remember_entity,
    # so ANY predicate silently passed alignment for it -- confirmed live:
    # this round's own C.2 acceptance test had to fall back to the unrelated
    # person_named predicate (remember_person's, not remember_entity's own)
    # and PolicyGuard let it through unchecked. Must now be rejected.
    plan = _plan(
        {"type": "Action", "skill": "remember_entity", "args": {"target": "chair", "alias": "my chair"}},
        {"predicate": "person_named"},
    )

    result = PolicyGuard().check(plan)
    assert not result.ok
    assert any("person_named" in error and "does not match" in error for error in result.errors)


def test_policy_rejects_remember_entity_without_alias():
    plan = _plan(
        {"type": "Action", "skill": "remember_entity", "args": {"target": "chair"}},
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("alias" in error for error in result.errors)


def test_policy_rejects_remember_entity_without_target():
    plan = _plan(
        {"type": "Action", "skill": "remember_entity", "args": {"alias": "my chair"}},
    )

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


def test_policy_accepts_wait_for_event_within_bounds():
    plan = _plan(
        {
            "type": "WaitForEvent",
            "predicate": "participant_ready",
            "args": {"role": "customer"},
            "poll_interval_sec": 2.0,
            "timeout_sec": 600.0,
        }
    )

    assert PolicyGuard().check(plan).ok


def test_policy_accepts_wait_for_event_default_poll_interval():
    plan = _plan({"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 60.0})

    assert PolicyGuard().check(plan).ok


def test_policy_rejects_wait_for_event_poll_interval_out_of_bounds():
    plan = _plan(
        {"type": "WaitForEvent", "predicate": "participant_ready", "poll_interval_sec": 0.1, "timeout_sec": 60.0}
    )

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("poll_interval_sec" in error for error in result.errors)


def test_policy_rejects_wait_for_event_timeout_missing():
    plan = _plan({"type": "WaitForEvent", "predicate": "participant_ready"})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("timeout_sec" in error for error in result.errors)


def test_policy_rejects_wait_for_event_timeout_too_large():
    plan = _plan({"type": "WaitForEvent", "predicate": "participant_ready", "timeout_sec": 1801.0})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("timeout_sec" in error for error in result.errors)


def test_policy_rejects_wait_for_event_unsafe_predicate():
    plan = _plan({"type": "WaitForEvent", "predicate": "__import__", "timeout_sec": 60.0})

    result = PolicyGuard().check(plan)

    assert not result.ok
    assert any("policy-safe" in error for error in result.errors)


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
