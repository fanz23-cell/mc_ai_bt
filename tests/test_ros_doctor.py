from mc_ai_bt.ros_doctor import (
    DoctorCheck,
    _core_services,
    _format_results,
    _human_confirmation_actions,
    _overall_ok,
    _skill_actions,
    _visual_check_actions,
)


def test_doctor_summary_passes_when_all_checks_pass():
    results = [
        DoctorCheck("ai_bt.list_missions", True, "/mc_ai_bt/list_missions", "service"),
        DoctorCheck("call.resource_authority.list", True, "0 lease status(es)", "call"),
    ]

    assert _overall_ok(results)
    formatted = _format_results(results)
    assert "PASS service ai_bt.list_missions" in formatted
    assert "summary: PASS 2  FAIL 0" in formatted


def test_doctor_summary_fails_when_any_check_fails():
    results = [
        DoctorCheck("ai_bt.list_missions", True, "/mc_ai_bt/list_missions", "service"),
        DoctorCheck("skill.go_to_place", False, "unavailable", "action"),
    ]

    assert not _overall_ok(results)
    formatted = _format_results(results)
    assert "FAIL action skill.go_to_place -- unavailable" in formatted
    assert "summary: PASS 1  FAIL 1" in formatted


def test_core_services_include_operator_control_surfaces():
    labels = {label for label, _srv_type, _name in _core_services()}

    assert "ai_bt.reprioritize_mission" in labels
    assert "ai_bt.set_policy_state" in labels
    assert "ai_bt.respond_human_confirmation" in labels


def test_optional_action_checks_include_embodied_and_visual_servers():
    skill_labels = {label for label, _action_type, _name in _skill_actions()}
    visual_labels = {label for label, _action_type, _name in _visual_check_actions()}
    human_labels = {label for label, _action_type, _name in _human_confirmation_actions()}

    assert "skill.embodied" in skill_labels
    assert "visual.visual_check" in visual_labels
    assert "human.request_confirmation" in human_labels
