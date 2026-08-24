from mc_ai_bt.ros_doctor import DoctorCheck, _format_results, _overall_ok


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
