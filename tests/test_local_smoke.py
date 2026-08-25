from mc_ai_bt.local_smoke import main, run_smoke


def test_local_smoke_runs_go_to_place_end_to_end():
    result = run_smoke("go to test_place")

    assert result.ok
    assert result.stage == "done"
    assert result.facts == {
        "say_submitted": "I'm going to test_place.",
        "robot_at_place": "test_place",
    }


def test_local_smoke_runs_animation_end_to_end():
    result = run_smoke("wave hello")

    assert result.ok
    assert result.facts == {"animation_played": "wave"}


def test_local_smoke_runs_look_and_point_end_to_end():
    look = run_smoke("look left")
    point = run_smoke("point at the test_object")

    assert look.ok
    assert look.facts["animation_played"] == "look_at"
    assert point.ok
    assert point.facts["animation_played"] == "point_at"


def test_local_smoke_cli_returns_zero_for_valid_intent(capsys):
    code = main(["move", "forward"])
    captured = capsys.readouterr()

    assert code == 0
    assert '"ok": true' in captured.out
