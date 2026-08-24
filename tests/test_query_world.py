import json

from mc_ai_bt.query_world import (
    CERTAINTY_FALSE,
    CERTAINTY_TRUE,
    CERTAINTY_UNKNOWN,
    QueryWorldEngine,
)


def test_query_world_answers_current_place():
    world = {
        "facts": {
            "navigation": {
                "current_place": {
                    "value": "kitchen",
                    "observed_at": 10,
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("where are you?", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert answer.answer_text == "I am at kitchen."


def test_query_world_answers_active_task():
    world = {
        "facts": {
            "tasks": {
                "active_mission": {
                    "value": {
                        "title": "go to kitchen",
                        "state_name": "RUNNING",
                        "status_text": "running",
                    }
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("what are you doing?", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert answer.answer_text == "I am running: go to kitchen."


def test_query_world_reports_no_active_task_when_fact_is_null():
    world = {"facts": {"tasks": {"active_mission": {"value": None}}}}

    answer = QueryWorldEngine().answer_json("current task", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert "do not have an active" in answer.answer_text


def test_query_world_answers_active_safety_stop():
    world = {
        "facts": {
            "safety": {
                "estop": {
                    "value": {"active": True},
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("are you in safety stop?", json.dumps(world))

    assert answer.certainty == CERTAINTY_FALSE
    assert "active safety-stop" in answer.answer_text


def test_query_world_answers_clear_safety_stop():
    world = {
        "facts": {
            "safety": {
                "estop": {
                    "value": {"active": False},
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("急停了吗？", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert "do not have an active" in answer.answer_text


def test_query_world_answers_object_visibility():
    world = {
        "facts": {
            "objects": {
                "object:cup": {
                    "value": {
                        "object_name": "cup",
                        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                        "score": 0.87,
                    }
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("do you see the cup?", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert "cup" in answer.answer_text
    assert "confidence 0.87" in answer.answer_text


def test_query_world_answers_chinese_object_visibility_alias():
    world = {
        "facts": {
            "objects": {
                "object:cup": {
                    "value": {
                        "object_name": "cup",
                        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                        "score": 0.87,
                    }
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("你看到杯子了吗？", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert "cup" in answer.answer_text


def test_query_world_reports_false_when_object_fact_says_not_visible():
    world = {
        "facts": {
            "objects": {
                "object:cup": {
                    "value": {
                        "object_name": "cup",
                        "visible": False,
                        "score": 0.0,
                    }
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("do you see the cup?", json.dumps(world))

    assert answer.certainty == CERTAINTY_FALSE
    assert "do not currently" in answer.answer_text


def test_query_world_answers_people_visibility():
    world = {
        "facts": {
            "people": {
                "visible_people": {
                    "value": {
                        "count": 2,
                        "people": [{"id": "person:0"}, {"id": "person:1"}],
                    }
                }
            }
        }
    }

    answer = QueryWorldEngine().answer_json("how many people do you see?", json.dumps(world))

    assert answer.certainty == CERTAINTY_TRUE
    assert "2 visible people" in answer.answer_text


def test_query_world_reports_false_when_no_people_visible():
    world = {"facts": {"people": {"visible_people": {"value": {"count": 0, "people": []}}}}}

    answer = QueryWorldEngine().answer_json("do you see anyone?", json.dumps(world))

    assert answer.certainty == CERTAINTY_FALSE
    assert "do not currently" in answer.answer_text


def test_query_world_unknown_when_no_relevant_fact_exists():
    answer = QueryWorldEngine().answer_json("where are you?", json.dumps({"facts": {}}))

    assert answer.certainty == CERTAINTY_UNKNOWN
    assert "do not have" in answer.answer_text
