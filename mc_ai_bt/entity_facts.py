from __future__ import annotations

import re
from typing import Any


def normalise_entity_name(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def object_fact_key(name: str) -> str:
    return "object:" + normalise_entity_name(name)


def object_name_from_args(args: dict[str, Any]) -> str:
    return str(
        args.get("name")
        or args.get("object_name")
        or args.get("object")
        or args.get("target")
        or ""
    ).strip()


def min_score_from_args(args: dict[str, Any]) -> float:
    try:
        value = float(args.get("min_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def person_id_from_args(args: dict[str, Any]) -> str:
    return str(
        args.get("id")
        or args.get("person_id")
        or args.get("name")
        or args.get("person")
        or ""
    ).strip()


def min_count_from_args(args: dict[str, Any]) -> int:
    try:
        value = int(args.get("min_count", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(20, value))


def find_object_fact(objects: dict[str, Any], object_name: str) -> Any:
    wanted = normalise_entity_name(object_name)
    if not wanted:
        return None

    direct = object_fact_key(object_name)
    if direct in objects:
        return objects[direct]

    for key, entry in objects.items():
        if object_fact_refers_to(str(key), entry, object_name):
            return entry
    return None


def object_fact_refers_to(key: str, entry: Any, object_name: str) -> bool:
    wanted = normalise_entity_name(object_name)
    if not wanted:
        return False

    key_name = key.removeprefix("object:")
    if normalise_entity_name(key_name) == wanted:
        return True

    value = fact_value(entry)
    if isinstance(value, dict):
        observed_name = normalise_entity_name(value.get("object_name"))
        if observed_name and observed_name == wanted:
            return True
    return False


def object_visibility_evidence(
    entry: Any,
    object_name: str,
    *,
    min_score: float = 0.0,
) -> dict[str, Any]:
    expected = normalise_entity_name(object_name)
    if not expected:
        return {
            "expected": object_name,
            "observed": None,
            "min_score": min_score,
            "matched": None,
            "reason": "missing_expected_object",
        }
    if entry is None:
        return {
            "expected": object_name,
            "observed": None,
            "min_score": min_score,
            "matched": None,
            "reason": "missing_fact",
        }

    value = fact_value(entry)
    if isinstance(value, str):
        value = {"object_name": value, "visible": True}
    elif value is True:
        value = {"object_name": object_name, "visible": True}
    elif value is False:
        value = {"object_name": object_name, "visible": False}

    if not isinstance(value, dict):
        return {
            "expected": object_name,
            "observed": value,
            "min_score": min_score,
            "matched": None,
            "reason": "unknown_shape",
        }

    observed_name = str(value.get("object_name") or object_name).strip()
    visible_value = value.get("visible", True)
    score = _optional_float(value.get("score"))
    observed = {
        "object_name": observed_name,
        "visible": visible_value,
        "score": score,
    }

    if normalise_entity_name(observed_name) != expected:
        return {
            "expected": object_name,
            "observed": observed,
            "min_score": min_score,
            "matched": False,
            "reason": "object_name_mismatch",
        }
    if visible_value is False:
        return {
            "expected": object_name,
            "observed": observed,
            "min_score": min_score,
            "matched": False,
            "reason": "not_visible",
        }
    if min_score > 0.0 and score is None:
        return {
            "expected": object_name,
            "observed": observed,
            "min_score": min_score,
            "matched": None,
            "reason": "score_missing",
        }
    if score is not None and score < min_score:
        return {
            "expected": object_name,
            "observed": observed,
            "min_score": min_score,
            "matched": False,
            "reason": "score_below_min",
        }

    return {
        "expected": object_name,
        "observed": observed,
        "min_score": min_score,
        "matched": True,
        "reason": "visible",
    }


def people_visibility_evidence(
    entry: Any,
    *,
    person_id: str = "",
    min_count: int = 1,
) -> dict[str, Any]:
    value = fact_value(entry)
    if value is None:
        return {
            "expected": {"person_id": person_id, "min_count": min_count},
            "observed": None,
            "matched": None,
            "reason": "missing_fact",
        }

    people: list[Any] = []
    count: int | None = None
    if isinstance(value, dict):
        people_value = value.get("people")
        if isinstance(people_value, list):
            people = people_value
        count = _optional_int(value.get("count"))
        if count is None and people:
            count = len(people)
    elif isinstance(value, list):
        people = value
        count = len(value)
    else:
        count = _optional_int(value)

    observed = {
        "count": count,
        "person_ids": _person_ids(people),
    }
    wanted = normalise_entity_name(person_id)
    if wanted:
        if people:
            matched = any(_person_matches(person, wanted) for person in people)
            return {
                "expected": {"person_id": person_id, "min_count": min_count},
                "observed": observed,
                "matched": matched,
                "reason": "person_visible" if matched else "person_not_visible",
            }
        if count == 0:
            return {
                "expected": {"person_id": person_id, "min_count": min_count},
                "observed": observed,
                "matched": False,
                "reason": "no_people_visible",
            }
        return {
            "expected": {"person_id": person_id, "min_count": min_count},
            "observed": observed,
            "matched": None,
            "reason": "missing_person_detail",
        }

    if count is None:
        return {
            "expected": {"person_id": person_id, "min_count": min_count},
            "observed": observed,
            "matched": None,
            "reason": "count_missing",
        }
    matched = count >= min_count
    return {
        "expected": {"person_id": person_id, "min_count": min_count},
        "observed": observed,
        "matched": matched,
        "reason": "people_visible" if matched else "too_few_people_visible",
    }


def fact_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _person_ids(people: list[Any]) -> list[str]:
    ids: list[str] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        raw = person.get("id") or person.get("person_id") or person.get("name")
        if raw:
            ids.append(str(raw))
    return ids


def _person_matches(person: Any, wanted: str) -> bool:
    if isinstance(person, str):
        return normalise_entity_name(person) == wanted
    if not isinstance(person, dict):
        return False
    candidates = (
        person.get("id"),
        person.get("person_id"),
        person.get("name"),
    )
    return any(normalise_entity_name(candidate) == wanted for candidate in candidates if candidate)
