from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .entity_facts import (
    fact_value,
    normalise_entity_name,
    object_fact_key,
    object_name_from_args,
    person_id_from_args,
)


TRI_TRUE = "TRUE"
TRI_FALSE = "FALSE"
TRI_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PlacePredicateResult:
    state: str
    reason: str
    confidence: float = 0.0


def evaluate_place_predicate(
    facts: dict[str, Any],
    predicate: str,
    args: dict[str, Any],
) -> PlacePredicateResult:
    evaluator = _build_world_state_evaluator(facts)
    if evaluator is None:
        return PlacePredicateResult(TRI_UNKNOWN, "mc_world_state predicate evaluator unavailable")
    normalised_args = _normalise_place_args(predicate, args)
    result = evaluator.evaluate_facts(facts, predicate, normalised_args)
    return PlacePredicateResult(
        str(getattr(result, "state", TRI_UNKNOWN)),
        str(getattr(result, "reason", "")),
        float(getattr(result, "confidence", 0.0) or 0.0),
    )


def _build_world_state_evaluator(facts: dict[str, Any]):
    try:
        from mc_world_state.place_regions import PlaceRegionIndex
        from mc_world_state.predicate_evaluator import PredicateEvaluator
    except Exception:
        return None

    places = PlaceRegionIndex()
    for scope_name in ("places", "place_regions"):
        scoped = facts.get(scope_name)
        if not isinstance(scoped, dict):
            continue
        for key, entry in scoped.items():
            places.upsert_from_fact(str(key), fact_value(entry))
    return PredicateEvaluator(places)


def _normalise_place_args(predicate: str, args: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(args)
    place = str(args.get("place") or args.get("place_name") or "").strip()
    if place:
        normalised["place"] = place
    if predicate == "person_at_place":
        person_id = _normalise_person_id(person_id_from_args(args))
        if person_id:
            normalised["person_id"] = person_id
    elif predicate == "object_at_place":
        object_id = _normalise_object_id(
            str(args.get("object_id") or args.get("id") or "").strip()
            or object_name_from_args(args)
        )
        if object_id:
            normalised["object_id"] = object_id
    return normalised


def _normalise_person_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("person:"):
        return value
    return "person:" + normalise_entity_name(value)


def _normalise_object_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("object:"):
        return value
    return object_fact_key(value)
