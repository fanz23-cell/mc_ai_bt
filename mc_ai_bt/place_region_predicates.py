from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .entity_facts import fact_value

# Mirrors mc_world_state/place_regions.py, mc_world_state/predicate_evaluator.py's
# person_at_place/object_at_place logic, and the small tri-state/position helpers from
# mc_world_state/schemas.py. mc_ai_bt and mc_world_state are separate Docker images with
# no shared runtime (see OMEGACLAW_AI_BT_INTEGRATION.md), so this cannot be a real import
# -- kept in sync by tests/test_cross_repo_place_region_consistency.py, which runs both
# copies through the same cases and fails on any divergence whenever mc_world_state is
# checked out as a sibling. Same pattern test_cross_repo_skill_consistency.py and
# test_cross_repo_resource_semantics.py already use for this exact cross-image problem.
#
# FOUND LIVE 2026-08-31: place_predicates.py used to do
# `from mc_world_state.place_regions import PlaceRegionIndex` directly, inside a bare
# try/except that silently returned None on any ImportError -- which is EVERY call in
# production, since these are separate images. person_at_place/object_at_place could
# never resolve to TRUE/FALSE, only ever UNKNOWN. This module is also what go_to_place
# and come_to_me's real (non-circular) success verification is built on -- see
# skill_adapters.py's _verify_robot_in_place/_verify_robot_near_person.

TRI_UNKNOWN = "UNKNOWN"
TRI_TRUE = "TRUE"
TRI_FALSE = "FALSE"


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def tri_state(value: Any) -> tuple[str, float]:
    if isinstance(value, dict):
        state = str(value.get("state") or "").strip().upper()
        confidence = _confidence(value.get("confidence", 0.0))
        if state in {TRI_TRUE, TRI_FALSE, TRI_UNKNOWN}:
            return state, confidence
    if isinstance(value, bool):
        return (TRI_TRUE if value else TRI_FALSE), 1.0
    text = str(value or "").strip().upper()
    if text in {TRI_TRUE, TRI_FALSE, TRI_UNKNOWN}:
        return text, 1.0
    return TRI_UNKNOWN, 0.0


def visible_from_value(value: Any) -> tuple[str, float]:
    if isinstance(value, dict):
        if "visible" in value:
            return tri_state(value.get("visible"))
        coverage = str(value.get("coverage") or "").strip().lower()
        if coverage in {"seen_in_camera", "visible"}:
            return TRI_TRUE, _confidence(value.get("confidence", 1.0))
        if coverage in {"outside_fov", "occluded"}:
            return TRI_UNKNOWN, _confidence(value.get("confidence", 0.0))
    if value is True:
        return TRI_TRUE, 1.0
    if value is False:
        return TRI_FALSE, 1.0
    return TRI_UNKNOWN, 0.0


def position_from_value(value: Any) -> tuple[str, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    pos = value.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        return (
            str(pos.get("frame") or value.get("frame_id") or "map"),
            float(pos.get("x")),
            float(pos.get("y")),
            float(pos.get("z", 0.0)),
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PlaceRegion:
    name: str
    frame_id: str
    x: float
    y: float
    z: float = 0.0
    occupancy_radius_m: float = 2.0

    def contains(self, position: tuple[str, float, float, float]) -> bool:
        frame, x, y, _z = position
        if frame and self.frame_id and frame != self.frame_id:
            return False
        return math.hypot(x - self.x, y - self.y) <= self.occupancy_radius_m


class PlaceRegionIndex:
    def __init__(self, regions: list[PlaceRegion] | None = None) -> None:
        self._regions = {region.name: region for region in (regions or [])}

    def upsert_from_fact(self, name: str, value: Any) -> None:
        region = place_region_from_value(name, value)
        if region is not None:
            self._regions[region.name] = region

    def get(self, name: str) -> PlaceRegion | None:
        return self._regions.get(name)


def place_region_from_value(name: str, value: Any) -> PlaceRegion | None:
    if not isinstance(value, dict):
        return None
    region_name = str(value.get("name") or name).strip()
    if not region_name:
        return None
    nav_pose = value.get("nav_pose") if isinstance(value.get("nav_pose"), dict) else {}
    position = value.get("position") if isinstance(value.get("position"), dict) else {}
    source = nav_pose or position or value
    try:
        x = float(source.get("x"))
        y = float(source.get("y"))
        z = float(source.get("z", 0.0))
    except (TypeError, ValueError):
        return None
    try:
        radius = float(value.get("occupancy_radius_m", 2.0) or 2.0)
    except (TypeError, ValueError):
        radius = 2.0
    return PlaceRegion(
        name=region_name,
        frame_id=str(value.get("frame_id") or source.get("frame") or "map"),
        x=x,
        y=y,
        z=z,
        occupancy_radius_m=max(0.1, radius),
    )


@dataclass(frozen=True)
class PredicateResult:
    state: str
    reason: str
    confidence: float = 0.0


class PredicateEvaluator:
    def __init__(self, places: PlaceRegionIndex | None = None) -> None:
        self._places = places or PlaceRegionIndex()

    def evaluate_facts(self, facts: dict[str, Any], predicate: str, args: dict[str, Any]) -> PredicateResult:
        if predicate == "person_at_place":
            return self._entity_at_place(facts.get("people"), args, entity_key="person_id")
        if predicate == "object_at_place":
            return self._entity_at_place(facts.get("objects"), args, entity_key="object_id")
        return PredicateResult(TRI_UNKNOWN, f"unsupported predicate: {predicate}")

    def _entity_at_place(self, scoped: Any, args: dict[str, Any], *, entity_key: str) -> PredicateResult:
        if not isinstance(scoped, dict):
            return PredicateResult(TRI_UNKNOWN, "entity scope missing")
        place_name = str(args.get("place") or args.get("place_name") or args.get("name") or "").strip()
        entity_id = str(args.get(entity_key) or args.get("id") or "").strip()
        region = self._places.get(place_name)
        if region is None:
            return PredicateResult(TRI_UNKNOWN, f"place region not found: {place_name}")
        candidate = _find_entity(scoped, entity_id)
        if candidate is None:
            return PredicateResult(TRI_UNKNOWN, f"entity not found: {entity_id or '*'}")
        value = candidate.get("value") if isinstance(candidate, dict) and "value" in candidate else candidate
        visible, confidence = visible_from_value(value)
        if visible == TRI_FALSE:
            return PredicateResult(TRI_UNKNOWN, "entity is not currently visible", confidence)
        position = position_from_value(value)
        if position is None:
            return PredicateResult(TRI_UNKNOWN, "entity position missing", confidence)
        if region.contains(position):
            return PredicateResult(TRI_TRUE, f"inside place region: {place_name}", max(confidence, 0.5))
        return PredicateResult(TRI_FALSE, f"outside place region: {place_name}", max(confidence, 0.5))


def _loose_name(raw: Any) -> str:
    # Mirrors normalise_entity_name (underscores/hyphens == spaces, case-insensitive) --
    # the two sides of an object/person lookup have used different raw spellings
    # ("test_object" stored vs "test object" queried, via this same normalisation
    # elsewhere) since before this fix; matching loosely is what makes them agree at all.
    text = str(raw or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_entity(scoped: dict[str, Any], entity_id: str) -> Any:
    if not entity_id:
        return None
    for key in (entity_id, entity_id.removeprefix("person:"), entity_id.removeprefix("object:")):
        if key in scoped:
            return scoped[key]
    # FOUND LIVE 2026-08-31: mc_world_state's objects scope no longer keys purely by
    # class name (node.py's _on_object_localization now keys by class+position cell, to
    # keep multiple simultaneously-visible instances distinct) -- an exact-key miss must
    # not fall back to "the first entry in the dict, whatever it is": that used to
    # silently answer object_at_place about an arbitrary, possibly wrong, object. Match
    # each entry's own identifying field instead, same pattern entity_facts.py and
    # semantic_verifier.py already use for exactly this reason.
    wanted = _loose_name(entity_id.removeprefix("person:").removeprefix("object:"))
    for entry in scoped.values():
        value = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
        if not isinstance(value, dict):
            continue
        for field in ("object_name", "object_id", "person_id", "name", "id", "track_id"):
            if field not in value:
                continue
            candidate = str(value.get(field) or "")
            candidate = candidate.removeprefix("object:").removeprefix("person:")
            if _loose_name(candidate) == wanted:
                return entry
    return None


# --- mc_ai_bt-side glue, NOT mirrored from mc_world_state (nothing to keep in sync here:
# it only composes the mirrored primitives above) -----------------------------------------

def place_contains_pose(
    places: PlaceRegionIndex, place_name: str, position: tuple[str, float, float, float]
) -> PredicateResult:
    """Is a directly-measured pose inside the named place's region? Used by go_to_place's
    real success verification (a fresh robot pose, not a self-reported claim)."""
    region = places.get(place_name)
    if region is None:
        return PredicateResult(TRI_UNKNOWN, f"place region not found: {place_name}")
    if region.contains(position):
        return PredicateResult(TRI_TRUE, f"inside place region: {place_name}", 1.0)
    return PredicateResult(TRI_FALSE, f"outside place region: {place_name}", 1.0)


def place_regions_from_facts(facts: dict[str, Any]) -> PlaceRegionIndex:
    places = PlaceRegionIndex()
    for scope_name in ("places", "place_regions"):
        scoped = facts.get(scope_name)
        if not isinstance(scoped, dict):
            continue
        for key, entry in scoped.items():
            places.upsert_from_fact(str(key), fact_value(entry))
    return places
