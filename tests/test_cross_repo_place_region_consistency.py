"""Dev/CI-time check that mc_ai_bt's place_region_predicates.py (a mirror, since mc_ai_bt
and mc_world_state are separate Docker images with no shared runtime -- same reasoning as
test_cross_repo_skill_consistency.py and test_cross_repo_resource_semantics.py; see
OMEGACLAW_AI_BT_INTEGRATION.md §9.1) produces identical results to mc_world_state's real
PlaceRegionIndex/PredicateEvaluator for the same inputs.

Found live 2026-08-31: place_predicates.py's _build_world_state_evaluator used to
`from mc_world_state.place_regions import PlaceRegionIndex` directly inside a bare
try/except that silently returned None on any ImportError -- which is EVERY call in
production, since these are separate images. person_at_place/object_at_place could never
resolve to TRUE/FALSE, only ever UNKNOWN, no matter what mc_world_state actually knew.
place_region_predicates.py replaces that with a real, working local copy; this test is
what keeps that copy honest against the one real definition in mc_world_state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORLD_STATE = Path(__file__).resolve().parents[2] / "mc_world_state"

if not _WORLD_STATE.is_dir():
    pytest.skip(
        f"mc_world_state checkout not found at {_WORLD_STATE} -- this test only runs when "
        "mc_ai_bt and mc_world_state are checked out as sibling directories (dev machine, "
        "CI workspace checkout). Not an error in an isolated single-repo checkout.",
        allow_module_level=True,
    )

sys.path.insert(0, str(_WORLD_STATE))

from mc_ai_bt.place_region_predicates import (  # noqa: E402
    PlaceRegionIndex as MirrorIndex,
    PredicateEvaluator as MirrorEvaluator,
    place_region_from_value as mirror_place_region_from_value,
    position_from_value as mirror_position_from_value,
    visible_from_value as mirror_visible_from_value,
)
from mc_world_state.place_regions import (  # noqa: E402
    PlaceRegionIndex as RealIndex,
    place_region_from_value as real_place_region_from_value,
)
from mc_world_state.predicate_evaluator import PredicateEvaluator as RealEvaluator  # noqa: E402
from mc_world_state.schemas import (  # noqa: E402
    position_from_value as real_position_from_value,
    visible_from_value as real_visible_from_value,
)

_KITCHEN = {
    "name": "kitchen",
    "position": {"x": 3.0, "y": -1.5, "z": 0.0, "frame": "map"},
    "occupancy_radius_m": 1.5,
}
_LIVING_ROOM = {
    "name": "living_room",
    "nav_pose": {"x": -2.0, "y": 4.0, "z": 0.0},
    "frame_id": "map",
}
_PLACE_VALUES = {"kitchen": _KITCHEN, "living_room": _LIVING_ROOM, "malformed": {"no_position_here": True}}

_ENTITY_VALUES = [
    {"visible": True, "position": {"x": 3.2, "y": -1.4, "z": 0.0, "frame": "map"}},
    {"visible": True, "position": {"x": 10.0, "y": 10.0, "z": 0.0, "frame": "map"}},
    {"visible": False, "position": {"x": 3.0, "y": -1.5, "z": 0.0, "frame": "map"}},
    {"coverage": "seen_in_camera", "position": {"x": -2.1, "y": 3.9, "z": 0.0, "frame": "map"}},
    {"position": {"x": 3.0, "y": -1.5, "z": 0.0}},  # no visible/coverage key at all
    {},
]


def test_place_region_from_value_agrees_for_every_known_place():
    for name, value in _PLACE_VALUES.items():
        mirror = mirror_place_region_from_value(name, value)
        real = real_place_region_from_value(name, value)
        assert (mirror is None) == (real is None), f"presence mismatch for {name}"
        if mirror is not None and real is not None:
            assert (mirror.name, mirror.frame_id, mirror.x, mirror.y, mirror.z, mirror.occupancy_radius_m) == (
                real.name, real.frame_id, real.x, real.y, real.z, real.occupancy_radius_m
            ), f"PlaceRegion fields disagree for {name}"


def test_visible_and_position_from_value_agree_for_every_entity_shape():
    for entity in _ENTITY_VALUES:
        assert mirror_visible_from_value(entity) == real_visible_from_value(entity), f"visible_from_value disagrees for {entity}"
        assert mirror_position_from_value(entity) == real_position_from_value(entity), f"position_from_value disagrees for {entity}"


@pytest.mark.parametrize("predicate,scope,entity_key", [
    ("person_at_place", "people", "person_id"),
    ("object_at_place", "objects", "object_id"),
])
def test_predicate_evaluator_agrees_for_every_place_and_entity_combination(predicate, scope, entity_key):
    for place_name in ("kitchen", "living_room", "nowhere"):
        for entity in _ENTITY_VALUES:
            facts = {
                "places": {"kitchen": _KITCHEN, "living_room": _LIVING_ROOM},
                scope: {"target": entity},
            }
            args = {"place": place_name, entity_key: "target"}

            mirror_places = MirrorIndex()
            for key, value in facts["places"].items():
                mirror_places.upsert_from_fact(key, value)
            mirror_result = MirrorEvaluator(mirror_places).evaluate_facts(facts, predicate, args)

            real_places = RealIndex()
            for key, value in facts["places"].items():
                real_places.upsert_from_fact(key, value)
            real_result = RealEvaluator(real_places).evaluate_facts(facts, predicate, args)

            assert mirror_result.state == real_result.state, (
                f"{predicate}(place={place_name}, entity={entity}): "
                f"mirror={mirror_result.state} real={real_result.state}"
            )
