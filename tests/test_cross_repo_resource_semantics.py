"""Dev/CI-time check that mc_ai_bt's PlanValidator and mc_resource_authority's real
lease arbiter agree on which resource names conflict.

Not a production/deployment check -- the two repos ship as separate Docker images with no
shared runtime, so this can only run where both repos are checked out as siblings (same
reasoning as test_cross_repo_skill_consistency.py; see OMEGACLAW_AI_BT_INTEGRATION.md §9.1).

Found live 2026-08-29 (§C5): validator.py's own Parallel-resource-conflict check did a plain
set intersection on raw resource strings, with no notion of mc_resource_authority's resource
*families* ("body" conflicts with left_arm/right_arm/gaze/head/torso). A Parallel plan pairing
"body" and "gaze" cleared PlanValidator and PolicyGuard with zero errors, and was only denied
later at the real resource lease (or, for skills that skip leasing entirely -- see
skill_adapters.py's _run_action -- silently raced at the ROS action-server level instead).
validator.py now keeps its own copy of the same conflict-family logic; this test is what
keeps that copy honest against the one real definition in mc_resource_authority.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_RESOURCE_AUTHORITY = Path(__file__).resolve().parents[2] / "mc_resource_authority"

if not _RESOURCE_AUTHORITY.is_dir():
    pytest.skip(
        f"mc_resource_authority checkout not found at {_RESOURCE_AUTHORITY} -- this test "
        "only runs when mc_ai_bt and mc_resource_authority are checked out as sibling "
        "directories (dev machine, CI workspace checkout). Not an error in an isolated "
        "single-repo checkout.",
        allow_module_level=True,
    )

sys.path.insert(0, str(_RESOURCE_AUTHORITY))

from mc_ai_bt.validator import resource_conflict_family, resources_conflict  # noqa: E402
from mc_resource_authority.authority import (  # noqa: E402
    resource_conflict_family as real_resource_conflict_family,
    resources_conflict as real_resources_conflict,
)

# Every resource name either side's own logic actually knows about -- the union, not just
# one side's list, so a resource only one copy has ever heard of still gets exercised.
_KNOWN_RESOURCES = (
    "base",
    "voice",
    "face",
    "body",
    "left_arm",
    "right_arm",
    "gaze",
    "head",
    "torso",
)


def test_conflict_family_agrees_for_every_known_resource():
    mismatches = {
        resource: (resource_conflict_family(resource), real_resource_conflict_family(resource))
        for resource in _KNOWN_RESOURCES
        if resource_conflict_family(resource) != real_resource_conflict_family(resource)
    }
    assert not mismatches, (
        f"mc_ai_bt/validator.py's resource_conflict_family disagrees with "
        f"mc_resource_authority's real definition for: {mismatches}. Keep validator.py's "
        f"copy (a plan-time schema check) in sync with authority.py's (the real runtime "
        f"arbiter) by hand whenever either changes."
    )


def test_resources_conflict_agrees_for_every_known_pair():
    mismatches = [
        (left, right)
        for left in _KNOWN_RESOURCES
        for right in _KNOWN_RESOURCES
        if resources_conflict(left, right) != real_resources_conflict(left, right)
    ]
    assert not mismatches, (
        f"mc_ai_bt/validator.py's resources_conflict disagrees with mc_resource_authority's "
        f"real definition for these (left, right) pairs: {mismatches}."
    )
