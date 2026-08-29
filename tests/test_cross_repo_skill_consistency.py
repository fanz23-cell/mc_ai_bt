"""Dev/CI-time check that mc_ai_bt and seattle_lab agree on which skill names actually
cross the /mc_embodied_skills/execute action.

Not a production/deployment check -- the two repos ship as separate Docker images with no
shared runtime, so this can only run where both repos are checked out as siblings (true for
this workspace; see OMEGACLAW_AI_BT_INTEGRATION.md §9.1 for why that's judged sufficient:
the only point a new skill actually gets added is also the only point both repos are
present together).

Found live 2026-08-29, building approach_entity: mc_ai_bt's skill_adapters.py sent a goal
for a skill name seattle_lab's own EmbodiedSkillNode action server didn't recognize at all
("unsupported skill: approach_entity") -- passed mc_ai_bt's own test suite and policy
validation cleanly, because nothing on that side knew seattle_lab existed. This test exists
to catch exactly that class of bug before it reaches a live mission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SEATTLE_LAB = Path(__file__).resolve().parents[2] / "seattle_lab"

if not _SEATTLE_LAB.is_dir():
    pytest.skip(
        f"seattle_lab checkout not found at {_SEATTLE_LAB} -- this test only runs when "
        "mc_ai_bt and seattle_lab are checked out as sibling directories (dev machine, CI "
        "workspace checkout). Not an error in an isolated single-repo checkout.",
        allow_module_level=True,
    )

sys.path.insert(0, str(_SEATTLE_LAB))

# skill_adapters.py imports rclpy/mc_one (ROS message bindings) at module level for its
# other contents, even though EMBODIED_SKILLS itself needs neither -- same reason
# seattle_lab's own test_skill_adapters.py guards its import the same way.
pytest.importorskip("builtin_interfaces")
pytest.importorskip("mc_one")

from mc_ai_bt.skill_adapters import EMBODIED_SKILLS  # noqa: E402
from mc_embodied_skills.skills import ACCEPTED_SKILL_NAMES  # noqa: E402


def test_every_embodied_dispatch_skill_is_accepted_by_seattle_lab():
    """Every skill name mc_ai_bt actually sends to /mc_embodied_skills/execute must be
    accepted by seattle_lab's EmbodiedSkillNode._goal_callback, or the goal is rejected
    outright (STATUS_BLOCKED-equivalent: 'Goal was rejected', no route to any real
    implementation) -- exactly what happened live building approach_entity."""
    missing = EMBODIED_SKILLS - ACCEPTED_SKILL_NAMES
    assert not missing, (
        f"mc_ai_bt sends these skill names to /mc_embodied_skills/execute, but seattle_lab's "
        f"ACCEPTED_SKILL_NAMES (mc_embodied_skills/skills.py) does not recognize them, so "
        f"every goal for them is rejected before ever reaching an implementation: {sorted(missing)}. "
        f"Add each to seattle_lab's BASE_SKILLS/ARM_SKILLS/GAZE_SKILLS/OBSERVATION_SKILLS "
        f"(whichever category fits) so ACCEPTED_SKILL_NAMES picks it up."
    )
