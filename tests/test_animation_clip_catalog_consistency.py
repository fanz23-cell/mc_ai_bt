"""Dev/CI-time check that skill_registry.py's KNOWN_ANIMATION_CLIPS (the clip names the
play_animation skill's description advertises to the LLM planner) actually exist as real
clip files in the animation library.

Not a production/deployment check -- mc_ai_bt ships as its own Docker image with no shared
runtime with the animation library's repo, so this can only run where mc_one_codey is
checked out as a sibling (true for this workspace).

Found live 2026-08-31: play_animation's description named no clips at all, so the LLM
planner invented plausible-sounding names ("wave", "nod") that don't exist in the library
and silently fail to play -- confirmed by checking the real clip files, which have no
"wave" or "nod" but do have "wave_and_jaw". Curing that one pair by hand would leave the
same class of bug for the next invented name, so KNOWN_ANIMATION_CLIPS became the single
list both the planner-facing description and this test check against the real catalog on
disk -- exactly the "computed from one source" convention skill_registry.py already uses
for dispatch sets (see its own module docstring) and test_cross_repo_skill_consistency.py
follows for the seattle_lab boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
# check_clip.py (seattle_lab/test/check_clip.py) treats chest/ as canonical and diffs
# desktop/ against it -- same convention here, not a second opinion about which copy wins.
_CLIPS_DIR = _REPO_ROOT / "mc_one_codey" / "chest" / "context" / "animations" / "clips"

if not _CLIPS_DIR.is_dir():
    pytest.skip(
        f"animation clip catalog not found at {_CLIPS_DIR} -- this test only runs when "
        "mc_one_codey is checked out as a sibling directory (dev machine, CI workspace "
        "checkout). Not an error in an isolated single-repo checkout.",
        allow_module_level=True,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_ai_bt.skill_registry import KNOWN_ANIMATION_CLIPS, WAVE_CLIP  # noqa: E402


def _advertised_clip_names() -> set[str]:
    return {name for names in KNOWN_ANIMATION_CLIPS.values() for name in names}


def test_every_advertised_clip_exists_in_the_catalog():
    """Every clip name play_animation's description tells the planner about must have a
    real <name>.json in the animation library, or the planner is being told to use
    something that will silently fail to play -- exactly the "wave"/"nod" bug this list
    exists to prevent from recurring."""
    advertised = _advertised_clip_names()
    on_disk = {path.stem for path in _CLIPS_DIR.glob("*.json")}
    missing = advertised - on_disk
    assert not missing, (
        f"skill_registry.py's KNOWN_ANIMATION_CLIPS advertises these clip names to the "
        f"planner, but no matching file exists under {_CLIPS_DIR}: {sorted(missing)}. "
        f"Either the clip was renamed/removed (fix KNOWN_ANIMATION_CLIPS) or this is "
        f"exactly the invented-name bug the list exists to catch."
    )


def test_wave_clip_constant_is_advertised():
    """WAVE_CLIP is what planner.py's BootstrapPlanner fallback actually sends for
    'wave'/'挥手' -- it must be one of the names this module vouches for, not a stray
    string nobody checks."""
    assert WAVE_CLIP in _advertised_clip_names()
