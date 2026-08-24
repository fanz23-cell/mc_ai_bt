from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    resources: tuple[str, ...]
    description: str


DEFAULT_SKILLS: dict[str, SkillSpec] = {
    "say": SkillSpec("say", ("voice", "face"), "Speak through the legacy TTS path."),
    "go_to_place": SkillSpec("go_to_place", ("base",), "Navigate to a named place."),
    "come_to_me": SkillSpec("come_to_me", ("base",), "Navigate to the tracked person."),
    "simple_move": SkillSpec("simple_move", ("base",), "Relative base movement."),
    "play_animation": SkillSpec("play_animation", ("body",), "Play an animator clip."),
    "look_at": SkillSpec("look_at", ("body",), "Look/focus using animator generator."),
    "point_at": SkillSpec("point_at", ("body",), "Point at a target using animator generator."),
}


class SkillRegistry:
    def __init__(self, skills: dict[str, SkillSpec] | None = None) -> None:
        self._skills = dict(skills or DEFAULT_SKILLS)

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills.keys()))
