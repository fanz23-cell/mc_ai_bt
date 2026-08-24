from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .planner import BootstrapPlanner, LlmJsonPlanner, Planner
from .skill_registry import SkillRegistry


@dataclass(frozen=True)
class PlannerSettings:
    backend: str = "bootstrap"
    model_name: str = ""
    temperature: float = 0.0
    timeout_sec: float = 15.0


def build_planner(
    settings: PlannerSettings,
    *,
    skill_registry: SkillRegistry | None = None,
    logger: Any = None,
) -> Planner:
    backend = (settings.backend or "bootstrap").strip().lower()
    if backend == "bootstrap":
        _log(logger, "info", "AI-BT planner: BootstrapPlanner")
        return BootstrapPlanner()
    if backend == "langchain_openai":
        return _build_langchain_openai(settings, skill_registry=skill_registry, logger=logger)
    raise ValueError(f"unsupported planner backend: {settings.backend!r}")


def _build_langchain_openai(
    settings: PlannerSettings,
    *,
    skill_registry: SkillRegistry | None,
    logger: Any,
) -> Planner:
    if not settings.model_name.strip():
        raise ValueError("planner_model_name is required for langchain_openai backend")
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "langchain_openai planner backend requires langchain_openai to be installed"
        ) from exc

    model = ChatOpenAI(
        model=settings.model_name,
        temperature=float(settings.temperature),
        timeout=float(settings.timeout_sec),
    )
    _log(logger, "info", f"AI-BT planner: LlmJsonPlanner(langchain_openai model={settings.model_name})")
    return LlmJsonPlanner(model, skill_registry=skill_registry)


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
