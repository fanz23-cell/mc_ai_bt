import sys
from types import ModuleType

import pytest

from mc_ai_bt.planner import BootstrapPlanner, LlmJsonPlanner
from mc_ai_bt.planner_factory import PlannerSettings, build_planner


def test_factory_defaults_to_bootstrap():
    planner = build_planner(PlannerSettings())

    assert isinstance(planner, BootstrapPlanner)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_planner(PlannerSettings(backend="mystery"))


def test_langchain_backend_requires_model_name():
    with pytest.raises(ValueError):
        build_planner(PlannerSettings(backend="langchain_openai"))


def test_langchain_backend_can_be_injected_with_fake_module(monkeypatch):
    fake_module = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            return "{}"

    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    planner = build_planner(
        PlannerSettings(
            backend="langchain_openai",
            model_name="test-model",
            temperature=0.2,
            timeout_sec=3.0,
        )
    )

    assert isinstance(planner, LlmJsonPlanner)
