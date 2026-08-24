import json
from types import SimpleNamespace

from mc_ai_bt.planner import LlmJsonPlanner, build_planner_messages


class FakeModel:
    def __init__(self, content: str):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_llm_json_planner_extracts_json_from_model_response():
    model = FakeModel(
        """
        ```json
        {"root":{"type":"Action","skill":"say","args":{"text":"hi"}},
         "goal_spec":{"type":"human","verification":{"mode":"implicit_conversation"}}}
        ```
        """
    )

    plan = json.loads(LlmJsonPlanner(model).plan("say hi", '{"world":{}}'))

    assert plan["schema"] == "mc_ai_bt.plan.v1"
    assert plan["root"]["skill"] == "say"
    assert plan["context_json"] == '{"world":{}}'
    assert model.messages[0][0] == "system"


def test_planner_prompt_names_allowed_skills_and_constraints():
    messages = build_planner_messages("go to kitchen", "{}")
    system = messages[0][1]

    assert "mc_ai_bt.plan.v1" in system
    assert "go_to_place" in system
    assert "point_at:" in system
    assert "look_at may use only direction" in system
    assert "Do not invent ROS topics" in system
    assert "UNKNOWN" in system
