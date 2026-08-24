from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    mission_id: str
    plan_version: int = 0
    execution_id: str = ""
    parent_mission_id: str = ""
    source: str = ""
    operator_id: str = ""

    @classmethod
    def new(
        cls,
        *,
        parent_mission_id: str = "",
        source: str = "",
        operator_id: str = "",
    ) -> "Identity":
        return cls(
            mission_id=uuid.uuid4().hex,
            plan_version=0,
            execution_id="",
            parent_mission_id=parent_mission_id,
            source=source,
            operator_id=operator_id,
        )

    def next_plan(self) -> "Identity":
        return Identity(
            mission_id=self.mission_id,
            plan_version=self.plan_version + 1,
            execution_id="",
            parent_mission_id=self.parent_mission_id,
            source=self.source,
            operator_id=self.operator_id,
        )

    def for_execution(self) -> "Identity":
        return Identity(
            mission_id=self.mission_id,
            plan_version=self.plan_version,
            execution_id=uuid.uuid4().hex,
            parent_mission_id=self.parent_mission_id,
            source=self.source,
            operator_id=self.operator_id,
        )
