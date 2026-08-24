from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from .identity import Identity
from .mission import (
    Mission,
    MissionEvent,
    STATE_BLOCKED,
    STATE_CANCELED,
    STATE_FAILED,
    STATE_SUCCEEDED,
)


SCHEMA = "mc_ai_bt.mission_journal.v1"


class MissionJournal:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: MissionEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": SCHEMA,
            "event": event.event,
            "message": event.message,
            "payload_json": event.payload_json,
            "mission": _mission_dict(event.mission),
        }
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    def load_latest_missions(self, *, recover_nonterminal: bool = True) -> tuple[Mission, ...]:
        if not self._path.exists():
            return ()
        latest: dict[str, Mission] = {}
        with self._path.open("r", encoding="utf-8") as stream:
            for line in stream:
                mission = _mission_from_record(line)
                if mission is None:
                    continue
                latest[mission.identity.mission_id] = mission
        missions = tuple(latest.values())
        if not recover_nonterminal:
            return missions
        return tuple(_recover_mission(mission) for mission in missions)


def _mission_dict(mission: Mission) -> dict[str, Any]:
    value = asdict(mission)
    value["identity"] = asdict(mission.identity)
    return value


def _mission_from_record(raw: str) -> Mission | None:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    mission_value = record.get("mission")
    if not isinstance(mission_value, dict):
        return None
    return _mission_from_dict(mission_value)


def _mission_from_dict(value: dict[str, Any]) -> Mission | None:
    identity_value = value.get("identity")
    if not isinstance(identity_value, dict):
        return None
    try:
        identity = Identity(**_filter_fields(identity_value, Identity))
        mission_kwargs = _filter_fields(value, Mission)
        mission_kwargs["identity"] = identity
        return Mission(**mission_kwargs)
    except (TypeError, ValueError):
        return None


def _filter_fields(value: dict[str, Any], cls) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: item for key, item in value.items() if key in names}


def _recover_mission(mission: Mission) -> Mission:
    if mission.state in {STATE_CANCELED, STATE_FAILED, STATE_SUCCEEDED, STATE_BLOCKED}:
        return mission
    return replace(
        mission,
        identity=replace(mission.identity, execution_id=""),
        state=STATE_BLOCKED,
        active_node="",
        status_text="recovered after node restart; manual review required",
        error_code="recovered_after_restart",
    )
