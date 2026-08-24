from __future__ import annotations

import uuid

from mc_one.msg import AiBtIdentity

from .identity import Identity


def identity_to_msg(identity: Identity) -> AiBtIdentity:
    msg = AiBtIdentity()
    msg.mission_id = identity.mission_id
    msg.plan_version = identity.plan_version
    msg.execution_id = identity.execution_id
    msg.parent_mission_id = identity.parent_mission_id
    msg.source = identity.source
    msg.operator_id = identity.operator_id
    return msg


def empty_identity_msg(*, source: str = "mc_ai_bt") -> AiBtIdentity:
    msg = AiBtIdentity()
    msg.execution_id = uuid.uuid4().hex
    msg.source = source
    return msg


def copy_identity_msg(identity: AiBtIdentity | None, *, source: str = "mc_ai_bt") -> AiBtIdentity:
    if identity is None:
        return empty_identity_msg(source=source)
    msg = AiBtIdentity()
    msg.mission_id = str(getattr(identity, "mission_id", "") or "")
    msg.plan_version = int(getattr(identity, "plan_version", 0) or 0)
    msg.execution_id = str(getattr(identity, "execution_id", "") or "")
    msg.parent_mission_id = str(getattr(identity, "parent_mission_id", "") or "")
    msg.source = str(getattr(identity, "source", "") or source)
    msg.operator_id = str(getattr(identity, "operator_id", "") or "")
    return msg
