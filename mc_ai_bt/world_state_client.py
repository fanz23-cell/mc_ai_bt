from __future__ import annotations

import json
import threading
from typing import Any

from mc_one.srv import GetWorldSnapshot
from rclpy.node import Node

from .world_facts import WorldFactUpdate


class WorldStateClient:
    def __init__(
        self,
        node: Node,
        *,
        service_name: str = "/mc_world_state/get_snapshot",
        callback_group=None,
    ) -> None:
        self._client = node.create_client(
            GetWorldSnapshot,
            service_name,
            callback_group=callback_group,
        )

    def snapshot(self, scopes: tuple[str, ...], max_age_sec: float):
        if not self._client.service_is_ready():
            return None
        request = GetWorldSnapshot.Request()
        request.scopes = list(scopes)
        request.max_age_sec = max_age_sec
        request.include_private = False
        request.query_json = "{}"
        ok, response_or_message = _wait_future(self._client.call_async(request), timeout_sec=2.0)
        if not ok:
            return None
        response = response_or_message
        if not bool(getattr(response, "success", False)):
            return None
        return response.snapshot

    def snapshot_json(self, scopes: tuple[str, ...], max_age_sec: float) -> str:
        snapshot = self.snapshot(scopes, max_age_sec)
        if snapshot is None:
            return ""
        return str(getattr(snapshot, "world_json", "") or "")


class ReferenceResolverClient:
    """B (MissionGoalContract V1, GPT-approved): the planning-time CALLER of
    the existing, unmodified /mc_world_state/resolve_entity_reference
    service -- the same authoritative service remember_person/remember_entity
    already call at execution time (seattle_lab/mc_embodied_skills/node.py),
    just invoked one stage earlier so a spatial semantic target
    (entity_class/relation/reference_frame) can be materialized into a real
    entity_id + grounding_ref BEFORE synthesizing a producer Action, instead
    of only ever happening inside a single skill's own execution. This adds
    no new grounding capability -- WorldState remains the sole authority on
    what "nearest"/"left"/"right" resolves to; this is one more client of a
    service that already exists."""

    def __init__(
        self,
        node: Node,
        *,
        service_name: str = "/mc_world_state/resolve_entity_reference",
        callback_group=None,
    ) -> None:
        from mc_one.srv import ResolveEntityReference

        self._service_type = ResolveEntityReference
        self._client = node.create_client(
            ResolveEntityReference,
            service_name,
            callback_group=callback_group,
        )

    def resolve(
        self,
        *,
        entity_class: str,
        relation: str,
        reference_frame: str,
        max_age_sec: float = 0.0,
        timeout_sec: float = 2.0,
    ) -> tuple[str, str, str]:
        """Returns (state, live_entity_id, grounding_ref). state is one of
        RESOLVED/AMBIGUOUS/NOT_FOUND/UNKNOWN (the service's own vocabulary)
        -- or "UNKNOWN" if the service is not ready / the call times out,
        never fabricated as RESOLVED. live_entity_id/grounding_ref are only
        ever non-empty when state == RESOLVED."""
        if not self._client.service_is_ready():
            return "UNKNOWN", "", ""
        request = self._service_type.Request()
        request.entity_class = entity_class
        request.relation = relation
        request.reference_frame = reference_frame
        request.max_age_sec = max_age_sec
        ok, response_or_message = _wait_future(
            self._client.call_async(request),
            timeout_sec=timeout_sec,
        )
        if not ok:
            return "UNKNOWN", "", ""
        response = response_or_message
        if not bool(getattr(response, "success", False)):
            return "UNKNOWN", "", ""
        state = str(getattr(response, "state", "") or "UNKNOWN")
        if state != "RESOLVED":
            return state, "", ""
        return state, str(getattr(response, "live_entity_id", "") or ""), str(
            getattr(response, "grounding_ref", "") or "")


class WorldStateWriter:
    def __init__(
        self,
        node: Node,
        *,
        service_name: str = "/mc_world_state/update_facts",
        bind_entity_alias_service_name: str = "/mc_world_state/bind_entity_alias",
        callback_group=None,
    ) -> None:
        from mc_one.srv import BindEntityAlias, UpdateWorldFacts

        self._service_type = UpdateWorldFacts
        self._client = node.create_client(
            UpdateWorldFacts,
            service_name,
            callback_group=callback_group,
        )
        self._bind_entity_alias_type = BindEntityAlias
        self._bind_entity_alias_client = node.create_client(
            BindEntityAlias,
            bind_entity_alias_service_name,
            callback_group=callback_group,
        )

    def update(self, update: WorldFactUpdate, *, timeout_sec: float = 0.5) -> tuple[bool, str]:
        return self.update_fact(
            source=update.source,
            scope=update.scope,
            key=update.key,
            value=update.value,
            merge=update.merge,
            timeout_sec=timeout_sec,
        )

    def update_fact(
        self,
        *,
        source: str,
        scope: str,
        key: str,
        value: Any,
        merge: bool = False,
        timeout_sec: float = 0.5,
    ) -> tuple[bool, str]:
        if not self._client.service_is_ready():
            return False, "world_state update service is not ready"

        request = self._service_type.Request()
        request.source = source
        request.scope = scope
        request.key = key
        request.value_json = json.dumps(value, sort_keys=True, separators=(",", ":"))
        request.merge = merge
        request.include_snapshot = False
        ok, response_or_message = _wait_future(
            self._client.call_async(request),
            timeout_sec=timeout_sec,
        )
        if not ok:
            return False, str(response_or_message)
        response = response_or_message
        if not bool(getattr(response, "success", False)):
            return False, str(getattr(response, "message", "") or "world_state update rejected")
        return True, str(getattr(response, "message", "") or "ok")

    def bind_entity_alias(
        self,
        *,
        alias: str,
        entity_class: str,
        live_entity_id: str,
        created_by: str,
        evidence_ref: str = "",
        timeout_sec: float = 0.5,
    ) -> tuple[bool, str]:
        """2026-09-03 architecture consolidation: the domain-command
        counterpart to update_fact/update -- see world_facts.py's own
        comment on why entity_alias_bound is no longer a generic
        WorldFactUpdate. Calls /mc_world_state/bind_entity_alias directly."""
        if not self._bind_entity_alias_client.service_is_ready():
            return False, "bind_entity_alias service is not ready"

        request = self._bind_entity_alias_type.Request()
        request.alias = alias
        request.entity_class = entity_class
        request.live_entity_id = live_entity_id
        request.created_by = created_by
        request.evidence_ref = evidence_ref
        ok, response_or_message = _wait_future(
            self._bind_entity_alias_client.call_async(request),
            timeout_sec=timeout_sec,
        )
        if not ok:
            return False, str(response_or_message)
        response = response_or_message
        if not bool(getattr(response, "success", False)):
            return False, str(getattr(response, "message", "") or "bind_entity_alias rejected")
        return True, str(getattr(response, "message", "") or "ok")


def _wait_future(future, *, timeout_sec: float):
    done = threading.Event()
    box = {}

    def _done(fut):
        try:
            box["result"] = fut.result()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    future.add_done_callback(_done)
    if not done.wait(timeout=max(0.0, timeout_sec)):
        return False, "future timed out"
    if "error" in box:
        exc = box["error"]
        return False, f"{type(exc).__name__}: {exc}"
    return True, box.get("result")
