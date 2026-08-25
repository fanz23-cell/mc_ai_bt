from __future__ import annotations

import re
from typing import Any


PERSON_WORDS = {"person", "people", "someone", "anyone", "人", "某人", "一个人"}


def visual_check_goal_spec(check: dict[str, Any]) -> dict[str, Any]:
    if check.get("type") == "visual":
        query = str(check.get("query") or "").strip()
        parsed = visual_query_goal_spec(query)
        if parsed:
            verification = check.get("verification")
            if isinstance(verification, dict) and verification:
                parsed["verification"] = verification
            extra_args = check.get("args")
            if isinstance(extra_args, dict) and extra_args:
                args = dict(parsed.get("args") or {})
                args.update(extra_args)
                parsed["args"] = args
            parsed["summary"] = check.get("summary") or query
            return parsed
        return dict(check)
    if "type" in check:
        return dict(check)
    if isinstance(check.get("predicate"), str) and check["predicate"].strip():
        return {
            "type": "structured",
            "predicate": check.get("predicate"),
            "args": check.get("args") or {},
            "verification": check.get("verification") or {"mode": "world_state_or_visual_check"},
            "summary": check.get("summary", ""),
        }

    query = str(check.get("query") or "").strip()
    if not query:
        return {}

    parsed = visual_query_goal_spec(query)
    if not parsed:
        return {}
    extra_args = check.get("args")
    if isinstance(extra_args, dict) and extra_args:
        args = dict(parsed.get("args") or {})
        args.update(extra_args)
        parsed["args"] = args
    return parsed


def visual_query_goal_spec(query: str) -> dict[str, Any]:
    text = _normalise(query)
    if _looks_like_people_query(text):
        return {
            "type": "structured",
            "predicate": "person_visible",
            "args": {"min_count": 1},
            "verification": {"mode": "world_state_or_visual_check"},
            "summary": query,
        }

    object_name = _extract_object_name(text)
    if object_name:
        return {
            "type": "structured",
            "predicate": "object_visible",
            "args": {"name": object_name},
            "verification": {"mode": "world_state_or_visual_check"},
            "summary": query,
        }
    return {}


def _normalise(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[?!.,;:？！。，；：]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _looks_like_people_query(text: str) -> bool:
    phrases = (
        "do you see anyone",
        "can you see anyone",
        "do you see a person",
        "can you see a person",
        "do you see people",
        "can you see people",
        "how many people",
        "visible people",
        "anyone here",
        "有人吗",
        "有没有人",
        "看到人",
        "几个人",
    )
    return any(phrase in text for phrase in phrases)


def _extract_object_name(text: str) -> str:
    patterns = (
        r"\bdo you see (?:the )?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
        r"\bcan you see (?:the )?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
        r"\bis (?:the )?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+) visible$",
        r"\bis (?:the )?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+) here$",
        r"\bwhere is (?:the )?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+)$",
        r"(?:你)?(?:看到|看见)(?:了)?(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+?)(?:了吗|吗)?$",
        r"有没有(?P<object>[\u4e00-\u9fffA-Za-z0-9 _-]+?)(?:吗)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_object_name(match.group("object"))
    return ""


def _clean_object_name(raw: str) -> str:
    cleaned = re.sub(r"\b(right now|nearby|around|please)\b", " ", raw)
    cleaned = re.sub(r"(这个|那个|这里|那里|附近|周围|吗|了)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" _-")
    if cleaned in PERSON_WORDS:
        return "person"
    return cleaned
