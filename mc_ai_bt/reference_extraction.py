"""Deterministic ReferenceConstraint extraction -- runs BEFORE the planner.

Identity/Grounding foundation finalization (2026-09-03, GPT re-review): the
previous round let the SAME planner LLM that produces the rest of the plan
also author reference_constraints (entity_class/relation/reference_frame/
source_span) as its own self-certified "proof" that a spatial claim is
real. GPT's counter-examples showed this closes nothing structural: a
planner that mis-attributes "on your left" to a person when it actually
described a chair would still write internally-consistent fields
(entity_class="person", reference_frame="robot", source_span="on your
left" -- each individually a TRUE statement about the words) that pass
every deterministic check a guard can run, because the guard was checking
the model's own claims against each other and against the raw utterance,
never against an INDEPENDENT judgment of what those words actually modify.

The fix: extraction moves here, before the planner is ever invoked. This
does not solve general reference resolution -- it recognizes a handful of
tight, narrow-grammar sentence shapes (English + Chinese) where a
person-word and a robot-relative spatial phrase are directly, structurally
bound to each other, with no room for a different entity or a different
reference frame to have produced the same substring. Everything else
produces zero constraints -- a safe, honest "cannot disambiguate this way"
rather than a guess. planning_pipeline.py's guard then only lets an Action
REFERENCE a constraint this module already produced (by constraint_id, via
context_json.reference_constraints); the planner can no longer create,
modify, or embellish one -- it was never handed the pen.

Deliberately narrow, per explicit direction (GPT review): "宁可支持的语言
范围小，也不要让永久身份绑定建立在LLM自己签发的证明上" (better a smaller
supported language range than let a permanent identity binding rest on the
LLM's own self-issued proof). Extend the pattern lists below as real
missions surface a legitimate phrasing they miss -- never loosen the
adjacency requirements that keep a match from crossing into a different
entity or clause.
"""

from __future__ import annotations

import re
from typing import Any

# Person-word and relation-phrase must be directly, grammatically bound --
# no comma, no clause break, nothing that could let an unrelated entity's
# description ("the chair is on your left") or an unrelated instruction
# ("look to your left, then...") satisfy the same substring. Only "your"
# (addressing the robot) counts as reference_frame="robot" -- "my" would be
# the SPEAKER's own left/right, a different frame this system does not
# resolve, so it is deliberately never matched.
_PERSON_RELATION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "front": (
        re.compile(
            r"\b(?:the |a |this )?person\b(?:\s+(?:that|who|which)\s+is)?\s+"
            r"(?:right\s+)?in front of you\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:你|您)?正前方的人"),
        re.compile(r"(?:你|您)面前的人"),
    ),
    "left": (
        re.compile(
            r"\b(?:the |a |this )?person\b(?:\s+(?:that|who|which)\s+is)?\s+on your left\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:你|您)左边的人"),
    ),
    "right": (
        re.compile(
            r"\b(?:the |a |this )?person\b(?:\s+(?:that|who|which)\s+is)?\s+on your right\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:你|您)右边的人"),
    ),
    "nearest": (
        re.compile(r"\b(?:the )?(?:nearest|closest)\s+person\b", re.IGNORECASE),
        re.compile(r"离(?:你|您)最近的人"),
        re.compile(r"最近的人"),
    ),
}


# Identity/Grounding foundation finalization v2 (2026-09-03, GPT
# re-review): a constraint's OWN relation/frame/class can all be real and
# correctly verified, and it can STILL be borrowed by the wrong
# remember_person/remember_entity action -- e.g. "The person on your left
# is waving. Remember the person on your right as 44." extracts two real
# constraints (left, right); nothing before this fix stopped a confused
# planner from referencing the LEFT one for alias "44". _bind_alias_
# immediately_after captures which alias/name a constraint's own sentence
# assigns, WHEN it does so unambiguously -- planning_pipeline.py's guard
# then requires a NON-EMPTY bind_alias that EXACTLY matches the alias/name
# the referencing action actually uses. Both halves are unconditional: an
# empty bind_alias is rejected outright, and a captured one must match --
# neither check depends on how many constraints the mission produced (a v2
# round conditioned the match on "2+ constraints exist" and a v3 round
# still let an EMPTY bind_alias through when it was the only constraint;
# both exemptions turned out to be real holes, so v4 removed them -- see
# that guard's own comment).
#
# Two tiers of capture, both narrow and explicit:
#
# 1. _ALIAS_AFTER_PATTERNS: a tight "as <token>"/"叫<token>" marker
#    IMMEDIATELY after the relation phrase (only a bare comma allowed in
#    between, no clause break) -- "Remember the person on your left as
#    44." "is <token>" is deliberately NOT a marker, in either language
#    ("is waving" would otherwise be captured as alias="waving") -- only
#    "as"/"叫" are specific, low-ambiguity naming constructions in this
#    command-style context.
# 2. _CROSS_SENTENCE_ALIAS_PATTERNS: the original E.1 live-tested
#    phrasing -- "There is a person right in front of you. Remember them
#    as 44." -- names its subject via a PRONOUN in the immediately
#    FOLLOWING sentence, not the same clause. Recognized as its own
#    explicit, narrow pattern (a sentence break, then "Remember them/him/
#    her/it as <token>" verbatim, or the Chinese equivalent) -- not a
#    general "search anywhere after this point" rule.
#
# Missing a real alias assignment either way only means that constraint
# stays unusable for identity binding (see the guard) -- the safe
# direction, per the same "smaller supported range over any false accept"
# principle the relation patterns themselves already follow.
_ALIAS_AFTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s*,?\s*as\s+(?P<alias>[A-Za-z0-9_]+)", re.IGNORECASE),
    re.compile(r"\s*叫\s*(?P<alias>[\w一-鿿]+)"),
)
_ALIAS_LOOKAHEAD_WINDOW = 20

_CROSS_SENTENCE_ALIAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\s*[.!?]\s*Remember\s+(?:them|him|her|it)\s+as\s+(?P<alias>[A-Za-z0-9_]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\s*[。！？.]\s*(?:记住|把)(?:他们|他|她|它|TA)(?:叫|记成|记为|记作)\s*(?P<alias>[\w一-鿿]+)"),
)
_CROSS_SENTENCE_ALIAS_LOOKAHEAD_WINDOW = 40


def _bind_alias_immediately_after(text: str, end: int) -> str:
    window = text[end:end + _ALIAS_LOOKAHEAD_WINDOW]
    for pattern in _ALIAS_AFTER_PATTERNS:
        match = pattern.match(window)  # anchored at position 0 -- must be IMMEDIATELY adjacent
        if match:
            return match.group("alias")
    window = text[end:end + _CROSS_SENTENCE_ALIAS_LOOKAHEAD_WINDOW]
    for pattern in _CROSS_SENTENCE_ALIAS_PATTERNS:
        match = pattern.match(window)  # anchored -- must be the IMMEDIATELY next sentence
        if match:
            return match.group("alias")
    return ""


def extract_reference_constraints(intent_text: str) -> list[dict[str, Any]]:
    """Deterministic, trusted reference_constraints for `intent_text` -- see
    the module docstring for why this exists and how narrow it is by
    design. Never raises; returns [] for text with no recognized pattern.
    Each constraint's source_span is copied VERBATIM from the original (not
    lowercased) intent_text, exactly as matched -- a real quote, not a
    reconstruction. `bind_alias` (see _bind_alias_immediately_after above)
    is "" when no unambiguous naming marker was found. An extracted
    constraint is still a legal, real constraint in that case -- but
    planning_pipeline.py's guard will REJECT any attempt to use it for
    identity binding (remember_person/remember_entity), unconditionally,
    including when it is the only constraint this call produced. Only a
    non-empty bind_alias that exactly matches the referencing action's own
    alias/name is usable for binding."""
    text = intent_text or ""
    constraints: list[dict[str, Any]] = []
    seen_spans: list[tuple[int, int]] = []
    for relation, patterns in _PERSON_RELATION_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = match.span()
                if any(a < span[1] and span[0] < b for a, b in seen_spans):
                    continue  # do not double-count an overlapping match
                seen_spans.append(span)
                constraints.append({
                    "constraint_id": f"ref_{len(constraints) + 1}",
                    "entity_class": "person",
                    "relation": relation,
                    "reference_frame": "robot",
                    "source_span": text[span[0]:span[1]],
                    "bind_alias": _bind_alias_immediately_after(text, span[1]),
                })
    return constraints


# Mirrors planner.py's BootstrapPlanner Chinese look_at fragments and
# planning_pipeline.py's own (now-removed) _DIRECTION_TO_RELATION collapse
# (front_up/front_down -> front, etc.) so a look_at(direction="front_up")
# is checked against the same "front" instruction phrases a plain "front"
# direction would be.
_LOOK_DIRECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "left": (
        re.compile(r"\b(?:look|turn|glance)\b[^.!?]{0,10}\bleft\b", re.IGNORECASE),
        re.compile(r"看左边|往左看|向左看|往左转|向左转|左转"),
    ),
    "right": (
        re.compile(r"\b(?:look|turn|glance)\b[^.!?]{0,10}\bright\b", re.IGNORECASE),
        re.compile(r"看右边|往右看|向右看|往右转|向右转|右转"),
    ),
    "front": (
        re.compile(
            r"\b(?:look|turn|glance)\b[^.!?]{0,15}\b(?:ahead|forward|straight ahead|front)\b",
            re.IGNORECASE,
        ),
        re.compile(r"看前面|往前看|向前看"),
    ),
}

_DIRECTION_TO_LOOK_KEY = {
    "front": "front", "front_up": "front", "front_down": "front",
    "left": "left", "left_up": "left", "left_down": "left",
    "right": "right", "right_up": "right", "right_down": "right",
}


def has_explicit_look_instruction(intent_text: str, direction: str) -> bool:
    """True if intent_text contains a genuine look/turn instruction toward
    `direction` (a look_at Action's own direction arg) -- e.g. "look to
    your left", "往左转". Used to tell an explicit, user-requested physical
    action apart from planner-added redundant filler (see
    planning_pipeline.py's _terminal_self_locating_action): a look_at with
    real textual support here must never be silently dropped as
    "redundant" even though remember_person/remember_entity would locate
    internally anyway -- dropping it would mean the robot never performs
    the turn the user explicitly asked for."""
    key = _DIRECTION_TO_LOOK_KEY.get(str(direction or "").strip().lower())
    if key is None:
        return False
    text = intent_text or ""
    return any(pattern.search(text) for pattern in _LOOK_DIRECTION_PATTERNS[key])
