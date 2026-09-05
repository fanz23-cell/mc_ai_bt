"""FOUND LIVE 2026-09-05 (E1 TURN2, three failures in a row).

Omega composes its own intent_text rather than passing the user's words
through. The same six-character request, "去33那里。", arrived as an English
rewrite one run and as "去离我最近、我之前记住的那个叫33的..." another, and each
rewrite produced a differently-shaped plan: an extra locate step one time, an
extra precondition gate the next. Asking the model not to paraphrase fixed the
naming phrase and did not generalise -- three runs, three shapes.

So provenance stops depending on the model's cooperation. The Bridge is the
same process that handed the utterance to Omega, so it carries the user's real
words in the machine-owned context, and the deterministic reference logic reads
those instead of the rewrite. submit_mission still takes one natural-language
argument (FROZEN #14) and the context is still machine-built (#15).

These tests pin the part that lives in this repo.
"""

import json

from mc_ai_bt.context_builder import _reference_source_text
from mc_ai_bt.mission import Mission
from mc_ai_bt.planning_pipeline import _reference_source_text as _pipeline_source_text
from mc_ai_bt.reference_extraction import extract_reference_constraints


class _M:
    def __init__(self, intent_text):
        self.intent_text = intent_text


def test_the_users_own_words_win_over_omegas_rewrite():
    caller = {"user_utterance": "记住离你最近的人叫33。"}
    mission = _M("Remember the person closest to you and label them as 33")
    assert _reference_source_text(mission, caller) == "记住离你最近的人叫33。"


def test_the_submitted_text_is_used_when_the_bridge_carried_nothing():
    # Omega-initiated missions (a reminder coming due, a ROBOT_EVENT it acted
    # on) have no user utterance behind them, and the Bridge deliberately
    # sends nothing rather than a stale one.
    mission = _M("go check the kitchen")
    for caller in ({}, {"user_utterance": ""}, {"user_utterance": "   "}, None):
        assert _reference_source_text(mission, caller or {}) == "go check the kitchen"


def test_the_rewrite_loses_the_constraint_that_the_users_words_keep():
    # This is the actual live regression, as data: the extractor recovers a
    # usable constraint from what the user said and not from the paraphrase.
    users_words = "记住离你最近的人叫33。"
    omegas_rewrite = "记住离你最近的人（离机器人最近的那个人）的身份，把这个人标记/命名为“33”。"

    from_user = extract_reference_constraints(users_words)
    from_rewrite = extract_reference_constraints(omegas_rewrite)

    assert from_user[0]["bind_alias"] == "33"
    assert from_rewrite[0]["bind_alias"] == "", (
        "the paraphrase genuinely loses the naming marker -- that is why the "
        "user's own words have to reach the extractor")


def test_the_pipeline_validates_spans_against_the_same_text_they_came_from():
    # A span quoted from the user's words must be checked against those words.
    # Checking it against Omega's rewrite would reject a perfectly real
    # constraint, which is why context_builder records reference_source_text
    # and the guard reads it back.
    users_words = "记住离你最近的人叫33。"
    context = json.dumps({
        "reference_constraints": extract_reference_constraints(users_words),
        "reference_source_text": users_words,
    })
    assert _pipeline_source_text(context, "an unrelated rewrite") == users_words


def test_the_pipeline_falls_back_for_context_without_the_field():
    # Older/absent context must keep working exactly as before.
    assert _pipeline_source_text("{}", "去33那里") == "去33那里"
    assert _pipeline_source_text("", "去33那里") == "去33那里"
    assert _pipeline_source_text("not json at all", "去33那里") == "去33那里"
