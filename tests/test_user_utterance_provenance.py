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

from mc_ai_bt.context_builder import effective_execution_intent
from mc_ai_bt.mission import Mission
from mc_ai_bt.planning_pipeline import _effective_execution_intent as _pipeline_intent
from mc_ai_bt.reference_extraction import extract_reference_constraints


class _M:
    def __init__(self, intent_text):
        self.intent_text = intent_text


def test_the_users_own_words_win_over_omegas_rewrite():
    caller = {"user_utterance": "记住离你最近的人叫33。"}
    mission = _M("Remember the person closest to you and label them as 33")
    assert effective_execution_intent(mission, caller) == "记住离你最近的人叫33。"


def test_the_submitted_text_is_used_when_the_bridge_carried_nothing():
    # Omega-initiated missions (a reminder coming due, a ROBOT_EVENT it acted
    # on) have no user utterance behind them, and the Bridge deliberately
    # sends nothing rather than a stale one.
    mission = _M("go check the kitchen")
    for caller in ({}, {"user_utterance": ""}, {"user_utterance": "   "}, None):
        assert effective_execution_intent(mission, caller or {}) == "go check the kitchen"


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
        "effective_execution_intent": users_words,
    })
    assert _pipeline_intent(context, "an unrelated rewrite") == users_words


def test_the_pipeline_falls_back_for_context_without_the_field():
    # Older/absent context must keep working exactly as before.
    assert _pipeline_intent("{}", "去33那里") == "去33那里"
    assert _pipeline_intent("", "去33那里") == "去33那里"
    assert _pipeline_intent("not json at all", "去33那里") == "去33那里"


# --- Authoritative Execution Intent (2026-09-05) ---------------------------
# The live evidence: one six-character request reached the planner as four
# different sentences across four runs, and the fourth added a physical goal
# the user never asked for. These pin the contract that ends that.

_USERS_WORDS = "去33那里。"

_OMEGAS_FOUR_REWRITES = [
    "去33那里",
    "去离我最近、我之前记住的那个叫33的人那里",
    "Navigate to the person named 33 and approach them",
    "Go to the person known as 33 and stop when you are directly in front of them, facing them",
]


def test_every_omega_rewrite_of_one_request_yields_the_same_authoritative_intent():
    """The whole point: whatever Omega does to the wording, the planner plans
    against what the user actually said."""
    caller = {"user_utterance": _USERS_WORDS}
    resolved = {
        effective_execution_intent(_M(rewrite), caller) for rewrite in _OMEGAS_FOUR_REWRITES
    }
    assert resolved == {_USERS_WORDS}, (
        "four rewrites must collapse to one authoritative intent; "
        f"got {resolved}")


def test_the_fourth_rewrite_no_longer_reaches_the_planner():
    # This is the one that added "stop when you are directly in front of them,
    # facing them" and produced locate_entity + face_entity. The user asked for
    # none of that.
    expanded = _OMEGAS_FOUR_REWRITES[3]
    intent = effective_execution_intent(_M(expanded), {"user_utterance": _USERS_WORDS})
    assert intent == _USERS_WORDS
    assert "facing them" not in intent
    assert "in front of" not in intent


def test_an_omega_originated_mission_keeps_omegas_own_intent():
    # E2 will have these: a reminder coming due, a world event Omega acted on.
    # There is no user turn behind them and Omega's text is the real goal.
    omega_task = "go remind Alice about her medication"
    assert effective_execution_intent(_M(omega_task), {}) == omega_task


def test_an_unrelated_user_remark_is_never_bound_to_an_omega_mission():
    """The Bridge only carries user_utterance while its own per-turn reply
    stream is open, so an unrelated remark cannot arrive here at all. Pinned
    from this side too: given no carried utterance, Omega's text wins, no
    matter how recently the user happened to say something else."""
    omega_task = "go check on Alice because a world event fired"
    for caller in ({}, {"user_utterance": ""}):
        assert effective_execution_intent(_M(omega_task), caller) == omega_task


def test_the_audit_trail_keeps_both_texts():
    import json as _json
    from mc_ai_bt.context_builder import ContextBuilder
    from mc_ai_bt.mission import Identity, Mission

    mission = Mission(
        identity=Identity(mission_id="m1"),
        intent_text=_OMEGAS_FOUR_REWRITES[3],
        source="omegaclaw",
        operator_id="omegaclaw",
        priority=10,
        allow_queue=True,
        context_json=_json.dumps({"user_utterance": _USERS_WORDS}),
    )
    context = _json.loads(ContextBuilder().build_json(mission, ()))
    assert context["effective_execution_intent"] == _USERS_WORDS
    assert context["omega_submitted_intent"] == _OMEGAS_FOUR_REWRITES[3]
    assert context["effective_intent_source"] == "user"
    # what Omega submitted must remain visible, never overwritten
    assert context["mission"]["intent_text"] == _OMEGAS_FOUR_REWRITES[3]
