"""Identity/Grounding foundation finalization (2026-09-03, GPT re-review):
direct unit coverage for the deterministic extractor that replaced letting
the planner LLM author its own reference_constraints. Every test here
encodes one of GPT's own worked examples -- these are the actual
counter-examples that broke the prior (planner-authored) design, so each
one here is a real regression test, not a hypothetical."""

from mc_ai_bt.reference_extraction import extract_reference_constraints, has_explicit_look_instruction


def _relation_of(constraints):
    return [c["relation"] for c in constraints]


def test_extracts_front_from_the_original_working_e1_phrasing():
    # The exact live-tested-working E.1 phrasing from earlier rounds --
    # confirms this round's rewrite did not regress the one case already
    # proven live.
    result = extract_reference_constraints("There is a person right in front of you. Remember them as 44.")

    assert len(result) == 1
    assert result[0]["relation"] == "front"
    assert result[0]["entity_class"] == "person"
    assert result[0]["reference_frame"] == "robot"
    assert result[0]["source_span"] == "a person right in front of you"


def test_extracts_left_when_person_and_relation_are_directly_bound():
    result = extract_reference_constraints("The person on your left, remember them as 44")

    assert _relation_of(result) == ["left"]
    assert result[0]["source_span"] == "The person on your left"


def test_extracts_right():
    result = extract_reference_constraints("Remember the person on your right as 44.")
    assert _relation_of(result) == ["right"]


def test_extracts_nearest_with_closest_synonym():
    assert _relation_of(extract_reference_constraints("Remember the nearest person as 44")) == ["nearest"]
    assert _relation_of(extract_reference_constraints("Remember the closest person as 44")) == ["nearest"]


def test_extracts_nothing_when_the_relation_describes_a_different_entity():
    # GPT's own counter-example: "on your left" genuinely appears in the
    # text, but it describes the CHAIR, not the person being bound -- the
    # extractor's tight person-relation adjacency requirement means this
    # produces zero constraints, not a mis-attributed one.
    result = extract_reference_constraints("The chair is on your left. Remember this person as 44.")

    assert result == []


def test_extracts_nothing_when_the_relation_is_to_a_different_reference_frame():
    # GPT's own counter-example: "in front" is relative to the SOFA, not
    # the robot -- the pattern requires "in front of you" literally, so a
    # different object of "in front of" never matches at all.
    result = extract_reference_constraints("Remember the person in front of the sofa as 44.")

    assert result == []


def test_extracts_nothing_for_a_bare_utterance_with_no_bearing_words():
    result = extract_reference_constraints("Remember this person as 44.")

    assert result == []


def test_extracts_nothing_when_the_relation_precedes_an_unrelated_look_instruction():
    # GPT's own counter-example: "left" and "person" both appear, but
    # "left" describes a look_at MOVEMENT (a separate clause/instruction),
    # not the person's position -- the tight adjacency requirement (no
    # comma, no clause break between "person" and "on your left") means
    # this correctly produces zero constraints.
    result = extract_reference_constraints("Look to your left, then remember this person as 44.")

    assert result == []


def test_extracts_nothing_for_my_instead_of_your():
    # "my left" is the SPEAKER's own left, not the robot's -- a different,
    # unsupported reference frame. Deliberately never matched, even though
    # it is superficially similar to "your left".
    result = extract_reference_constraints("The person on my left, remember them as 44")

    assert result == []


def test_extracts_a_chinese_left_phrase():
    result = extract_reference_constraints("记住你左边的人叫44")

    assert _relation_of(result) == ["left"]
    assert result[0]["source_span"] == "你左边的人"


def test_extracts_a_chinese_front_phrase():
    assert _relation_of(extract_reference_constraints("记住正前方的人叫44")) == ["front"]
    assert _relation_of(extract_reference_constraints("记住你面前的人叫44")) == ["front"]


def test_extracts_a_chinese_nearest_phrase():
    assert _relation_of(extract_reference_constraints("记住离你最近的人叫44")) == ["nearest"]


def test_extracts_multiple_non_overlapping_constraints():
    result = extract_reference_constraints(
        "The person on your left is 33. The person on your right is 44.")

    assert _relation_of(result) == ["left", "right"]
    assert result[0]["constraint_id"] != result[1]["constraint_id"]


def test_extract_reference_constraints_never_raises_on_empty_or_none():
    assert extract_reference_constraints("") == []
    assert extract_reference_constraints(None) == []


# --- bind_alias capture (2026-09-03, GPT re-review v2) -----------------------
# A constraint's relation/frame/class being real is not enough on its own --
# it can still be borrowed by the wrong remember_person/remember_entity
# action when 2+ constraints exist in the same mission. bind_alias captures
# which alias/name a constraint's OWN sentence assigns, when it does so
# immediately and unambiguously; planning_pipeline.py's guard then requires
# an exact match whenever more than one constraint exists.

def test_bind_alias_captured_from_an_immediately_adjacent_as_clause():
    result = extract_reference_constraints("Remember the person on your left as 44.")
    assert result[0]["bind_alias"] == "44"


def test_bind_alias_captured_from_the_chinese_jiao_marker():
    result = extract_reference_constraints("记住你左边的人叫44")
    assert result[0]["bind_alias"] == "44"


def test_bind_alias_captures_the_original_e1_cross_sentence_phrasing():
    # (2026-09-03, GPT re-review v3): the original E.1 phrasing used to be
    # documented as an "acceptable bind_alias='' case, safe because it is
    # the only constraint" -- GPT's re-review found relying on that
    # singleton exemption was itself incomplete (a single constraint CAN
    # still be used for the wrong alias if the planner names one that
    # doesn't match). Fixed properly: reference_extraction.py now has an
    # explicit, narrow cross-sentence pattern ("<clause>. Remember them/
    # him/her/it as <token>.") that captures this bind_alias directly,
    # rather than depending on there being nothing else to confuse it
    # with. See test_bind_alias_is_empty_with_no_naming_continuation_at_all
    # below for a genuinely still-unresolvable case.
    result = extract_reference_constraints("There is a person right in front of you. Remember them as 44.")
    assert result[0]["bind_alias"] == "44"


def test_bind_alias_is_empty_with_no_naming_continuation_at_all():
    # No "as"/"叫" marker anywhere nearby, same-clause or next-sentence --
    # genuinely no deterministic way to know who this is for. The
    # constraint itself is still real and legal (relation/frame/class are
    # all verified), but as of v4 planning_pipeline.py's guard REJECTS any
    # identity-binding use of it unconditionally -- including when it is
    # the only constraint in the mission. See that guard's own test,
    # test_reference_constraint_guard_rejects_a_single_constraint_with_no_
    # captured_bind_alias, which pins exactly this sentence's behavior.
    result = extract_reference_constraints("There is a person right in front of you. They seem friendly.")
    assert result[0]["bind_alias"] == ""


def test_bind_alias_is_never_captured_from_the_word_is():
    # GPT's own worked counter-example: "is <word>" is deliberately NOT a
    # naming marker in either language -- "is waving" must never be
    # captured as if "waving" were an alias. Only "as" (English) and "叫"
    # (Chinese) are specific enough naming constructions to trust.
    result = extract_reference_constraints(
        "The person on your left is waving. Remember the person on your right as 44.")

    assert _relation_of(result) == ["left", "right"]
    left, right = result
    assert left["bind_alias"] == ""  # "is waving" -- never mistaken for a name
    assert right["bind_alias"] == "44"  # "as 44" -- a real, adjacent naming marker


def test_bind_alias_is_never_captured_across_a_clause_break():
    # A comma alone does not grant enough adjacency for "is"-style
    # markers; only "as"/"叫" are trusted at all, and only immediately
    # after the relation phrase, per the module's own tight-adjacency
    # design (mirroring the relation patterns themselves).
    result = extract_reference_constraints("The person on your left, who is waving, is 33.")
    assert result[0]["bind_alias"] == ""


# --- has_explicit_look_instruction -------------------------------------

def test_look_instruction_recognizes_look_to_your_left():
    assert has_explicit_look_instruction("Look to your left, then remember this person as 44.", "left") is True


def test_look_instruction_recognizes_turn_and_glance_variants():
    assert has_explicit_look_instruction("Turn left and tell me what you see.", "left") is True
    assert has_explicit_look_instruction("Glance right for a second.", "right") is True


def test_look_instruction_maps_up_down_variants_to_their_ground_plane_direction():
    assert has_explicit_look_instruction("Look to your left, then remember this person.", "left_up") is True
    assert has_explicit_look_instruction("Look to your left, then remember this person.", "left_down") is True


def test_look_instruction_is_false_for_a_declarative_position_sentence():
    # The original working E.1 phrasing describes WHERE the person is, not
    # an instruction to look -- must not be treated as a look instruction
    # (it never should have been, but this pins the boundary explicitly).
    assert has_explicit_look_instruction(
        "There is a person right in front of you. Remember them as 44.", "front") is False


def test_look_instruction_is_false_with_no_verb_at_all():
    assert has_explicit_look_instruction("The person on your left, remember them as 44", "left") is False


def test_look_instruction_recognizes_chinese_turn_phrasing():
    assert has_explicit_look_instruction("往左转,然后记住这个人叫44", "left") is True
    assert has_explicit_look_instruction("看右边,然后记住这个人叫44", "right") is True
    assert has_explicit_look_instruction("往前看,然后记住这个人叫44", "front") is True


def test_look_instruction_never_raises_on_empty_or_none():
    assert has_explicit_look_instruction("", "left") is False
    assert has_explicit_look_instruction(None, "left") is False
    assert has_explicit_look_instruction("look left", "") is False
    assert has_explicit_look_instruction("look left", None) is False
