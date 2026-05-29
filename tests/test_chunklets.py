"""Stage 2 tests — chunklet grouping."""

from __future__ import annotations

import pytest

from fancychunk import OversizedSentenceError, split_chunklets


# TV-201 / SPEC-CHUNK-260 — empty input.
def test_tv_201_empty_input() -> None:
    assert split_chunklets([]) == []


# TV-202 / SPEC-CHUNK-261 — single sentence passes through.
def test_tv_202_single_sentence() -> None:
    assert split_chunklets(["Just one sentence."]) == ["Just one sentence."]


# TV-203 / SPEC-CHUNK-200 — concatenation round-trip.
@pytest.mark.parametrize(
    "sentences",
    [
        ["a" * 1000, "b" * 1000, "c" * 1000],
        ["one ", "two ", "three"],
        ["# Heading\n\n", "Body sentence.\n"],
    ],
)
def test_tv_203_round_trip(sentences: list[str]) -> None:
    out = split_chunklets(sentences)
    assert "".join(out) == "".join(sentences)


# TV-204 / SPEC-CHUNK-201 — hard size constraint forces split.
def test_tv_204_size_constraint() -> None:
    s = ["a" * 1000, "b" * 1000, "c" * 1000]
    out = split_chunklets(s, max_size=2048)
    for c in out:
        assert len(c) <= 2048
    assert "".join(out) == "".join(s)


# TV-205 / SPEC-CHUNK-240, SPEC-CHUNK-241 — heading dominates its run.
def test_tv_205_heading_probas_dominate() -> None:
    from fancychunk.chunklets import _per_sentence_boundary_probas

    sentences = [
        "First paragraph sentence.\n\n",
        "## A new section\n\n",
        "Body sentence one.\n",
        "Body sentence two.\n",
    ]
    probas = _per_sentence_boundary_probas(sentences)
    # Before suppression: [0.5 (paragraph), 1.0 (heading), 0.5
    # (paragraph), 0.0]. All three non-zero entries form one run; only
    # the heading's 1.0 survives.
    assert probas[1] == 1.0
    assert probas[0] == 0.0
    assert probas[2] == 0.0


# TV-206 / SPEC-CHUNK-221, -262 — statement cost drives multi-chunklet split
# when the total fits in max_size.
def test_tv_206_statement_target_drives_split() -> None:
    s = " ".join(["word"] * 10) + ". "
    sentences = [s] * 12
    out = split_chunklets(sentences, max_size=2048)
    assert "".join(out) == "".join(sentences)
    # 12 sentences sum to ~624 chars but the DP still chooses to split
    # them into multiple chunklets; the precise count depends on the
    # statement-cost minimum (each sentence ≈ 0.75 statements, so
    # chunklets of 4 sentences ≈ 3 statements is the math-optimal).
    assert len(out) > 1


# TV-208 / SPEC-CHUNK-241 — consecutive non-zero suppression keeps only the strongest.
def test_tv_208_consecutive_suppression() -> None:
    from fancychunk.chunklets import _per_sentence_boundary_probas

    sentences = [
        "Intro.\n\n",
        "> Blockquote line.\n\n",
        "* Bullet one.\n\n",
        "Continued text.\n",
    ]
    probas = _per_sentence_boundary_probas(sentences)
    # Sentence 2 ("Bullet one") was assigned 0.25 (list strength)
    # before suppression; after suppression it must be 0 because it
    # sits in a run dominated by sentence 1's 0.75 (blockquote).
    assert probas[2] == 0.0
    # Sentence 1 retains the strongest value in its run.
    assert probas[1] == 0.75


# TV-209 / SPEC-CHUNK-240 — interior sentences of a one-line paragraph
# score zero. Only the sentence that *opens* a block gets the structural
# strength; sentences sharing the same source line are interior.
def test_tv_209_interior_sentences_score_zero() -> None:
    from fancychunk.chunklets import _per_sentence_boundary_probas

    sentences = [
        "First para only sentence.\n\n",
        "Second para one. ",
        "Second para two. ",
        "Second para three.\n\n",
        "## Heading\n\n",
        "Body.\n",
    ]
    probas = _per_sentence_boundary_probas(sentences)
    # Sentences 1-3 share the second paragraph's single source line.
    # Only sentence 1 opens the block; 2 and 3 are interior -> 0.0.
    # The guard (not SPEC-CHUNK-241 suppression) is what zeroes them,
    # so the heading at index 4 keeps its 1.0 instead of the whole
    # document collapsing to a single surviving boundary.
    assert probas[2] == 0.0
    assert probas[3] == 0.0
    assert probas[4] == 1.0


# Regression (Issue 1) — a forced split lands at the paragraph boundary,
# not mid-paragraph. Two one-line paragraphs whose combined length
# exceeds max_size must split at the `\n\n` between them: the first
# paragraph stays whole and the second opens a new chunklet. Before the
# SPEC-CHUNK-240 block-opener guard, the second paragraph's opening
# sentence scored 0.0 (its paragraph cue was suppressed along with the
# interior sentences), so boundary cost gave no reason to start there and
# the split fell mid-paragraph, orphaning a continuation.
def test_forced_split_prefers_paragraph_boundary() -> None:
    p1 = " ".join(
        f"First para sentence {i} has a moderate amount of text content."
        for i in range(6)
    )
    p2 = " ".join(
        f"Second para sentence {i} also has a moderate amount of text."
        for i in range(6)
    )
    from fancychunk import split_sentences

    # Each paragraph is a single source line split into several sentences.
    sentences = split_sentences(p1 + "\n\n" + p2 + "\n", max_len=2048)
    out = split_chunklets(sentences, max_size=len(p1) + 40)
    assert any(p1 in c for c in out)  # first paragraph kept whole
    assert any(c.lstrip().startswith("Second para sentence 0") for c in out)


# TV-210 / SPEC-CHUNK-251 — constant-zero costs yield a single chunklet.
def test_tv_210_constant_zero_costs() -> None:
    s = ["Identical sentence content."] * 6
    out = split_chunklets(
        s,
        max_size=2048,
        boundary_cost=lambda p: 0.0,
        statement_cost=lambda x: 0.0,
    )
    assert out == ["".join(s)]


# TV-211 / SPEC-CHUNK-263 — sentence exceeds max_size.
def test_tv_211_oversized_sentence_raises() -> None:
    with pytest.raises(OversizedSentenceError):
        split_chunklets(["a" * 3000, "short tail.\n"], max_size=2048)


# SPEC-CHUNK-202 — chunklet count in [1, len(sentences)].
def test_chunklet_count_bounds() -> None:
    sentences = ["First sentence. ", "Second sentence. ", "Third sentence."]
    out = split_chunklets(sentences)
    assert 1 <= len(out) <= len(sentences)


# SPEC-CHUNK-250 — determinism.
def test_determinism() -> None:
    s = [f"Sentence number {i}. " for i in range(8)]
    assert split_chunklets(s) == split_chunklets(s)
