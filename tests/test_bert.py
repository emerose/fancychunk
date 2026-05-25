"""Optional integration test: late chunking against a real BERT model.

Gated by ``FANCYCHUNK_TEST_USE_BERT=1`` because the model is ~700 MB
and downloads via Hugging Face on first run. The fast suite uses
deterministic fakes that satisfy the embedder contract; this test
verifies the same code path runs against a real transformer.

The adapter below is also one of the reference implementations in
``examples/embedders/huggingface_offsets.py`` — kept here as a test
fixture and there as a user-facing example.
"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import pytest

from fancychunk import embed_with_late_chunking

pytestmark = pytest.mark.skipif(
    os.environ.get("FANCYCHUNK_TEST_USE_BERT") != "1",
    reason="set FANCYCHUNK_TEST_USE_BERT=1 to run real-BERT integration tests",
)


@pytest.fixture(scope="module")
def bert_embedder() -> Any:
    """SegmentEmbedder backed by ``bert-base-multilingual-cased`` using
    HuggingFace's ``offset_mapping`` to align tokens to sentences."""
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    class HFOffsetEmbedder:
        n_ctx = 512

        def count_tokens(self, sentences: list[str]) -> list[int]:
            return [
                len(tokenizer.encode(s, add_special_tokens=False))
                for s in sentences
            ]

        def embed_segment(
            self, sentences: list[str]
        ) -> tuple[np.ndarray, list[int]]:
            # Join with no separator and use offset_mapping to map
            # each token back to its source sentence by character
            # offset. Special tokens ([CLS], [SEP]) get offset (0, 0)
            # and are absorbed into the first/last sentences per
            # SPEC-CHUNK-420 option (b).
            joined = "".join(sentences)
            enc = tokenizer(
                joined,
                return_offsets_mapping=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            with torch.no_grad():
                h = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).last_hidden_state[0]
            mat = cast(np.ndarray, h.numpy())

            # Compute sentence character spans, then count tokens
            # whose offset falls inside each.
            spans = []
            pos = 0
            for s in sentences:
                spans.append((pos, pos + len(s)))
                pos += len(s)

            offsets = enc["offset_mapping"][0].tolist()
            counts = [0] * len(sentences)
            for tok_idx, (a, b) in enumerate(offsets):
                if a == 0 and b == 0:
                    # Special token — defer; assigned at the end.
                    continue
                for s_idx, (sa, sb) in enumerate(spans):
                    if sa <= a < sb or sa < b <= sb:
                        counts[s_idx] += 1
                        break

            # Absorb specials: leading specials → sentence 0; trailing
            # → last sentence. Walk the offset list from the ends.
            for tok_idx, (a, b) in enumerate(offsets):
                if a == 0 and b == 0:
                    counts[0] += 1
                else:
                    break
            for tok_idx in range(len(offsets) - 1, -1, -1):
                a, b = offsets[tok_idx]
                if a == 0 and b == 0:
                    counts[-1] += 1
                else:
                    break

            return mat, counts

    return HFOffsetEmbedder()


SENTENCES = [
    "The quicksort algorithm sorts an array.",
    "It uses a pivot to partition the elements.",
    "Random pivots give expected O(n log n) time.",
    "Worst case is still O(n^2) for adversarial inputs.",
]


def test_offset_method_produces_valid_output(bert_embedder: Any) -> None:
    out = embed_with_late_chunking(SENTENCES, bert_embedder)
    assert out.shape == (4, 768)
    assert np.all(np.isfinite(out))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_normalize_false_returns_raw_means(bert_embedder: Any) -> None:
    out = embed_with_late_chunking(SENTENCES, bert_embedder, normalize=False)
    assert out.shape == (4, 768)
    norms = np.linalg.norm(out, axis=1)
    assert np.all(norms > 0)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_determinism_against_real_model(bert_embedder: Any) -> None:
    a = embed_with_late_chunking(SENTENCES, bert_embedder)
    b = embed_with_late_chunking(SENTENCES, bert_embedder)
    assert np.allclose(a, b)
