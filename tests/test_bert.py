"""Optional integration test: late chunking against a real BERT model.

Gated by ``FANCYCHUNK_TEST_USE_BERT=1`` because the model is ~700 MB
and downloads via Hugging Face on first run. The fast test suite uses
deterministic fakes that satisfy the embedder contract; this test
verifies the same code path runs against a real transformer.
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
    """Wrap ``bert-base-multilingual-cased`` in the
    ``TokenLevelEmbedder`` protocol."""
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    class RealBert:
        n_ctx = 512

        def tokenize(self, text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=True))

        def detokenize(self, tokens: list[int]) -> str:
            return cast(str, tokenizer.decode(tokens, skip_special_tokens=False))

        def embed(self, text: str) -> np.ndarray:
            ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
            with torch.no_grad():
                out = model(ids).last_hidden_state[0]
            return cast(np.ndarray, out.numpy())

    return RealBert()


SENTENCES = [
    "The quicksort algorithm sorts an array.",
    "It uses a pivot to partition the elements.",
    "Random pivots give expected O(n log n) time.",
    "Worst case is still O(n^2) for adversarial inputs.",
]


def test_default_sentinel_produces_valid_output(bert_embedder: Any) -> None:
    """Default sentinel (⊕) falls back to isolated-tokenization apportionment
    for BERT — the spec allows this; output must still satisfy the contract."""
    out = embed_with_late_chunking(SENTENCES, bert_embedder)
    assert out.shape == (4, 768)
    assert np.all(np.isfinite(out))
    # normalize=True default → all rows unit norm.
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_bert_compatible_sentinel_hits_precise_path(bert_embedder: Any) -> None:
    """``§`` is a single token in mBERT's vocabulary regardless of context,
    so passing it via the keyword argument hits SPEC-CHUNK-420's precise
    sentinel-token alignment without falling back."""
    out = embed_with_late_chunking(SENTENCES, bert_embedder, sentinel="§")
    assert out.shape == (4, 768)
    assert np.all(np.isfinite(out))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_normalize_false_returns_raw_means(bert_embedder: Any) -> None:
    out = embed_with_late_chunking(SENTENCES, bert_embedder, normalize=False)
    assert out.shape == (4, 768)
    # Raw mean-pooled rows have non-unit norm.
    norms = np.linalg.norm(out, axis=1)
    assert np.all(norms > 0)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_determinism_against_real_model(bert_embedder: Any) -> None:
    a = embed_with_late_chunking(SENTENCES, bert_embedder)
    b = embed_with_late_chunking(SENTENCES, bert_embedder)
    assert np.allclose(a, b)
