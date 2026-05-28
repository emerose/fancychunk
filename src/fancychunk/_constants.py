"""Named constants from the specs."""

from __future__ import annotations

DEFAULT_MAX_SIZE_CHARS = 2048

BOUNDARY_SCORE_THRESHOLD = 0.25

TARGET_STATEMENTS_PER_CHUNKLET = 3
STATEMENT_COST_FLOOR = 1e-6
STATEMENT_COST_SCALE = 0.5

MIN_Q25_WORDS = 1.0
STATEMENTS_AT_Q25 = 0.75
QUARTILE_GAP_STATEMENTS = 0.50

BOUNDARY_STRENGTH_HEADING = 1.00
BOUNDARY_STRENGTH_BLOCKQUOTE = 0.75
BOUNDARY_STRENGTH_PARAGRAPH = 0.50
BOUNDARY_STRENGTH_LIST = 0.25

TYPICAL_CHUNKLET_LOWER_QUANTILE = 0.15
TYPICAL_CHUNKLET_UPPER_QUANTILE = 0.85
HEADING_SPLIT_BEFORE_DIVISOR = 4.0
HEADING_SPLIT_AFTER_FORBID = 1.0

# SPEC-CHUNK-324 — paragraph-boundary preference. Cutting between two
# sentences of the *same paragraph* breaks the argument flow. This
# penalty is added to a partition point's similarity when the boundary
# is not a paragraph (or stronger) break, so the optimizer prefers
# paragraph breaks when one is available within budget. It is additive
# and modest: when a paragraph exceeds ``max_size`` every feasible cut
# in the window is mid-paragraph and the uniform penalty cancels, so the
# embedder signal still decides where to cut.
MID_PARAGRAPH_PENALTY = 0.25

# SPEC-CHUNK-323 — small-chunk badness. The chunk-partition DP adds a
# per-chunk badness term, graded ``penalty × max(0, 1 − size/(fraction ×
# max_size))`` (relative to the [0, 1] partition-similarity cut cost), so
# the optimizer extends an undersized chunk forward rather than emit it.
# A chunk is merged only when its badness exceeds the *split-quality gap*
# it would give up — so the effective size cutoff scales with how
# distinct the neighbours are, rather than being a fixed number.
#
# Two terms, combined by taking the larger:
#
# * **Front matter** — the document's leading chunk when it is a title
#   with no body of its own, only a subsection (e.g. ``# Title`` then
#   ``## Abstract``). Such a preamble is worthless alone regardless of
#   how distinct it is, so it gets a strong penalty over a wide size
#   range (the abstract makes it too long for the general term below).
# * **General** — any chunk gets a gentle penalty that only bites on a
#   genuinely tiny chunk (a ~20-char fragment is a poor retrieval unit
#   even when it is a distinct topic). It is deliberately weak and
#   short-range so it does not merge legitimately short sections — chunk
#   size alone cannot tell front matter from a short-but-coherent
#   section, so the heavy lifting stays with the front-matter term.
#
# (A bare heading head needs no term of its own: SPEC-CHUNK-322 already
# pins the split-after-heading cost to the maximum, so the DP never
# voluntarily isolates a heading, and a tiny heading the general term
# would catch anyway.)
FRONT_MATTER_CHUNK_PENALTY = 3.0
FRONT_MATTER_CHUNK_TARGET_FRACTION = 0.5
SMALL_CHUNK_PENALTY = 1.5
SMALL_CHUNK_TARGET_FRACTION = 0.2

DEFAULT_PREAMBLE_FRACTION = 0.382

MAX_HEADING_LEVELS = 6
HEADING_PATH_SEPARATOR = "\n"
