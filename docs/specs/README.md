# fancychunk specs

Behavioral specifications for fancychunk's text-splitting pipeline. An
implementor reading only these specs (never the upstream source they
were extracted from) must be able to produce a conforming
implementation.

## The reimplementor test

Every sentence in these specs must pass this test:

> Could someone implement this behavior from this sentence alone,
> never having seen the source it was extracted from?

When the answer is "no, they'd need to see the original code," the
sentence is rewritten. Internal function names, variable names, library
choices, and source-file structure are absent by design.

## What is preserved verbatim

- **Numeric constants** that define behavior: `threshold = 0.25`,
  `max_size = 2048`, golden-ratio split `0.382`, target ≈ 3 statements,
  heading boundary probability `1.0`, etc.
- **External vocabulary**: Markdown token type names (`heading_open`,
  `paragraph_open`, `bullet_list_open`, `blockquote_open`,
  `ordered_list_open`) — these are CommonMark / markdown-it parser
  output, not internal names.
- **Patterns the system relies on**: the heading regex `^#+\s` matches
  the Markdown heading syntax itself.
- **Algorithm classes** described by their mathematical type: "dynamic
  programming", "binary integer programming", "covering constraint" —
  these are mathematical concepts, not implementation references.

## What is abstracted

- Function names, variable names, class names, module names from any
  upstream source.
- Specific solver libraries (`scipy.linprog`, `cvxpy`, CP-SAT), array
  libraries (`numpy`, `jax`), or sentence segmentation libraries
  (`wtpsplit`, `spacy`, `nltk`). The spec says *what* output is needed;
  the implementor picks the tool.
- File layout, module boundaries, and whether stages are classes,
  functions, or pipelines.

## Glossary

| Term | Meaning |
|------|---------|
| **document** | Input Markdown string. UTF-8. |
| **sentence** | A contiguous substring of the document, the smallest unit fancychunk operates on. Sentences exhaustively partition the document; their concatenation reconstructs it byte-for-byte. |
| **chunklet** | A paragraph-sized contiguous group of sentences. Targets ≈ 3 statements of information content. The basic unit that gets embedded. |
| **chunk** | A contiguous group of chunklets representing one semantic unit. The unit the caller indexes for retrieval. |
| **statement** | A soft, document-relative measure of a sentence's information content. A sentence with the document's median word count is defined as containing one statement. |
| **boundary probability** | A real number in `[0, 1]` indicating how likely a position is to be a structural break. Comes from two sources: predicted (a sentence segmentation model) and known (Markdown structure). |
| **discourse vector** | The mean embedding of "typical" chunklets in a document, used to subtract a document's overall topic from chunklet embeddings so similarity reflects *local* topic shifts. |
| **partition similarity** | The cosine similarity between adjacent chunklets after discourse correction. Low values indicate a good place to split. |
| **late chunking** | An embedding strategy where each sentence's embedding is computed within the context of a longer surrounding document segment, then mean-pooled per sentence. Produces context-aware sentence embeddings. |
| **preamble** | In late chunking, the leading portion of an encoded segment that provides context but whose output embeddings are discarded. |

## Specification ID ranges

| Range | Stage |
|-------|-------|
| SPEC-CHUNK-1xx | Sentence splitting |
| SPEC-CHUNK-2xx | Chunklet grouping |
| SPEC-CHUNK-3xx | Semantic chunking |
| SPEC-CHUNK-4xx | Late chunking |
| SPEC-CHUNK-9xx | Cross-cutting (concatenation, determinism) |

Uncertainties carry `U-CHUNK-NNN` and are flagged inline.
