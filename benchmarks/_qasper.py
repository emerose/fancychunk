"""Qasper dataset loader + Markdown reconstruction.

Qasper is QA over full NLP papers from ACL/EMNLP/NAACL — long
documents with real section structure (Abstract, Methods, Results,
etc.), real scientific prose, and ground-truth evidence spans.
Perfect fit for evaluating chunking strategies that exploit document
structure (heading-aware splits, late chunking).

Each paper is reconstructed into a flat Markdown string:

    # Title

    ## Abstract
    <abstract>

    ## <Section 1 name>
    <paragraph>

    <paragraph>

    ## <Section 2 name>
    ...

Evidence spans appear as exact substrings in the reconstruction
(the headings we synthesize don't appear in evidence text), so
chunk-relevance is a clean substring check.

The HF schema for ``allenai/qasper`` nests questions and answers
inside a column-of-lists structure. We flatten and skip malformed
or unanswerable entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QasperQuestion:
    """One question paired with its gold evidence."""

    question_id: str
    question: str
    evidence: list[str]  # gold evidence text spans from the paper
    answer: str           # free-form answer; "" if extractive/yes-no only
    unanswerable: bool


@dataclass
class QasperPaper:
    """One paper, reconstructed to Markdown with its questions."""

    id: str
    title: str
    markdown: str
    questions: list[QasperQuestion]


def load_qasper(
    split: str = "validation",
    limit: int | None = None,
    min_questions: int = 1,
) -> list[QasperPaper]:
    """Load and reconstruct Qasper papers.

    Parameters
    ----------
    split:
        HF split — ``"train"`` (~888), ``"validation"`` (~281), or
        ``"test"`` (~416).
    limit:
        Optional cap on the number of papers returned. Useful for
        quick local iteration.
    min_questions:
        Skip papers with fewer than this many answerable questions.
        Default 1.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/qasper", split=split)
    papers: list[QasperPaper] = []

    for row in ds:
        if limit is not None and len(papers) >= limit:
            break
        paper = _row_to_paper(row)
        # Drop papers with no answerable questions (nothing to retrieve).
        answerable = [q for q in paper.questions if not q.unanswerable and q.evidence]
        if len(answerable) < min_questions:
            continue
        paper.questions = answerable
        papers.append(paper)

    return papers


def _row_to_paper(row: dict[str, Any]) -> QasperPaper:
    return QasperPaper(
        id=str(row.get("id", "")),
        title=str(row.get("title", "")),
        markdown=_reconstruct_markdown(row),
        questions=_extract_questions(row),
    )


def _reconstruct_markdown(row: dict[str, Any]) -> str:
    """Build a single Markdown string from the paper's structured fields."""
    parts: list[str] = []
    title = (row.get("title") or "").strip()
    if title:
        parts.append(f"# {title}\n\n")

    abstract = (row.get("abstract") or "").strip()
    if abstract:
        parts.append("## Abstract\n\n")
        parts.append(abstract)
        parts.append("\n\n")

    full_text = row.get("full_text") or {}
    section_names: list[str] = full_text.get("section_name") or []
    section_paras: list[list[str]] = full_text.get("paragraphs") or []

    for name, paras in zip(section_names, section_paras):
        name = (name or "").strip()
        if name:
            parts.append(f"## {name}\n\n")
        for para in paras:
            para = (para or "").strip()
            if not para:
                continue
            parts.append(para)
            parts.append("\n\n")

    return "".join(parts)


def _extract_questions(row: dict[str, Any]) -> list[QasperQuestion]:
    """Flatten the nested qas structure into per-question records.

    HF schema (the gnarly part):
        qas: {
            "question_id": list[str],
            "question": list[str],
            "answers": list[{
                "answer": list[{
                    "evidence": list[str],
                    "extractive_spans": list[str],
                    "free_form_answer": str,
                    "unanswerable": bool,
                    "yes_no": bool | None,
                    "highlighted_evidence": list[str],
                }],
                "annotation_id": list[str],
            }]
        }

    A single (question_id, question) may have multiple annotators'
    answers. We union the evidence across annotators and take the
    first non-empty free-form answer.
    """
    out: list[QasperQuestion] = []
    qas = row.get("qas") or {}
    ids: list[str] = qas.get("question_id") or []
    questions: list[str] = qas.get("question") or []
    answers_outer: list[Any] = qas.get("answers") or []

    for qid, qtext, ans_block in zip(ids, questions, answers_outer):
        ann_list = (ans_block or {}).get("answer") or []
        evidence: list[str] = []
        free_form = ""
        unanswerable = True
        for ann in ann_list:
            if not isinstance(ann, dict):
                continue
            if not ann.get("unanswerable", True):
                unanswerable = False
            for span in ann.get("evidence") or []:
                span = (span or "").strip()
                if span and span not in evidence:
                    evidence.append(span)
            ff = (ann.get("free_form_answer") or "").strip()
            if ff and not free_form:
                free_form = ff
        out.append(
            QasperQuestion(
                question_id=str(qid),
                question=str(qtext),
                evidence=evidence,
                answer=free_form,
                unanswerable=unanswerable,
            )
        )
    return out
