"""Mojo-backed BLEU, chrF, and TER with SacreBLEU-compatible APIs."""

from __future__ import annotations

from typing import Optional, Sequence

from .metrics import BLEU, CHRF, TER, BLEUScore, CHRFScore, TERScore
from .version import __version__


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    smooth_method="exp",
    smooth_value=None,
    force=False,
    lowercase=False,
    tokenize="13a",
    use_effective_order=False,
) -> BLEUScore:
    return BLEU(
        lowercase=lowercase,
        force=force,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    ).corpus_score(hypotheses, references)


def raw_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    smooth_value: Optional[float] = 0.1,
) -> BLEUScore:
    return corpus_bleu(
        hypotheses,
        references,
        smooth_method="floor",
        smooth_value=smooth_value,
        force=True,
        tokenize="none",
        use_effective_order=True,
    )


def sentence_bleu(
    hypothesis: str,
    references: Sequence[str],
    smooth_method: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    tokenize="13a",
    use_effective_order: bool = True,
) -> BLEUScore:
    return BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    ).sentence_score(hypothesis, references)


def corpus_chrf(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    return CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    ).corpus_score(hypotheses, references)


def sentence_chrf(
    hypothesis: str,
    references: Sequence[str],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    return CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    ).sentence_score(hypothesis, references)


def corpus_ter(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> TERScore:
    return TER(
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    ).corpus_score(hypotheses, references)


def sentence_ter(
    hypothesis: str,
    references: Sequence[str],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> TERScore:
    return TER(
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    ).sentence_score(hypothesis, references)


__all__ = [
    "BLEU",
    "BLEUScore",
    "CHRF",
    "CHRFScore",
    "TER",
    "TERScore",
    "corpus_bleu",
    "raw_corpus_bleu",
    "sentence_bleu",
    "corpus_chrf",
    "sentence_chrf",
    "corpus_ter",
    "sentence_ter",
    "__version__",
]
