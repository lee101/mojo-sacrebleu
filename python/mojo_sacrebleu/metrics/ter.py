"""Translation edit rate with Mojo dynamic programming."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._score import Score, Signature
from .._ter import translation_edit_rate
from .._tokenizers import ter_tokenize
from .base import Metric


class TERScore(Score):
    def __init__(self, score: float, num_edits: float, ref_length: float):
        super().__init__("TER", score)
        self.num_edits = int(num_edits)
        self.ref_length = ref_length


class TER(Metric):
    def __init__(
        self,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
        references: Optional[Sequence[Sequence[str]]] = None,
    ):
        super().__init__()
        self.normalized = normalized
        self.no_punct = no_punct
        self.asian_support = asian_support
        self.case_sensitive = case_sensitive
        self.tokenizer_signature = "tercom"
        if references is not None:
            self._ref_cache = self._prepare_references(references)

    def _preprocess_segment(self, sentence: str) -> str:
        return ter_tokenize(
            sentence.rstrip(),
            self.normalized,
            self.no_punct,
            self.asian_support,
            self.case_sensitive,
        )

    def _extract_reference_info(self, references: Sequence[str]):
        return {"ref_words": [reference.split() for reference in references]}

    def _compute_segment_statistics(self, hypothesis: str, ref_info) -> list[float]:
        words_hyp = hypothesis.split()
        best_edits = 10**16
        total_ref_length = 0
        for words_ref in ref_info["ref_words"]:
            edits, ref_length = translation_edit_rate(words_hyp, words_ref)
            best_edits = min(best_edits, edits)
            total_ref_length += ref_length
        return [best_edits, total_ref_length / len(ref_info["ref_words"])]

    def _compute_score_from_stats(self, stats: list[float]) -> TERScore:
        edits, ref_length = stats
        if ref_length > 0:
            score = edits / ref_length
        elif edits > 0:
            score = 1.0
        else:
            score = 0.0
        return TERScore(100 * score, edits, ref_length)

    def _aggregate_and_compute(self, stats: list[list[float]]) -> TERScore:
        return self._compute_score_from_stats(np.sum(stats, axis=0).tolist())

    def get_signature(self) -> Signature:
        return Signature(
            {
                "metric": self,
                "values": {
                    "case": "mixed" if self.case_sensitive else "lc",
                    "tok": self.tokenizer_signature,
                    "norm": self.normalized,
                    "punct": not self.no_punct,
                    "asian": self.asian_support,
                },
                "abbreviations": {
                    "case": "c",
                    "tok": "t",
                    "norm": "nr",
                    "punct": "pn",
                    "asian": "as",
                },
            }
        )
