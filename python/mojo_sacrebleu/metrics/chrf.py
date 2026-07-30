"""chrF and chrF++ with n-gram intersections computed in Mojo."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._lib import addr, checked_call, lib, pack_sequences
from .._score import Score, Signature
from .base import Metric, encode_sequences


_PARALLEL_THRESHOLD = 256


class CHRFScore(Score):
    def __init__(self, score: float, char_order: int, word_order: int, beta: int):
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        super().__init__(f"chrF{beta}" + "+" * word_order, score)


class CHRF(Metric):
    CHAR_ORDER = 6
    WORD_ORDER = 0
    BETA = 2
    _PUNCTS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    def __init__(
        self,
        char_order: int = CHAR_ORDER,
        word_order: int = WORD_ORDER,
        beta: int = BETA,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
        references: Optional[Sequence[Sequence[str]]] = None,
    ):
        super().__init__()
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = char_order + word_order
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.eps_smoothing = eps_smoothing
        if references is not None:
            self._ref_cache = self._prepare_references(references)

    def _preprocess_segment(self, sentence: str) -> str:
        return sentence.lower() if self.lowercase else sentence

    def _remove_punctuation(self, sentence: str) -> list[str]:
        tokens = []
        for word in sentence.split():
            if len(word) == 1:
                tokens.append(word)
            elif word[-1] in self._PUNCTS:
                tokens.extend([word[:-1], word[-1]])
            elif word[0] in self._PUNCTS:
                tokens.extend([word[0], word[1:]])
            else:
                tokens.append(word)
        return tokens

    def _extract_reference_info(self, references: Sequence[str]):
        return {"references": list(references)}

    @staticmethod
    def _kernel_stats(hypothesis, reference, order: int) -> list[int]:
        if order == 0:
            return []
        hyp, ref = encode_sequences([hypothesis, reference])
        stats = np.zeros(3 * order, dtype=np.int64)
        checked_call(
            lib().msb_pair_ngram_stats,
            addr(hyp), len(hyp), addr(ref), len(ref), order, addr(stats), stats.size
        )
        return stats.tolist()

    def _statistics_for_reference(self, hypothesis: str, reference: str) -> list[int]:
        hyp_chars = list(hypothesis if self.whitespace else "".join(hypothesis.split()))
        ref_chars = list(reference if self.whitespace else "".join(reference.split()))
        stats = self._kernel_stats(hyp_chars, ref_chars, self.char_order)
        if self.word_order:
            stats.extend(
                self._kernel_stats(
                    self._remove_punctuation(hypothesis),
                    self._remove_punctuation(reference),
                    self.word_order,
                )
            )
        return stats

    @staticmethod
    def _batch_kernel_stats(pairs, order: int) -> np.ndarray:
        if order == 0:
            return np.empty((len(pairs), 0), dtype=np.int64)
        sequences = [sequence for pair in pairs for sequence in pair]
        data, offsets = pack_sequences(sequences)
        stats = np.empty((len(pairs), 3 * order), dtype=np.int64)
        checked_call(
            lib(parallel=len(pairs) >= _PARALLEL_THRESHOLD).msb_batch_pair_ngram_stats,
            addr(data),
            len(data),
            addr(offsets),
            len(offsets),
            len(pairs),
            order,
            _PARALLEL_THRESHOLD,
            addr(stats),
            stats.size,
        )
        return stats

    def _compute_corpus_statistics(self, hypotheses, cache):
        char_pairs = []
        word_pairs = []
        reference_counts = []
        for hypothesis, ref_info in zip(hypotheses, cache):
            references = ref_info["references"]
            reference_counts.append(len(references))
            hyp_chars = hypothesis if self.whitespace else "".join(hypothesis.split())
            hyp_words = self._remove_punctuation(hypothesis) if self.word_order else None
            for reference in references:
                ref_chars = reference if self.whitespace else "".join(reference.split())
                char_pairs.append((hyp_chars, ref_chars))
                if self.word_order:
                    word_pairs.append(
                        (hyp_words, self._remove_punctuation(reference))
                    )

        char_stats = self._batch_kernel_stats(char_pairs, self.char_order)
        word_stats = self._batch_kernel_stats(word_pairs, self.word_order)
        result = []
        pair_index = 0
        for reference_count in reference_counts:
            best_stats = []
            best_score = -1.0
            for _ in range(reference_count):
                stats = char_stats[pair_index].tolist()
                if self.word_order:
                    stats.extend(word_stats[pair_index].tolist())
                score = self._compute_f_score(stats)
                if score > best_score:
                    best_stats, best_score = stats, score
                pair_index += 1
            result.append(best_stats)
        return result

    def _compute_f_score(self, statistics: list[int]) -> float:
        eps = 1e-16
        factor = self.beta**2
        score = 0.0
        effective_order = 0
        avg_precision = 0.0
        avg_recall = 0.0
        for index in range(self.order):
            n_hyp, n_ref, n_match = statistics[3 * index : 3 * index + 3]
            precision = n_match / n_hyp if n_hyp > 0 else eps
            recall = n_match / n_ref if n_ref > 0 else eps
            denominator = factor * precision + recall
            score += (
                (1 + factor) * precision * recall / denominator
                if denominator > 0
                else eps
            )
            if n_hyp > 0 and n_ref > 0:
                avg_precision += precision
                avg_recall += recall
                effective_order += 1
        if self.eps_smoothing:
            return 100 * score / self.order
        if effective_order == 0:
            return 0.0
        avg_precision /= effective_order
        avg_recall /= effective_order
        if not avg_precision + avg_recall:
            return 0.0
        score = (1 + factor) * avg_precision * avg_recall
        score /= factor * avg_precision + avg_recall
        return 100 * score

    def _compute_segment_statistics(self, hypothesis: str, ref_info) -> list[int]:
        best_stats = []
        best_score = -1.0
        for reference in ref_info["references"]:
            stats = self._statistics_for_reference(hypothesis, reference)
            score = self._compute_f_score(stats)
            if score > best_score:
                best_stats, best_score = stats, score
        return best_stats

    def _compute_score_from_stats(self, stats: list[int]) -> CHRFScore:
        return CHRFScore(
            self._compute_f_score(stats), self.char_order, self.word_order, self.beta
        )

    def _aggregate_and_compute(self, stats: list[list[int]]) -> CHRFScore:
        return self._compute_score_from_stats(np.sum(stats, axis=0).tolist())

    def get_signature(self) -> Signature:
        return Signature(
            {
                "metric": self,
                "values": {
                    "case": "lc" if self.lowercase else "mixed",
                    "eff": "no" if self.eps_smoothing else "yes",
                    "nc": self.char_order,
                    "nw": self.word_order,
                    "space": "yes" if self.whitespace else "no",
                },
                "abbreviations": {
                    "case": "c",
                    "eff": "e",
                    "nc": "nc",
                    "nw": "nw",
                    "space": "s",
                },
            }
        )
