"""Common corpus and reference-cache behavior for metrics."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np


class Metric:
    def __init__(self):
        self._ref_cache = None
        self.n_bootstrap = None
        self.seed = None

    def _check_sentence_score_args(self, hypothesis: str, references: Sequence[str]) -> None:
        prefix = self.__class__.__name__
        if not isinstance(hypothesis, str):
            raise TypeError(f"{prefix}: The argument `hyp` should be a string.")
        if isinstance(references, str) or not isinstance(references, Sequence):
            raise TypeError(f"{prefix}: The argument `refs` should be a sequence of strings.")
        if not references or (not isinstance(references[0], str) and references[0] is not None):
            raise TypeError(f"{prefix}: Each element of `refs` should be a string.")

    def _prepare_references(self, references):
        if not isinstance(references, Sequence) or not references:
            raise TypeError(
                f"{self.__class__.__name__}: `refs` should be a sequence of sequence of strings."
            )
        if not isinstance(references[0], Sequence) or isinstance(references[0], str):
            raise TypeError(
                f"{self.__class__.__name__}: Each element of `refs` should be a sequence of strings."
            )
        lengths = {len(document) for document in references}
        if len(lengths) != 1:
            raise ValueError("SacreBLEU requires all reference documents to have equal length.")
        cache = []
        counts = set()
        for lines in zip(*references):
            defined = [self._preprocess_segment(line) for line in lines if line is not None]
            if not defined:
                raise ValueError("Each segment needs at least one defined reference.")
            counts.add(len(defined))
            cache.append(self._extract_reference_info(defined))
        self.num_refs = counts.pop() if len(counts) == 1 else -1
        return cache

    def _extract_corpus_statistics(self, hypotheses, references):
        if not isinstance(hypotheses, Sequence) or isinstance(hypotheses, str):
            raise TypeError(f"{self.__class__.__name__}: `hyps` should be a sequence of strings.")
        if not hypotheses or not isinstance(hypotheses[0], str):
            raise TypeError(
                f"{self.__class__.__name__}: Each element of `hyps` should be a string."
            )
        cache = self._prepare_references(references) if references is not None else self._ref_cache
        if cache is None:
            raise RuntimeError("No references provided and the cache is empty.")
        if len(hypotheses) != len(cache):
            raise ValueError(
                f"{self.__class__.__name__}: hypothesis and reference streams have different lengths."
            )
        return self._compute_corpus_statistics(
            [self._preprocess_segment(hypothesis) for hypothesis in hypotheses], cache
        )

    def _compute_corpus_statistics(self, hypotheses, cache):
        return [
            self._compute_segment_statistics(hypothesis, ref_info)
            for hypothesis, ref_info in zip(hypotheses, cache)
        ]

    def sentence_score(self, hypothesis: str, references: Sequence[str]):
        self._check_sentence_score_args(hypothesis, references)
        stats = self._extract_corpus_statistics(
            [hypothesis], [[reference] for reference in references]
        )
        return self._aggregate_and_compute(stats)

    def corpus_score(self, hypotheses, references, n_bootstrap: int = 1):
        stats = self._extract_corpus_statistics(hypotheses, references)
        actual = self._aggregate_and_compute(stats)
        if n_bootstrap > 1:
            seed_text = os.environ.get("SACREBLEU_SEED", "12345")
            seed = None if seed_text.lower() == "none" else int(seed_text)
            rng = np.random.default_rng(seed)
            samples = []
            for _ in range(n_bootstrap):
                indices = rng.choice(len(stats), size=len(stats), replace=True)
                samples.append(self._aggregate_and_compute([stats[i] for i in indices]))
            actual.estimate_ci(samples)
            self.n_bootstrap = n_bootstrap
            self.seed = seed_text.lower() if seed is None else seed
        return actual


def encode_sequences(sequences) -> list[np.ndarray]:
    vocabulary = {}
    encoded = []
    for sequence in sequences:
        row = []
        for item in sequence:
            if item not in vocabulary:
                vocabulary[item] = len(vocabulary) + 1
            row.append(vocabulary[item])
        encoded.append(np.ascontiguousarray(row, dtype=np.int64))
    return encoded
