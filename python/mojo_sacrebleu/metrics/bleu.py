"""BLEU with exact clipped n-gram statistics computed in Mojo."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .._lib import addr, checked_call, lib, pack_sequences
from .._score import Score, Signature
from .._tokenizers import TOKENIZERS as TOKENIZER_CLASSES
from .base import Metric, encode_sequences

MAX_NGRAM_ORDER = 4
_PARALLEL_THRESHOLD = 512


class BLEUScore(Score):
    def __init__(
        self,
        score: float,
        counts: list,
        totals: list,
        precisions: list[float],
        bp: float,
        sys_len: int,
        ref_len: int,
    ):
        super().__init__("BLEU", score)
        self.bp = bp
        self.counts = counts
        self.totals = totals
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.precisions = precisions
        self.prec_str = "/".join(f"{precision:.1f}" for precision in precisions)
        self.ratio = self.sys_len / self.ref_len if self.ref_len else 0
        self._verbose = (
            f"{self.prec_str} (BP = {self.bp:.3f} ratio = {self.ratio:.3f} "
            f"hyp_len = {self.sys_len:d} ref_len = {self.ref_len:d})"
        )


class BLEU(Metric):
    SMOOTH_DEFAULTS = {"none": None, "floor": 0.1, "add-k": 1, "exp": None}
    TOKENIZERS = TOKENIZER_CLASSES.keys()
    TOKENIZER_DEFAULT = "13a"

    def __init__(
        self,
        lowercase: bool = False,
        force: bool = False,
        tokenize: Optional[str] = None,
        smooth_method: str = "exp",
        smooth_value: Optional[float] = None,
        max_ngram_order: int = MAX_NGRAM_ORDER,
        effective_order: bool = False,
        trg_lang: str = "",
        references: Optional[Sequence[Sequence[str]]] = None,
    ):
        super().__init__()
        assert smooth_method in self.SMOOTH_DEFAULTS, f"Unknown smooth_method {smooth_method!r}"
        if tokenize is None:
            tokenize = {"zh": "zh"}.get(trg_lang, self.TOKENIZER_DEFAULT)
        if tokenize not in TOKENIZER_CLASSES:
            raise KeyError(f"Unknown tokenizer {tokenize!r}")
        self.lowercase = lowercase
        self._force = force
        self.trg_lang = trg_lang
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.max_ngram_order = max_ngram_order
        self.effective_order = effective_order
        self.tokenizer = TOKENIZER_CLASSES[tokenize]()
        self.tokenizer_signature = self.tokenizer.signature
        if references is not None:
            self._ref_cache = self._prepare_references(references)

    def _preprocess_segment(self, sentence: str) -> str:
        if self.lowercase:
            sentence = sentence.lower()
        return self.tokenizer(sentence.rstrip())

    def _extract_reference_info(self, references: Sequence[str]):
        return {"tokens": [reference.split() for reference in references]}

    def _compute_segment_statistics(self, hypothesis: str, ref_info) -> list[int]:
        encoded = encode_sequences([hypothesis.split(), *ref_info["tokens"]])
        hyp, reference_arrays = encoded[0], encoded[1:]
        offsets = np.zeros(len(reference_arrays) + 1, dtype=np.int64)
        for index, reference in enumerate(reference_arrays):
            offsets[index + 1] = offsets[index] + len(reference)
        references = (
            np.concatenate(reference_arrays)
            if any(len(reference) for reference in reference_arrays)
            else np.empty(0, dtype=np.int64)
        )
        kernel_stats = np.zeros(2 * self.max_ngram_order, dtype=np.int64)
        checked_call(
            lib().msb_bleu_stats,
            addr(hyp),
            len(hyp),
            addr(references),
            len(references),
            addr(offsets),
            len(offsets),
            len(reference_arrays),
            self.max_ngram_order,
            addr(kernel_stats),
            kernel_stats.size,
        )
        ref_lengths = [len(reference) for reference in reference_arrays]
        ref_len = min(ref_lengths, key=lambda length: (abs(len(hyp) - length), length))
        return [len(hyp), ref_len, *kernel_stats.tolist()]

    def _compute_corpus_statistics(self, hypotheses, cache):
        sequences = []
        sentence_offsets = np.empty(len(hypotheses) + 1, dtype=np.int64)
        sentence_offsets[0] = 0
        hyp_lengths = []
        ref_lengths = []
        for index, (hypothesis, ref_info) in enumerate(zip(hypotheses, cache)):
            hyp_tokens = hypothesis.split()
            references = ref_info["tokens"]
            sequences.append(hyp_tokens)
            sequences.extend(references)
            sentence_offsets[index + 1] = len(sequences)
            hyp_lengths.append(len(hyp_tokens))
            lengths = [len(reference) for reference in references]
            ref_lengths.append(
                min(lengths, key=lambda length: (abs(len(hyp_tokens) - length), length))
            )

        data, sequence_offsets = pack_sequences(sequences)
        kernel_stats = np.empty(
            (len(hypotheses), 2 * self.max_ngram_order), dtype=np.int64
        )
        checked_call(
            lib(parallel=len(hypotheses) >= _PARALLEL_THRESHOLD).msb_batch_bleu_stats,
            addr(data),
            len(data),
            addr(sequence_offsets),
            len(sequence_offsets),
            addr(sentence_offsets),
            len(sentence_offsets),
            len(hypotheses),
            self.max_ngram_order,
            _PARALLEL_THRESHOLD,
            addr(kernel_stats),
            kernel_stats.size,
        )
        return [
            [hyp_lengths[index], ref_lengths[index], *kernel_stats[index].tolist()]
            for index in range(len(hypotheses))
        ]

    @staticmethod
    def compute_bleu(
        correct: list,
        total: list,
        sys_len: int,
        ref_len: int,
        smooth_method: str = "none",
        smooth_value=None,
        effective_order: bool = False,
        max_ngram_order: int = MAX_NGRAM_ORDER,
    ) -> BLEUScore:
        assert smooth_method in BLEU.SMOOTH_DEFAULTS, f"Unknown smooth_method {smooth_method!r}"
        correct = list(correct)
        total = list(total)
        if smooth_value is None:
            smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]
        bp = 1.0
        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
        precisions = [0.0] * max_ngram_order
        if not any(correct):
            return BLEUScore(0.0, correct, total, precisions, bp, sys_len, ref_len)
        smooth_mteval = 1.0
        effective = max_ngram_order
        for n in range(1, max_ngram_order + 1):
            if smooth_method == "add-k" and n > 1:
                correct[n - 1] += smooth_value
                total[n - 1] += smooth_value
            if total[n - 1] == 0:
                break
            if effective_order:
                effective = n
            if correct[n - 1] == 0:
                if smooth_method == "exp":
                    smooth_mteval *= 2
                    precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
                elif smooth_method == "floor":
                    precisions[n - 1] = 100.0 * smooth_value / total[n - 1]
            else:
                precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]
        logs = [math.log(value) if value else -9999999999 for value in precisions[:effective]]
        score = bp * math.exp(sum(logs) / effective)
        return BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len)

    def _compute_score_from_stats(self, stats: list) -> BLEUScore:
        return self.compute_bleu(
            stats[2 : 2 + self.max_ngram_order],
            stats[2 + self.max_ngram_order :],
            int(stats[0]),
            int(stats[1]),
            self.smooth_method,
            self.smooth_value,
            self.effective_order,
            self.max_ngram_order,
        )

    def _aggregate_and_compute(self, stats: list[list[int]]) -> BLEUScore:
        return self._compute_score_from_stats(np.sum(stats, axis=0).tolist())

    def get_signature(self) -> Signature:
        smooth = self.smooth_method
        default = self.SMOOTH_DEFAULTS[smooth]
        if default is not None:
            value = default if self.smooth_value is None else self.smooth_value
            smooth += f"[{value:.2f}]"
        return Signature(
            {
                "metric": self,
                "values": {
                    "case": "lc" if self.lowercase else "mixed",
                    "eff": "yes" if self.effective_order else "no",
                    "tok": self.tokenizer_signature,
                    "smooth": smooth,
                },
                "abbreviations": {"case": "c", "eff": "e", "tok": "tok", "smooth": "s"},
            }
        )
