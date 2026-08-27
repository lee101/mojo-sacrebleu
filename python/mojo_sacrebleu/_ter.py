"""TER shift search with its edit-distance matrix computed in Mojo.

The shift selection rules follow SacreBLEU's Apache-2.0 TER implementation.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, i64, lib

_MAX_SHIFT_SIZE = 10
_MAX_SHIFT_DIST = 50
_MAX_SHIFT_CANDIDATES = 1000


class _EditDistance:
    def __init__(self, hypothesis: list[str], reference: list[str]):
        vocabulary = {word: idx + 1 for idx, word in enumerate(dict.fromkeys(reference + hypothesis))}
        self._ids = vocabulary
        self.reference = i64([vocabulary[word] for word in reference])
        self.costs = np.empty((len(hypothesis) + 1) * (len(reference) + 1), dtype=np.int64)
        self.ops = np.empty_like(self.costs, dtype=np.uint8)
        self.trace = np.empty(max(1, len(hypothesis) + len(reference)), dtype=np.uint8)
        self.trace_len = np.empty(1, dtype=np.int64)

    def __call__(self, hypothesis: list[str]) -> tuple[int, list[int]]:
        hyp = i64([self._ids[word] for word in hypothesis])
        distance = lib().msb_edit_distance(
            addr(hyp),
            len(hyp),
            addr(self.reference),
            len(self.reference),
            addr(self.costs),
            addr(self.ops),
            addr(self.trace),
            addr(self.trace_len),
            self.costs.size,
            self.ops.size,
            self.trace.size,
        )
        if distance < 0:
            raise ValueError("Mojo edit-distance kernel rejected invalid buffer metadata")
        return int(distance), self.trace[: self.trace_len[0]].tolist()


def _trace_to_alignment(trace: list[int]) -> tuple[dict[int, int], list[int], list[int]]:
    pos_hyp = -1
    pos_ref = -1
    hyp_err: list[int] = []
    ref_err: list[int] = []
    align: dict[int, int] = {}
    for raw_op in trace:
        op = 3 if raw_op == 2 else 2 if raw_op == 3 else raw_op
        if op == 0:
            pos_hyp += 1
            pos_ref += 1
            align[pos_ref] = pos_hyp
            hyp_err.append(0)
            ref_err.append(0)
        elif op == 1:
            pos_hyp += 1
            pos_ref += 1
            align[pos_ref] = pos_hyp
            hyp_err.append(1)
            ref_err.append(1)
        elif op == 2:
            pos_hyp += 1
            hyp_err.append(1)
        else:
            pos_ref += 1
            align[pos_ref] = pos_hyp
            ref_err.append(1)
    return align, ref_err, hyp_err


def _find_shifted_pairs(
    hypothesis: list[str],
    reference: list[str],
    reference_positions: dict[str, list[int]],
):
    hyp_len = len(hypothesis)
    ref_len = len(reference)
    for start_h in range(hyp_len):
        first_ref = max(0, start_h - _MAX_SHIFT_DIST)
        last_ref = min(ref_len, start_h + _MAX_SHIFT_DIST + 1)
        for start_r in reference_positions.get(hypothesis[start_h], ()):
            if start_r < first_ref:
                continue
            if start_r >= last_ref:
                break
            max_length = _MAX_SHIFT_SIZE
            remaining = hyp_len - start_h
            if remaining < max_length:
                max_length = remaining
            remaining = ref_len - start_r
            if remaining < max_length:
                max_length = remaining
            length = 0
            while (
                length < max_length
                and hypothesis[start_h + length] == reference[start_r + length]
            ):
                length += 1
                yield start_h, start_r, length


def _perform_shift(words: list[str], start: int, length: int, target: int) -> list[str]:
    if target < start:
        return (
            words[:target]
            + words[start : start + length]
            + words[target:start]
            + words[start + length :]
        )
    if target > start + length:
        return (
            words[:start]
            + words[start + length : target]
            + words[start : start + length]
            + words[target:]
        )
    return (
        words[:start]
        + words[start + length : length + target]
        + words[start : start + length]
        + words[length + target :]
    )


def _shift(
    hypothesis: list[str],
    reference: list[str],
    reference_positions: dict[str, list[int]],
    edit_distance: _EditDistance,
    checked: int,
) -> tuple[int, list[str], int]:
    pre_score, inverse_trace = edit_distance(hypothesis)
    align, ref_err, hyp_err = _trace_to_alignment(inverse_trace)
    best = None
    hyp_err_prefix = [0]
    ref_err_prefix = [0]
    for error in hyp_err:
        hyp_err_prefix.append(hyp_err_prefix[-1] + error)
    for error in ref_err:
        ref_err_prefix.append(ref_err_prefix[-1] + error)

    for start_h, start_r, length in _find_shifted_pairs(
        hypothesis, reference, reference_positions
    ):
        if hyp_err_prefix[start_h + length] == hyp_err_prefix[start_h]:
            continue
        if ref_err_prefix[start_r + length] == ref_err_prefix[start_r]:
            continue
        if start_h <= align[start_r] < start_h + length:
            continue

        previous_idx = -1
        for offset in range(-1, length):
            if start_r + offset == -1:
                idx = 0
            elif start_r + offset in align:
                idx = align[start_r + offset] + 1
            else:
                break
            if idx == previous_idx:
                continue
            previous_idx = idx
            shifted = _perform_shift(hypothesis, start_h, length, idx)
            candidate = (
                pre_score - edit_distance(shifted)[0],
                length,
                -start_h,
                -idx,
                shifted,
            )
            checked += 1
            if best is None or candidate > best:
                best = candidate
        if checked >= _MAX_SHIFT_CANDIDATES:
            break

    if best is None:
        return 0, hypothesis, checked
    return best[0], best[4], checked


def translation_edit_rate(hypothesis: list[str], reference: list[str]) -> tuple[int, int]:
    if not reference:
        return len(hypothesis), 0
    edit_distance = _EditDistance(hypothesis, reference)
    reference_positions: dict[str, list[int]] = {}
    for index, word in enumerate(reference):
        reference_positions.setdefault(word, []).append(index)
    shifted = hypothesis
    shifts = 0
    checked = 0
    while True:
        delta, candidate, checked = _shift(
            shifted, reference, reference_positions, edit_distance, checked
        )
        if checked >= _MAX_SHIFT_CANDIDATES or delta <= 0:
            break
        shifts += 1
        shifted = candidate
    return shifts + edit_distance(shifted)[0], len(reference)
