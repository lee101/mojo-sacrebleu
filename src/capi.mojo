"""C ABI wrappers for the Python bindings."""

from metrics import (
    BPtr,
    IPtr,
    batch_bleu_stats,
    batch_pair_ngram_stats,
    bleu_stats,
    edit_distance,
    pair_ngram_stats,
)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


@export("msb_bleu_stats")
def msb_bleu_stats(
    hyp: Int,
    hyp_len: Int,
    refs: Int,
    refs_len: Int,
    ref_offsets: Int,
    ref_offsets_len: Int,
    nrefs: Int,
    max_order: Int,
    stats: Int,
    stats_len: Int,
) abi("C") -> Int:
    if (
        hyp == 0 or refs == 0 or ref_offsets == 0 or stats == 0
        or hyp_len < 0 or hyp_len > 1000000000
        or refs_len < 0 or nrefs <= 0 or nrefs > 1000000000
        or max_order <= 0 or max_order > 1024
        or ref_offsets_len != nrefs + 1 or stats_len != 2 * max_order
    ):
        return -1
    var offsets = iptr(ref_offsets)
    if Int(offsets[0]) != 0 or Int(offsets[nrefs]) != refs_len:
        return -1
    for i in range(nrefs):
        if offsets[i] < 0 or offsets[i] > offsets[i + 1]:
            return -1
    bleu_stats(
        iptr(hyp),
        hyp_len,
        iptr(refs),
        offsets,
        nrefs,
        max_order,
        iptr(stats),
    )
    return 0


@export("msb_pair_ngram_stats")
def msb_pair_ngram_stats(
    hyp: Int,
    hyp_len: Int,
    reference: Int,
    ref_len: Int,
    max_order: Int,
    stats: Int,
    stats_len: Int,
) abi("C") -> Int:
    if (
        hyp == 0 or reference == 0 or stats == 0
        or hyp_len < 0 or hyp_len > 1000000000
        or ref_len < 0 or ref_len > 1000000000
        or max_order <= 0 or max_order > 1024
        or stats_len != 3 * max_order
    ):
        return -1
    pair_ngram_stats(
        iptr(hyp),
        hyp_len,
        iptr(reference),
        ref_len,
        max_order,
        iptr(stats),
    )
    return 0


@export("msb_batch_pair_ngram_stats")
def msb_batch_pair_ngram_stats(
    data: Int,
    data_len: Int,
    offsets: Int,
    offsets_len: Int,
    pair_count: Int,
    max_order: Int,
    parallel_threshold: Int,
    stats: Int,
    stats_len: Int,
) abi("C") -> Int:
    if (
        data == 0 or offsets == 0 or stats == 0
        or data_len < 0 or pair_count < 0 or pair_count > 1000000000
        or max_order <= 0 or max_order > 1024
        or parallel_threshold <= 0 or offsets_len != 2 * pair_count + 1
        or stats_len != 3 * pair_count * max_order
    ):
        return -1
    var sequence_offsets = iptr(offsets)
    if Int(sequence_offsets[0]) != 0 or Int(sequence_offsets[offsets_len - 1]) != data_len:
        return -1
    for i in range(offsets_len - 1):
        if sequence_offsets[i] < 0 or sequence_offsets[i] > sequence_offsets[i + 1]:
            return -1
    batch_pair_ngram_stats(
        iptr(data),
        sequence_offsets,
        pair_count,
        max_order,
        parallel_threshold,
        iptr(stats),
    )
    return 0


@export("msb_batch_bleu_stats")
def msb_batch_bleu_stats(
    data: Int,
    data_len: Int,
    sequence_offsets: Int,
    sequence_offsets_len: Int,
    sentence_offsets: Int,
    sentence_offsets_len: Int,
    sentence_count: Int,
    max_order: Int,
    parallel_threshold: Int,
    stats: Int,
    stats_len: Int,
) abi("C") -> Int:
    if (
        data == 0 or sequence_offsets == 0 or sentence_offsets == 0 or stats == 0
        or data_len < 0 or sentence_count < 0 or sentence_count > 1000000000
        or max_order <= 0 or max_order > 1024
        or parallel_threshold <= 0 or sentence_offsets_len != sentence_count + 1
        or sequence_offsets_len <= 0 or stats_len != 2 * sentence_count * max_order
    ):
        return -1
    var sequences = iptr(sequence_offsets)
    var sentences = iptr(sentence_offsets)
    if (
        Int(sequences[0]) != 0 or Int(sequences[sequence_offsets_len - 1]) != data_len
        or Int(sentences[0]) != 0
        or Int(sentences[sentence_count]) != sequence_offsets_len - 1
    ):
        return -1
    for i in range(sequence_offsets_len - 1):
        if sequences[i] < 0 or sequences[i] > sequences[i + 1]:
            return -1
    for i in range(sentence_count):
        if sentences[i] >= sentences[i + 1]:
            return -1
    batch_bleu_stats(
        iptr(data),
        sequences,
        sentences,
        sentence_count,
        max_order,
        parallel_threshold,
        iptr(stats),
    )
    return 0


@export("msb_edit_distance")
def msb_edit_distance(
    hyp: Int,
    hyp_len: Int,
    reference: Int,
    ref_len: Int,
    costs: Int,
    ops: Int,
    trace: Int,
    trace_len: Int,
    costs_len: Int,
    ops_len: Int,
    trace_capacity: Int,
) abi("C") -> Int:
    if (
        hyp == 0 or reference == 0 or costs == 0 or ops == 0
        or trace == 0 or trace_len == 0
        or hyp_len < 0 or hyp_len > 1000000000
        or ref_len < 0 or ref_len > 1000000000
        or costs_len != (hyp_len + 1) * (ref_len + 1)
        or ops_len != costs_len or trace_capacity < hyp_len + ref_len
    ):
        return -1
    return edit_distance(
        iptr(hyp),
        hyp_len,
        iptr(reference),
        ref_len,
        iptr(costs),
        bptr(ops),
        bptr(trace),
        iptr(trace_len),
    )
