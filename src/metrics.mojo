"""Exact n-gram statistics and edit-distance kernels."""

from std.sys.info import simd_width_of


comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def same_ngram(a: IPtr, ai: Int, b: IPtr, bi: Int, order: Int) -> Bool:
    if order == 0:
        return True
    if a[ai] != b[bi]:
        return False

    comptime W = simd_width_of[DType.float64]()
    var k = 1
    while k + W <= order:
        var matches = a.load[width=W](ai + k).eq(b.load[width=W](bi + k))
        if not matches.reduce_and():
            return False
        k += W
    while k < order:
        if a[ai + k] != b[bi + k]:
            return False
        k += 1
    return True


def count_ngram(seq: IPtr, n: Int, start: Int, order: Int, key: IPtr) -> Int:
    var count = 0
    if n < order:
        return count
    for i in range(n - order + 1):
        if same_ngram(key, start, seq, i, order):
            count += 1
    return count


def bleu_stats(
    hyp: IPtr,
    hyp_len: Int,
    refs: IPtr,
    ref_offsets: IPtr,
    nrefs: Int,
    max_order: Int,
    stats: IPtr,
):
    for order_idx in range(max_order):
        var order = order_idx + 1
        var total = hyp_len - order + 1
        if total < 0:
            total = 0
        stats[max_order + order_idx] = Int64(total)
        stats[order_idx] = 0

        for i in range(total):
            var occurrence = 1
            for prev in range(i):
                if same_ngram(hyp, i, hyp, prev, order):
                    occurrence += 1
            var matched = False
            for r in range(nrefs):
                var begin = Int(ref_offsets[r])
                var ref_len = Int(ref_offsets[r + 1]) - begin
                var ref_count = count_ngram(
                    refs + begin, ref_len, i, order, hyp
                )
                if ref_count >= occurrence:
                    matched = True
                    break
            if matched:
                stats[order_idx] += 1


def pair_ngram_stats(
    hyp: IPtr,
    hyp_len: Int,
    reference: IPtr,
    ref_len: Int,
    max_order: Int,
    stats: IPtr,
):
    for order_idx in range(max_order):
        var order = order_idx + 1
        var hyp_total = hyp_len - order + 1
        var ref_total = ref_len - order + 1
        if hyp_total < 0:
            hyp_total = 0
        if ref_total < 0:
            ref_total = 0

        stats[3 * order_idx] = Int64(hyp_total if ref_total > 0 else 0)
        stats[3 * order_idx + 1] = Int64(ref_total)
        stats[3 * order_idx + 2] = 0

        for i in range(hyp_total):
            var occurrence = 1
            for prev in range(i):
                if same_ngram(hyp, i, hyp, prev, order):
                    occurrence += 1
            var ref_count = count_ngram(reference, ref_len, i, order, hyp)
            if ref_count >= occurrence:
                stats[3 * order_idx + 2] += 1


def batch_pair_ngram_stats(
    data: IPtr,
    offsets: IPtr,
    pair_count: Int,
    max_order: Int,
    stats: IPtr,
):
    for pair_index in range(pair_count):
        var hyp_begin = Int(offsets[2 * pair_index])
        var ref_begin = Int(offsets[2 * pair_index + 1])
        var end = Int(offsets[2 * pair_index + 2])
        pair_ngram_stats(
            data + hyp_begin,
            ref_begin - hyp_begin,
            data + ref_begin,
            end - ref_begin,
            max_order,
            stats + pair_index * 3 * max_order,
        )


def batch_bleu_stats(
    data: IPtr,
    sequence_offsets: IPtr,
    sentence_offsets: IPtr,
    sentence_count: Int,
    max_order: Int,
    stats: IPtr,
):
    for sentence_index in range(sentence_count):
        var sequence_begin = Int(sentence_offsets[sentence_index])
        var sequence_end = Int(sentence_offsets[sentence_index + 1])
        var hyp_begin = Int(sequence_offsets[sequence_begin])
        var hyp_end = Int(sequence_offsets[sequence_begin + 1])
        bleu_stats(
            data + hyp_begin,
            hyp_end - hyp_begin,
            data,
            sequence_offsets + sequence_begin + 1,
            sequence_end - sequence_begin - 1,
            max_order,
            stats + sentence_index * 2 * max_order,
        )


def edit_distance(
    hyp: IPtr,
    hyp_len: Int,
    reference: IPtr,
    ref_len: Int,
    costs: IPtr,
    ops: BPtr,
    trace: BPtr,
    trace_len: IPtr,
) -> Int:
    var width = ref_len + 1
    costs[0] = 0
    ops[0] = 0
    for j in range(1, ref_len + 1):
        costs[j] = Int64(j)
        ops[j] = 2
    for i in range(1, hyp_len + 1):
        var row = i * width
        costs[row] = Int64(i)
        ops[row] = 3
        for j in range(1, ref_len + 1):
            var diag = costs[(i - 1) * width + j - 1]
            var op = UInt8(0)
            if hyp[i - 1] != reference[j - 1]:
                diag += 1
                op = 1

            var best = diag
            var deletion = costs[(i - 1) * width + j] + 1
            if deletion < best:
                best = deletion
                op = 3
            var insertion = costs[row + j - 1] + 1
            if insertion < best:
                best = insertion
                op = 2
            costs[row + j] = best
            ops[row + j] = op

    var i = hyp_len
    var j = ref_len
    var length = 0
    while i > 0 or j > 0:
        var op = ops[i * width + j]
        trace[length] = op
        length += 1
        if op == 0 or op == 1:
            i -= 1
            j -= 1
        elif op == 2:
            j -= 1
        else:
            i -= 1

    for k in range(length // 2):
        var tmp = trace[k]
        trace[k] = trace[length - k - 1]
        trace[length - k - 1] = tmp
    trace_len[0] = Int64(length)
    return Int(costs[hyp_len * width + ref_len])
