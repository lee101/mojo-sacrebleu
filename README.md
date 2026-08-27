# mojo-sacrebleu

BLEU, chrF, chrF++, and TER metrics implemented with Mojo kernels and exposed
through a SacreBLEU-compatible Python API.

```python
import mojo_sacrebleu as sacrebleu

hypotheses = ["The dog bit the man.", "It was unexpected."]
references = [["The dog bit the man.", "It was not expected."]]

print(sacrebleu.corpus_bleu(hypotheses, references))
print(sacrebleu.corpus_chrf(hypotheses, references))
print(sacrebleu.corpus_ter(hypotheses, references))
```

The compatibility functions have the same names and call signatures as
SacreBLEU 2.6.0. The object API is available too:

```python
from mojo_sacrebleu.metrics import BLEU, CHRF, TER

metric = BLEU(tokenize="13a", references=references)
score = metric.corpus_score(hypotheses, None)
print(score.score)
print(metric.get_signature())
```

## Coverage

| metric | covered behavior |
| --- | --- |
| BLEU | corpus and sentence scoring, multiple references, exact clipped n-grams, effective order, `none`/`floor`/`add-k`/`exp` smoothing, lowercase, custom maximum order |
| chrF | corpus and sentence chrF, chrF+, chrF++, custom character and word orders, beta, whitespace, lowercase, epsilon smoothing |
| TER | corpus and sentence TER, multiple references, shifts, normalization, punctuation removal, Asian support, case sensitivity |
| shared API | score objects and formatting, signatures, reference caching, bootstrap confidence estimates |

BLEU includes the self-contained `13a`, `none`, `char`, `intl`, and `zh`
tokenizers. The external-model tokenizers `ja-mecab`, `ko-mecab`, `spm`,
`flores101`, `flores200`, and `spBLEU-1K` are not included. SacreBLEU's
dataset download registry, command-line interface, and paired significance
testing are also outside this repository's scope.

The parity suite compares every covered metric directly with the real
`sacrebleu` package, including repeated n-grams, multi-reference selection,
Unicode, empty segments, TER shifts, and option combinations.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-sacrebleu.so`. The Python wrapper also
rebuilds a missing or stale library on first use. An already-built library can
be supplied with `MOJO_SACREBLEU_LIB=/path/to/libmojo-sacrebleu.so`.

The usage example can be run inside the environment with `pixi run python`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(Linux x86_64). Times are best of three complete API calls, including
tokenization, buffer preparation, and result construction.

| case | mojo-sacrebleu | sacrebleu | relative |
| --- | ---: | ---: | ---: |
| BLEU, 4k x 24 tokens | 66.0 ms | 423.8 ms | 6.42x faster |
| chrF, 800 x 28 tokens | 40.6 ms | 286.5 ms | 7.05x faster |
| chrF++, 800 x 28 tokens | 72.7 ms | 354.5 ms | 4.88x faster |
| TER substitutions, 80 x 26 tokens | 10.9 ms | 46.6 ms | 4.29x faster |
| TER shifts, 30 x 24 tokens | 37.9 ms | 479.0 ms | 12.64x faster |

Run `pixi run bench` to reproduce the table. TER shift search benefits most
because each candidate's dynamic-programming matrix is evaluated in compiled
Mojo rather than Python.

GPU acceleration is not included. The n-gram kernels are comparison- and
memory-dominated, while TER edit distance has serial row dependencies; none
has the arithmetic intensity needed to justify device transfer and launch
overhead.

## How it works

Python performs Unicode normalization and tokenization so public behavior
stays compatible with SacreBLEU. Tokens or characters are then assigned
integer IDs and packed into contiguous NumPy `int64` arrays. Large BLEU and
chrF corpora cross the FFI boundary in batches and are split into independent
rows with thresholded CPU parallelism; small inputs stay serial. The Mojo
kernels compute clipped BLEU counts, chrF n-gram intersections, and TER
edit-distance matrices. N-grams are compared exactly with SIMD plus scalar
tail handling; no probabilistic hashes or collision assumptions affect scores.

The shared library exposes non-parametric `@export` functions using
`abi("C")`. NumPy owns every input, output, and scratch allocation. Buffers
cross `ctypes` as integer addresses and Mojo reconstructs them as
`UnsafePointer[..., AnyOrigin[mut=True]]`, so the FFI performs no serialization
and Mojo owns no cross-language memory. TER's shift candidate logic remains in
Python while its quadratic edit-distance work, including traceback, runs in
Mojo.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite currently contains 51 parity and boundary-safety cases against
SacreBLEU 2.6.0.

## License

MIT
