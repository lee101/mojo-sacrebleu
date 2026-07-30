from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest
import sacrebleu
from sacrebleu.metrics import BLEU as RefBLEU
from sacrebleu.metrics import CHRF as RefCHRF
from sacrebleu.metrics import TER as RefTER

import mojo_sacrebleu as mojo
from mojo_sacrebleu._lib import addr, lib
from mojo_sacrebleu.metrics import BLEU, CHRF, TER
from mojo_sacrebleu.metrics.bleu import _PARALLEL_THRESHOLD as BLEU_PARALLEL_THRESHOLD
from mojo_sacrebleu.metrics.chrf import _PARALLEL_THRESHOLD as CHRF_PARALLEL_THRESHOLD


HYPOTHESES = [
    "The dog bit the man.",
    "It wasn't surprising.",
    "The man had just bitten him.",
]
REFERENCES = [
    [
        "The dog bit the man.",
        "It was not unexpected.",
        "The man bit him first.",
    ],
    [
        "The dog had bit the man.",
        "No one was surprised.",
        "The man had bitten the dog.",
    ],
]


def assert_score_equal(actual, expected, fields=()):
    assert actual.score == pytest.approx(expected.score, abs=1e-12)
    assert actual.name == expected.name
    assert repr(actual) == repr(expected)
    for field in fields:
        assert getattr(actual, field) == pytest.approx(getattr(expected, field), abs=1e-12)


def test_published_multi_reference_bleu_vector():
    actual = mojo.corpus_bleu(HYPOTHESES, REFERENCES)
    expected = sacrebleu.corpus_bleu(HYPOTHESES, REFERENCES)
    assert_score_equal(actual, expected, ("bp", "ratio", "sys_len", "ref_len"))
    assert actual.counts == expected.counts == [14, 7, 5, 3]
    assert actual.totals == expected.totals == [17, 14, 11, 8]


@pytest.mark.parametrize("tokenize", ["none", "13a", "char", "intl", "zh"])
def test_bleu_tokenizer_parity(tokenize):
    hypotheses = ["Numbers 1,234.5 &amp; words.", "中文 text—works"]
    references = [["Numbers 1,234.5 & words!", "中文 text — works"]]
    assert_score_equal(
        mojo.corpus_bleu(hypotheses, references, tokenize=tokenize),
        sacrebleu.corpus_bleu(hypotheses, references, tokenize=tokenize),
        ("bp", "sys_len", "ref_len"),
    )


@pytest.mark.parametrize(
    ("method", "value"),
    [("none", None), ("exp", None), ("floor", 0.25), ("add-k", 2.0)],
)
def test_bleu_smoothing_parity(method, value):
    hypothesis = "a b c x"
    references = ["a b y z"]
    assert_score_equal(
        mojo.sentence_bleu(
            hypothesis, references, smooth_method=method, smooth_value=value
        ),
        sacrebleu.sentence_bleu(
            hypothesis, references, smooth_method=method, smooth_value=value
        ),
        ("bp",),
    )


def test_bleu_repeated_ngram_multi_reference_clipping():
    hypotheses = ["a a a b b a"]
    references = [["a a b b b a"], ["a b a a b a"]]
    actual = mojo.corpus_bleu(hypotheses, references, tokenize="none")
    expected = sacrebleu.corpus_bleu(hypotheses, references, tokenize="none")
    assert actual.counts == expected.counts
    assert actual.totals == expected.totals
    assert_score_equal(actual, expected)


def test_short_sentence_effective_order():
    for effective in (False, True):
        assert_score_equal(
            mojo.corpus_bleu(
                ["one token"],
                [["one token"]],
                tokenize="none",
                use_effective_order=effective,
            ),
            sacrebleu.corpus_bleu(
                ["one token"],
                [["one token"]],
                tokenize="none",
                use_effective_order=effective,
            ),
        )


def test_bleu_custom_order_and_cached_references():
    actual_metric = BLEU(max_ngram_order=2, references=REFERENCES)
    expected_metric = RefBLEU(max_ngram_order=2, references=REFERENCES)
    actual = actual_metric.corpus_score(HYPOTHESES, None)
    expected = expected_metric.corpus_score(HYPOTHESES, None)
    assert actual.counts == expected.counts
    assert_score_equal(actual, expected)


def test_raw_bleu_and_lowercase_parity():
    hypotheses = ["A B c"]
    references = [["a b C"]]
    assert_score_equal(
        mojo.raw_corpus_bleu(hypotheses, references),
        sacrebleu.raw_corpus_bleu(hypotheses, references),
    )
    assert_score_equal(
        mojo.corpus_bleu(hypotheses, references, lowercase=True, tokenize="none"),
        sacrebleu.corpus_bleu(
            hypotheses, references, lowercase=True, tokenize="none"
        ),
    )


@pytest.mark.parametrize(
    ("char_order", "word_order", "beta", "whitespace", "eps", "lowercase"),
    [
        (6, 0, 2, False, False, False),
        (4, 0, 1, True, False, False),
        (6, 2, 2, False, False, False),
        (3, 1, 3, True, True, True),
    ],
)
def test_chrf_option_matrix(char_order, word_order, beta, whitespace, eps, lowercase):
    actual_metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=whitespace,
        eps_smoothing=eps,
        lowercase=lowercase,
    )
    expected_metric = RefCHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=whitespace,
        eps_smoothing=eps,
        lowercase=lowercase,
    )
    assert_score_equal(
        actual_metric.corpus_score(HYPOTHESES, REFERENCES),
        expected_metric.corpus_score(HYPOTHESES, REFERENCES),
    )


def test_chrf_unicode_and_repeated_characters():
    hypotheses = ["naïve café 漢字", "aaaaab"]
    references = [["naive café 漢字", "aaaabb"], ["naïve cafe 漢", "baaaaa"]]
    for word_order in (0, 2):
        assert_score_equal(
            mojo.corpus_chrf(hypotheses, references, word_order=word_order),
            sacrebleu.corpus_chrf(hypotheses, references, word_order=word_order),
        )


@pytest.mark.parametrize("length", [3, 4, 5, 6, 9, 10])
def test_chrf_simd_tail_parity(length):
    hypothesis = ("abcde" * 3)[:length]
    reference = ("abced" * 3)[:length]
    actual = CHRF(char_order=9).sentence_score(hypothesis, [reference])
    expected = RefCHRF(char_order=9).sentence_score(hypothesis, [reference])
    assert_score_equal(actual, expected)


@pytest.mark.parametrize(
    ("metric", "reference_metric", "threshold"),
    [
        (BLEU(tokenize="none"), RefBLEU(tokenize="none"), BLEU_PARALLEL_THRESHOLD),
        (CHRF(), RefCHRF(), CHRF_PARALLEL_THRESHOLD),
    ],
)
def test_batch_serial_and_parallel_threshold_parity(
    metric, reference_metric, threshold
):
    for size in (threshold - 1, threshold):
        hypotheses = [f"row {index} has a repeated token token" for index in range(size)]
        references = [
            [f"row {index} has one repeated token token" for index in range(size)]
        ]
        assert_score_equal(
            metric.corpus_score(hypotheses, references),
            reference_metric.corpus_score(hypotheses, references),
        )


def test_chrf_best_reference_rounding_tie():
    hypothesis = "f . a e"
    references = ["d b a", "f d f f e a d a"]
    assert_score_equal(
        mojo.sentence_chrf(hypothesis, references),
        sacrebleu.sentence_chrf(hypothesis, references),
    )


@pytest.mark.parametrize(
    ("hypothesis", "reference"),
    [
        ("a b c d e", "d e a b c"),
        ("the cat sat on the mat", "on the mat the cat sat"),
        ("x a b c y a b c z", "a b c x y z a b c"),
        ("a b c", "a x c"),
        ("", ""),
        ("some words", ""),
    ],
)
def test_ter_shift_and_edit_parity(hypothesis, reference):
    assert_score_equal(
        mojo.sentence_ter(hypothesis, [reference]),
        sacrebleu.sentence_ter(hypothesis, [reference]),
        ("num_edits", "ref_length"),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"normalized": True},
        {"no_punct": True},
        {"case_sensitive": True},
        {"normalized": True, "asian_support": True},
        {"no_punct": True, "asian_support": True},
    ],
)
def test_ter_tokenization_options(options):
    hypotheses = ["John's 2-cats, RUN!", "漢字。テスト"]
    references = [["john 's 2 cats run", "漢 字 テスト"]]
    assert_score_equal(
        mojo.corpus_ter(hypotheses, references, **options),
        sacrebleu.corpus_ter(hypotheses, references, **options),
        ("num_edits", "ref_length"),
    )


def test_ter_multiple_references_and_cache():
    actual_metric = TER(references=REFERENCES)
    expected_metric = RefTER(references=REFERENCES)
    assert_score_equal(
        actual_metric.corpus_score(HYPOTHESES, None),
        expected_metric.corpus_score(HYPOTHESES, None),
        ("num_edits", "ref_length"),
    )


@pytest.mark.parametrize(
    ("ours", "upstream"),
    [
        (mojo.corpus_bleu, sacrebleu.corpus_bleu),
        (mojo.sentence_bleu, sacrebleu.sentence_bleu),
        (mojo.corpus_chrf, sacrebleu.corpus_chrf),
        (mojo.sentence_chrf, sacrebleu.sentence_chrf),
        (mojo.corpus_ter, sacrebleu.corpus_ter),
        (mojo.sentence_ter, sacrebleu.sentence_ter),
    ],
)
def test_compat_function_signatures(ours, upstream):
    ours_params = inspect.signature(ours).parameters
    upstream_params = inspect.signature(upstream).parameters
    assert list(ours_params) == list(upstream_params)
    assert [item.default for item in ours_params.values()] == [
        item.default for item in upstream_params.values()
    ]


def test_score_format_json_and_signature_shape():
    metric = BLEU()
    score = metric.corpus_score(HYPOTHESES, REFERENCES)
    signature = str(metric.get_signature())
    assert signature.startswith("nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp")
    payload = json.loads(score.format(signature=signature, is_json=True))
    assert payload["name"] == "BLEU"
    assert payload["score"] == pytest.approx(48.53)


def test_bootstrap_confidence_and_signature():
    metric = BLEU(tokenize="none")
    score = metric.corpus_score(HYPOTHESES, REFERENCES, n_bootstrap=10)
    assert score._mean >= 0
    assert score._ci >= 0
    assert "bs:10|seed:12345" in str(metric.get_signature())


def test_ffi_rejects_null_pointers_and_addr_rejects_wrong_dtype():
    assert lib().msb_pair_ngram_stats(0, 0, 0, 0, 1, 0, 3) == -1
    with pytest.raises(TypeError):
        addr(np.zeros(1, dtype=np.int32))


def test_invalid_and_mismatched_streams_raise():
    with pytest.raises(TypeError):
        BLEU().sentence_score(["not a string"], ["reference"])
    with pytest.raises(ValueError):
        BLEU().corpus_score(["a", "b"], [["a"]])


def test_scores_are_finite_on_empty_segments():
    for actual in (
        mojo.sentence_bleu("", [""]),
        mojo.sentence_chrf("", [""]),
        mojo.sentence_ter("", [""]),
    ):
        assert math.isfinite(actual.score)
