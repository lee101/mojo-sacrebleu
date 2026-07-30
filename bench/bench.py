"""End-to-end metric benchmarks against SacreBLEU."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_sacrebleu as mojo  # noqa: E402
import sacrebleu  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def corpus(size: int, words: int, seed: int = 0):
    rng = random.Random(seed)
    vocabulary = [f"w{index}" for index in range(300)]
    hypotheses = []
    references = []
    for _ in range(size):
        reference = rng.choices(vocabulary, k=words)
        hypothesis = reference.copy()
        for index in range(0, words, 7):
            hypothesis[index] = rng.choice(vocabulary)
        references.append(" ".join(reference))
        hypotheses.append(" ".join(hypothesis))
    return hypotheses, [references]


def shifted_corpus(size: int, words: int):
    references = []
    hypotheses = []
    for row in range(size):
        tokens = [f"w{(row * 11 + index) % 100}" for index in range(words)]
        split = words // 3
        references.append(" ".join(tokens))
        hypotheses.append(" ".join(tokens[split:] + tokens[:split]))
    return hypotheses, [references]


def machine() -> str:
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model or platform.machine()} ({platform.system()} {platform.machine()})"


def main() -> None:
    bleu_data = corpus(4_000, 24, 1)
    chrf_data = corpus(800, 28, 2)
    ter_data = corpus(80, 26, 3)
    shift_data = shifted_corpus(30, 24)
    cases = [
        (
            "BLEU, 4k x 24 tokens",
            lambda: mojo.corpus_bleu(*bleu_data, tokenize="none"),
            lambda: sacrebleu.corpus_bleu(*bleu_data, tokenize="none"),
        ),
        (
            "chrF, 800 x 28 tokens",
            lambda: mojo.corpus_chrf(*chrf_data),
            lambda: sacrebleu.corpus_chrf(*chrf_data),
        ),
        (
            "chrF++, 800 x 28 tokens",
            lambda: mojo.corpus_chrf(*chrf_data, word_order=2),
            lambda: sacrebleu.corpus_chrf(*chrf_data, word_order=2),
        ),
        (
            "TER substitutions, 80 x 26 tokens",
            lambda: mojo.corpus_ter(*ter_data),
            lambda: sacrebleu.corpus_ter(*ter_data),
        ),
        (
            "TER shifts, 30 x 24 tokens",
            lambda: mojo.corpus_ter(*shift_data),
            lambda: sacrebleu.corpus_ter(*shift_data),
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-sacrebleu | sacrebleu | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases:
        ours()
        upstream()
        mojo_time = timeit(ours)
        upstream_time = timeit(upstream)
        relative = upstream_time / mojo_time
        label = "faster" if relative >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.1f} ms | "
            f"{upstream_time * 1e3:.1f} ms | {relative:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
