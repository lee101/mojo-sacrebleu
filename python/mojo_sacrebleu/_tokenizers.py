"""Tokenizers compatible with SacreBLEU's Apache-2.0 implementations."""

from __future__ import annotations

import re
from functools import lru_cache

import regex

_POST_RE = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def _post_tokenize(line: str) -> str:
    for pattern, replacement in _POST_RE:
        line = pattern.sub(replacement, line)
    return " ".join(line.split())


class NoneTokenizer:
    signature = "none"

    def __call__(self, line: str) -> str:
        return line


class Tokenizer13a:
    signature = "13a"

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> str:
        line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
        if "&" in line:
            line = (
                line.replace("&quot;", '"')
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
            )
        return _post_tokenize(f" {line} ")


class CharTokenizer:
    signature = "char"

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> str:
        return " ".join(line)


class IntlTokenizer:
    signature = "intl"
    patterns = [
        (regex.compile(r"(\P{N})(\p{P})"), r"\1 \2 "),
        (regex.compile(r"(\p{P})(\P{N})"), r" \1 \2"),
        (regex.compile(r"(\p{S})"), r" \1 "),
    ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> str:
        for pattern, replacement in self.patterns:
            line = pattern.sub(replacement, line)
        return " ".join(line.split())


_ZH_RANGES = [
    (0x3400, 0x4DB5), (0x4E00, 0x9FA5), (0x9FA6, 0x9FBB),
    (0xF900, 0xFA2D), (0xFA30, 0xFA6A), (0xFA70, 0xFAD9),
    (0x2000, 0x2A6D), (0x2F80, 0x2FA1), (0xFF00, 0xFFEF),
    (0x2E80, 0x2EFF), (0x3000, 0x303F), (0x31C0, 0x31EF),
    (0x2F00, 0x2FDF), (0x2FF0, 0x2FFF), (0x3100, 0x312F),
    (0x31A0, 0x31BF), (0xFE10, 0xFE1F), (0xFE30, 0xFE4F),
    (0x2600, 0x26FF), (0x2700, 0x27BF), (0x3200, 0x32FF),
    (0x3300, 0x33FF),
]


class ZhTokenizer:
    signature = "zh"

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> str:
        chunks = []
        for char in line.strip():
            code = ord(char)
            chunks.append(f" {char} " if any(a <= code <= b for a, b in _ZH_RANGES) else char)
        return _post_tokenize("".join(chunks))


TOKENIZERS = {
    "none": NoneTokenizer,
    "13a": Tokenizer13a,
    "char": CharTokenizer,
    "intl": IntlTokenizer,
    "zh": ZhTokenizer,
}


_ASIAN_PUNCT = r"([\u3001\u3002\u3008-\u3011\u3014-\u301f\uff61-\uff65\u30fb])"
_FULL_WIDTH_PUNCT = r"([\uff0e\uff0c\uff1f\uff1a\uff1b\uff01\uff02\uff08\uff09])"


def ter_tokenize(
    sentence: str,
    normalized: bool,
    no_punct: bool,
    asian_support: bool,
    case_sensitive: bool,
) -> str:
    if not sentence:
        return ""
    if not case_sensitive:
        sentence = sentence.lower()
    if normalized:
        sentence = re.sub(r"\n-", "", sentence)
        sentence = sentence.replace("\n", " ")
        sentence = (
            sentence.replace("&quot;", '"')
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )
        sentence = f" {sentence} "
        sentence = re.sub(r"([{-~[-` -&(-+:-@/])", r" \1 ", sentence)
        sentence = re.sub(r"'s ", " 's ", sentence)
        sentence = re.sub(r"'s$", " 's", sentence)
        sentence = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", sentence)
        sentence = re.sub(r"([\.,])([^0-9])", r" \1 \2", sentence)
        sentence = re.sub(r"([0-9])(-)", r"\1 \2 ", sentence)
        if asian_support:
            sentence = re.sub(r"([\u4e00-\u9fff\u3400-\u4dbf])", r" \1 ", sentence)
            sentence = re.sub(r"([\u31c0-\u31ef\u2e80-\u2eff])", r" \1 ", sentence)
            sentence = re.sub(
                r"([\u3300-\u33ff\uf900-\ufaff\ufe30-\ufe4f])", r" \1 ", sentence
            )
            sentence = re.sub(r"([\u3200-\u3f22])", r" \1 ", sentence)
            sentence = re.sub(_ASIAN_PUNCT, r" \1 ", sentence)
            sentence = re.sub(_FULL_WIDTH_PUNCT, r" \1 ", sentence)
    if no_punct:
        sentence = re.sub(r'[\.,\?:;!"\(\)]', "", sentence)
        if asian_support:
            sentence = re.sub(_ASIAN_PUNCT, "", sentence)
            sentence = re.sub(_FULL_WIDTH_PUNCT, "", sentence)
    return " ".join(sentence.split())
