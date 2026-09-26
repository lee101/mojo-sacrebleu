"""Builds and loads the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_SACREBLEU_LIB") or os.path.join(
    ROOT, "dist", "libmojo-sacrebleu.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "msb_bleu_stats": ([I] * 10, I),
    "msb_pair_ngram_stats": ([I] * 7, I),
    "msb_batch_pair_ngram_stats": ([I] * 9, I),
    "msb_batch_bleu_stats": ([I] * 11, I),
    "msb_edit_distance": ([I] * 11, I),
}


class BuildError(RuntimeError):
    pass


def _link_args() -> list[str]:
    """Record the AsyncRT runtime that ctypes resolves through this library."""
    mojo = shutil.which("mojo") or os.environ.get("MOJO_SACREBLEU_MOJO", "")
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(mojo))), "lib"),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib"),
    ]
    for lib_dir in candidates:
        if os.path.isfile(os.path.join(lib_dir, "libKGENCompilerRTShared.so")):
            return [
                "-Xlinker", "--no-as-needed",
                "-Xlinker", f"-L{lib_dir}",
                "-Xlinker", "-lKGENCompilerRTShared",
                "-Xlinker", "-lAsyncRTMojoBindings",
                "-Xlinker", "--as-needed",
            ]
    raise BuildError("cannot locate the Mojo toolchain lib directory")


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SACREBLEU_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_SACREBLEU_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_SACREBLEU_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ] + _link_args()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIB = None
_PARALLEL_DEVICE = None


def lib(*, parallel: bool = False) -> ctypes.CDLL:
    global _LIB, _PARALLEL_DEVICE
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = argtypes
            fn.restype = restype
    if parallel and _PARALLEL_DEVICE is None:
        init_parallel = _LIB.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        init_parallel.argtypes = []
        init_parallel.restype = ctypes.c_void_p
        _PARALLEL_DEVICE = init_parallel()
        if not _PARALLEL_DEVICE:
            raise RuntimeError("Mojo CPU parallel runtime initialization failed")
    return _LIB


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def pack_sequences(sequences) -> tuple[np.ndarray, np.ndarray]:
    vocabulary = {}
    flat = []
    offsets = np.empty(len(sequences) + 1, dtype=np.int64)
    offsets[0] = 0
    for index, sequence in enumerate(sequences):
        for item in sequence:
            token_id = vocabulary.get(item)
            if token_id is None:
                token_id = len(vocabulary) + 1
                vocabulary[item] = token_id
            flat.append(token_id)
        offsets[index + 1] = len(flat)
    return np.asarray(flat, dtype=np.int64), offsets


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.uint8)):
        raise TypeError("FFI buffers must have dtype int64 or uint8")
    return array.ctypes.data


def checked_call(function, *args) -> int:
    """Call a status-returning kernel and surface a rejected ABI contract."""
    status = int(function(*args))
    if status < 0:
        raise ValueError("Mojo kernel rejected invalid buffer metadata")
    return status


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
