#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# The ctypes layer acquires an AsyncRT CPU device through this library
# (KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice), so the shared object must
# record and resolve the AsyncRT runtime. No compiled object references those
# symbols, so the linker would drop the DT_NEEDED entries under the default
# --as-needed; force them in and restore the default afterwards.
mojo_prefix="$(dirname "$(dirname "$(readlink -f "$(command -v mojo)")")")"

mojo build --emit shared-lib -I "$repo_dir/src" \
  "$repo_dir/src/capi.mojo" -o "$repo_dir/dist/libmojo-sacrebleu.so" \
  -Xlinker --no-as-needed \
  -Xlinker "-L${mojo_prefix}/lib" \
  -Xlinker -lKGENCompilerRTShared \
  -Xlinker -lAsyncRTMojoBindings \
  -Xlinker --as-needed
