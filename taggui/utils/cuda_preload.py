"""Make pip-installed CUDA runtime libraries visible to llama-cpp-python.

The CUDA build of llama-cpp-python links against the CUDA runtime
(libcudart, libcublas, libcublasLt). In a pip environment these come from
the nvidia-* wheels under `<site-packages>/nvidia/cu<N>/`, not from a system
install. PyTorch's libraries find them through a baked-in RPATH, but
libllama.so has none, so `import llama_cpp` fails.

Loading the libraries by absolute path first means libllama.so's
dependencies resolve by soname against the already-loaded copies. Safe to
call when the libraries are missing (CPU build) or already on the loader
path (system CUDA install): it then does nothing.
"""
import os
import sys
from pathlib import Path

# Must match the CUDA major version of the llama-cpp-python wheel pinned in
# requirements.txt (the `cu132` extra-index-url, and the 13.x range of the
# `nvidia-cuda-runtime` / `nvidia-cublas` pins below it). If the wheel moves
# to another CUDA major, change this and those pins together.
CUDA_MAJOR = 13

# Dependency order: cublasLt must be loaded before cublas.
_LINUX_LIBS = (
    f'libcudart.so.{CUDA_MAJOR}',
    f'libcublasLt.so.{CUDA_MAJOR}',
    f'libcublas.so.{CUDA_MAJOR}',
)

_preloaded = False
_dll_dir_handles = []  # Windows: keep references to the added DLL directories


def _nvidia_cuda_dirs() -> list[Path]:
    try:
        import nvidia  # namespace package provided by the nvidia-* wheels
    except ImportError:
        return []
    return [Path(base) / f'cu{CUDA_MAJOR}' for base in nvidia.__path__
            if (Path(base) / f'cu{CUDA_MAJOR}').is_dir()]


def preload_cuda_libs() -> None:
    global _preloaded
    if _preloaded:
        return
    _preloaded = True

    for cuda_dir in _nvidia_cuda_dirs():
        if sys.platform == 'win32':
            bin_dir = cuda_dir / 'bin'
            if bin_dir.is_dir():
                _dll_dir_handles.append(os.add_dll_directory(str(bin_dir)))
            return
        if sys.platform.startswith('linux'):
            import ctypes
            lib_dir = cuda_dir / 'lib'
            for name in _LINUX_LIBS:
                path = lib_dir / name
                if not path.exists():
                    continue
                try:
                    ctypes.CDLL(str(path))
                except OSError as e:
                    print(f'Warning: failed to preload {path}: {e}',
                          file=sys.stderr)
            return
