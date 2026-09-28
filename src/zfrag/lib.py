from __future__ import annotations

import ctypes
import platform
from ctypes import c_char_p, c_float, c_size_t, c_uint32, c_uint64, c_void_p
from importlib.resources import files
from pathlib import Path


def _resolve_binary_path() -> Path:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        filename = "frag.dll"
    elif system == "Linux":
        filename = "libfrag.so"
    elif system == "Darwin":
        filename = "libfrag.dylib"
    else:
        raise OSError(f"Unsupported platform: {system} ({machine})")

    binary_resource = files("zfrag").joinpath("lib", filename)
    binary_path = Path(str(binary_resource))

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Prebuilt binary '{filename}' was not found at {binary_path}"
        )

    return binary_path


_lib_path = _resolve_binary_path()
lib = ctypes.CDLL(str(_lib_path))

lib.frag_model_create.restype = c_void_p
lib.frag_model_destroy.argtypes = [c_void_p]

lib.frag_train.argtypes = [c_void_p, c_char_p, c_size_t]
lib.frag_sign.argtypes = [c_void_p, c_char_p, c_size_t]
lib.frag_sign.restype = c_uint64

lib.frag_distance.argtypes = [c_uint64, c_uint64]
lib.frag_distance.restype = c_uint32

lib.frag_similarity.argtypes = [c_uint64, c_uint64]
lib.frag_similarity.restype = c_float


class Model:
    def __init__(self) -> None:
        self._ptr = lib.frag_model_create()
        if not self._ptr:
            raise MemoryError("Failed to allocate Frag Model")

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            lib.frag_model_destroy(self._ptr)
            self._ptr = None

    def train(self, text: str) -> None:
        data = text.encode("utf-8")
        lib.frag_train(self._ptr, data, len(data))

    def sign(self, text: str) -> int:
        data = text.encode("utf-8")
        return lib.frag_sign(self._ptr, data, len(data))

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    @staticmethod
    def similarity(a: int, b: int) -> float:
        return lib.frag_similarity(a, b)
