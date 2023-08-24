from collections.abc import MutableSequence
from pathlib import Path
from typing import Any, Tuple, Union


def is_gguf(path: Union[str, Path]) -> bool:
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        magic = f.read(4)
    return magic == "GGUF".encode()


class Vector(MutableSequence):
    """Provides a Python list-like interface for a C array to access and modify
    data in-place without copying the entire data between C and Python.
    """

    def __init__(self, data: Any, size: int):
        self._data = data
        self._size = size

    def __getitem__(self, index: int) -> Any:
        self._validate_index(index)
        return self._data[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._validate_index(index)
        self._data[index] = value

    def __len__(self) -> int:
        return self._size

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int):
            raise TypeError("list index must be integer")
        if not 0 <= index < self._size:
            raise IndexError("list index out of range")

    def __delitem__(self, index: int) -> None:
        raise NotImplementedError("This operation is not allowed.")

    def insert(self, index: int, value: Any) -> None:
        raise NotImplementedError("This operation is not allowed.")


def utf8_is_continuation_byte(byte: int) -> bool:
    """Checks if a byte is a UTF-8 continuation byte (most significant bit is 1)."""
    return (byte & 0b10000000) != 0


def utf8_split_incomplete(seq: bytes) -> Tuple[bytes, bytes]:
    """Splits a sequence of UTF-8 encoded bytes into complete and incomplete bytes."""
    i = len(seq)
    while i > 0 and utf8_is_continuation_byte(seq[i - 1]):
        i -= 1
    return seq[:i], seq[i:]
