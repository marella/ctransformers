from collections.abc import MutableSequence
from typing import Any


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
