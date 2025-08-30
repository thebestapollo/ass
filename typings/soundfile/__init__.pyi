# Minimal stub for soundfile to appease Pyright/Pylance
from typing import Any
from numpy.typing import NDArray

floating64 = Any

def read(file: Any, frames: int = ..., start: int = ..., stop: Any | None = ..., dtype: str = ..., always_2d: bool = ..., fill_value: Any | None = ..., out: Any | None = ..., samplerate: Any | None = ..., channels: Any | None = ..., format: Any | None = ..., subtype: Any | None = ..., endian: Any | None = ..., closefd: bool = ...) -> tuple[NDArray[Any], int]: ...

def write(file: Any, data: NDArray[Any], samplerate: int, subtype: Any | None = ..., endian: Any | None = ..., format: Any | None = ..., closefd: bool = ..., compression_level: Any | None = ..., bitrate_mode: Any | None = ...) -> None: ...
