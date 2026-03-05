from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf

from .engine import OutputChunk


@dataclass
class SinkStats:
    written_samples: int
    chunks: int


class WavChunkSink:
    """Collect output chunks and write a single wav file at finalize()."""

    def __init__(self, output_path: str, sample_rate: int, crossfade_ms: int = 5) -> None:
        self.output_path = output_path
        self.sample_rate = int(sample_rate)
        self.crossfade_samples = max(0, int(round(self.sample_rate * crossfade_ms / 1000.0)))

        self._segments: List[np.ndarray] = []
        self._written_samples = 0
        self._chunks = 0

    def write(self, chunk: OutputChunk) -> None:
        x = chunk.data.astype(np.float32)
        if x.size == 0:
            return

        if self._segments and self.crossfade_samples > 0:
            prev = self._segments[-1]
            k = min(self.crossfade_samples, prev.size, x.size)
            if k > 0:
                fade_out = np.linspace(1.0, 0.0, k, endpoint=False, dtype=np.float32)
                fade_in = 1.0 - fade_out
                blended = prev[-k:] * fade_out + x[:k] * fade_in
                self._segments[-1] = np.concatenate([prev[:-k], blended], axis=0)
                x = x[k:]

        if x.size > 0:
            self._segments.append(x)
            self._written_samples += x.size
        self._chunks += 1

    def finalize(self) -> None:
        if not self._segments:
            sf.write(self.output_path, np.zeros(1, dtype=np.float32), self.sample_rate)
            return
        out = np.concatenate(self._segments, axis=0)
        sf.write(self.output_path, out, self.sample_rate)

    @property
    def stats(self) -> SinkStats:
        return SinkStats(written_samples=self._written_samples, chunks=self._chunks)
