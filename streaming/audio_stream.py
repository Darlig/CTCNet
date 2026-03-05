import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    index: int
    start_sample: int
    end_sample: int
    chunk_samples: int
    hop_samples: int

    @property
    def timestamp_s(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_timestamp_s(self) -> float:
        return self.end_sample / self.sample_rate


class AudioFileStream:
    """Read an audio file as a simulated realtime chunk stream."""

    def __init__(
        self,
        audio_path: str,
        sample_rate: int,
        chunk_ms: int = 80,
        hop_ms: Optional[int] = None,
        realtime: bool = False,
        loop: bool = False,
        max_duration_s: Optional[float] = None,
    ) -> None:
        self.audio_path = audio_path
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.hop_ms = int(hop_ms if hop_ms is not None else chunk_ms)
        self.realtime = bool(realtime)
        self.loop = bool(loop)
        self.max_duration_s = max_duration_s

        self.chunk_samples = max(1, int(round(self.sample_rate * self.chunk_ms / 1000.0)))
        self.hop_samples = max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))

        if self.hop_samples > self.chunk_samples:
            raise ValueError("hop_ms must be <= chunk_ms for stream simulation")

        self._audio = self._load_audio()
        self.total_samples = int(self._audio.shape[0])

        self._emitted_chunks = 0
        self._emitted_samples = 0
        self._start_wall = None

    def _load_audio(self) -> np.ndarray:
        audio, sr = sf.read(self.audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != self.sample_rate:
            raise ValueError(
                f"audio sample rate mismatch: file={sr}, expected={self.sample_rate}. "
                "Please provide a matching file."
            )
        return audio.astype(np.float32)

    def __iter__(self) -> Iterator[AudioChunk]:
        self._emitted_chunks = 0
        self._emitted_samples = 0
        self._start_wall = time.time()

        start = 0
        loop_count = 0
        while True:
            if self.max_duration_s is not None:
                max_samples = int(self.max_duration_s * self.sample_rate)
                if self._emitted_samples >= max_samples:
                    break

            if start >= self.total_samples:
                if not self.loop:
                    break
                loop_count += 1
                start = 0

            end = min(start + self.chunk_samples, self.total_samples)
            chunk = self._audio[start:end]

            if chunk.shape[0] < self.chunk_samples:
                pad = np.zeros(self.chunk_samples - chunk.shape[0], dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            global_start = start + loop_count * self.total_samples
            global_end = global_start + self.chunk_samples

            if self.realtime:
                target_elapsed = global_start / self.sample_rate
                now_elapsed = time.time() - self._start_wall
                sleep_s = target_elapsed - now_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            yield AudioChunk(
                data=chunk,
                sample_rate=self.sample_rate,
                index=self._emitted_chunks,
                start_sample=global_start,
                end_sample=global_end,
                chunk_samples=self.chunk_samples,
                hop_samples=self.hop_samples,
            )

            self._emitted_chunks += 1
            self._emitted_samples += self.hop_samples
            start += self.hop_samples

    @property
    def emitted_chunks(self) -> int:
        return self._emitted_chunks

    @property
    def emitted_seconds(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return self._emitted_samples / self.sample_rate
