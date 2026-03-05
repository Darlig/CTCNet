from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

from .video_stream import VideoFrame


@dataclass
class SyncStats:
    queue_avg: float
    queue_max: int
    dropped_video_frames: int
    empty_align_count: int


class AVSynchronizer:
    """Timestamp-based alignment with bounded queue and drop policy."""

    def __init__(
        self,
        video_stream: Iterable[VideoFrame],
        max_queue_size: int = 200,
        av_offset_s: float = 0.0,
    ) -> None:
        self._iter = iter(video_stream)
        self.max_queue_size = int(max_queue_size)
        self.av_offset_s = float(av_offset_s)

        self._buffer: Deque[VideoFrame] = deque()
        self._next_frame: Optional[VideoFrame] = None
        self._video_done = False

        self._queue_sum = 0.0
        self._queue_count = 0
        self._queue_max = 0
        self._dropped = 0
        self._empty_align_count = 0

        self._prime_next()

    def _prime_next(self) -> None:
        if self._video_done:
            return
        try:
            self._next_frame = next(self._iter)
        except StopIteration:
            self._video_done = True
            self._next_frame = None

    def _push_frame(self, frame: VideoFrame) -> None:
        self._buffer.append(frame)
        if len(self._buffer) > self.max_queue_size:
            self._buffer.popleft()
            self._dropped += 1

    def fill_until(self, target_timestamp_s: float) -> None:
        target = target_timestamp_s + self.av_offset_s

        while self._next_frame is not None and self._next_frame.timestamp_s <= target:
            self._push_frame(self._next_frame)
            self._prime_next()

        qlen = len(self._buffer)
        self._queue_sum += qlen
        self._queue_count += 1
        self._queue_max = max(self._queue_max, qlen)

    def frames_for_interval(
        self,
        start_timestamp_s: float,
        end_timestamp_s: float,
        min_keep_timestamp_s: float,
    ) -> List[VideoFrame]:
        start = start_timestamp_s + self.av_offset_s
        end = end_timestamp_s + self.av_offset_s

        selected = [f for f in self._buffer if start <= f.timestamp_s < end]

        if not selected:
            older = [f for f in self._buffer if f.timestamp_s < end]
            if older:
                selected = [older[-1]]
            else:
                self._empty_align_count += 1

        while self._buffer and self._buffer[0].timestamp_s < min_keep_timestamp_s + self.av_offset_s:
            self._buffer.popleft()

        qlen = len(self._buffer)
        self._queue_sum += qlen
        self._queue_count += 1
        self._queue_max = max(self._queue_max, qlen)

        return selected

    @property
    def stats(self) -> SyncStats:
        avg = self._queue_sum / self._queue_count if self._queue_count else 0.0
        return SyncStats(
            queue_avg=avg,
            queue_max=self._queue_max,
            dropped_video_frames=self._dropped,
            empty_align_count=self._empty_align_count,
        )
