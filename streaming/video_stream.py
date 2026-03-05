import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from nichang.datas.transform import get_preprocessing_pipelines


@dataclass
class VideoFrame:
    data: np.ndarray
    index: int
    timestamp_s: float


class VideoFileStream:
    """Read frames from a video file as a simulated stream."""

    def __init__(self, video_path: str, fps: Optional[float] = None, realtime: bool = False) -> None:
        self.video_path = video_path
        self.realtime = bool(realtime)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.fps = float(fps if fps is not None and fps > 0 else src_fps)
        if self.fps <= 0:
            raise ValueError("invalid FPS from video and override not provided")

        self._emitted_frames = 0

    def __iter__(self) -> Iterator[VideoFrame]:
        self._emitted_frames = 0
        start_wall = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.video_path}")

        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                ts = idx / self.fps
                if self.realtime:
                    target_elapsed = ts
                    now_elapsed = time.time() - start_wall
                    sleep_s = target_elapsed - now_elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                yield VideoFrame(data=frame, index=idx, timestamp_s=ts)
                idx += 1
                self._emitted_frames = idx
        finally:
            cap.release()

    @property
    def emitted_frames(self) -> int:
        return self._emitted_frames


class MouthRoiFileStream:
    """Read precomputed mouth ROI npz as a simulated frame stream."""

    def __init__(self, npz_path: str, fps: float = 25.0, realtime: bool = False, loop: bool = False) -> None:
        self.npz_path = npz_path
        self.fps = float(fps)
        self.realtime = bool(realtime)
        self.loop = bool(loop)
        if self.fps <= 0:
            raise ValueError("fps must be positive")

        raw = np.load(npz_path)["data"]
        self.frames = get_preprocessing_pipelines()["val"](raw).astype(np.float32)
        self._emitted_frames = 0

    def __iter__(self) -> Iterator[VideoFrame]:
        self._emitted_frames = 0
        start_wall = time.time()

        idx = 0
        n = self.frames.shape[0]
        while True:
            ts = idx / self.fps
            frame_idx = idx % n

            if self.realtime:
                target_elapsed = ts
                now_elapsed = time.time() - start_wall
                sleep_s = target_elapsed - now_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            yield VideoFrame(data=self.frames[frame_idx], index=idx, timestamp_s=ts)
            idx += 1
            self._emitted_frames = idx

            if not self.loop and idx >= n:
                break

    @property
    def emitted_frames(self) -> int:
        return self._emitted_frames
