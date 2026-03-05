import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, Optional, Tuple

import cv2
import numpy as np

from utils.transform import apply_transform, cut_patch, warp_img

from .video_stream import VideoFileStream, VideoFrame


@dataclass
class MouthRoiStats:
    emitted_frames: int
    dropped_frames: int
    avg_process_ms: float
    max_process_ms: float
    feature_shape: Tuple[int, int]


class MouthROIExtractor:
    """Streaming mouth ROI extractor using per-frame landmarks."""

    STD_SIZE = (256, 256)
    STABLE_POINTS = [33, 36, 39, 42, 45]

    def __init__(
        self,
        landmark_npz: str,
        mean_face_path: str = "utils/20words_mean_face.npy",
        crop_width: int = 96,
        crop_height: int = 96,
        start_idx: int = 48,
        stop_idx: int = 68,
        window_margin: int = 12,
        convert_gray: bool = True,
    ) -> None:
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.start_idx = int(start_idx)
        self.stop_idx = int(stop_idx)
        self.window_margin = int(window_margin)
        self.convert_gray = bool(convert_gray)

        self.mean_face = np.load(mean_face_path).astype(np.float32)
        self.landmarks = self._load_landmarks(landmark_npz)

        self._q_frame: Deque[np.ndarray] = deque()
        self._q_landmark: Deque[np.ndarray] = deque()
        self._q_idx_ts: Deque[Tuple[int, float]] = deque()

        self._last_transform = None
        self._emitted_frames = 0
        self._dropped_frames = 0
        self._proc_ms = []

    def _load_landmarks(self, landmark_npz: str) -> np.ndarray:
        raw = np.load(landmark_npz, allow_pickle=True)["data"]

        # Normalize to (T, 68, 2) float32
        if raw.ndim == 4 and raw.shape[1] == 1 and raw.shape[2] >= self.stop_idx:
            lm = raw[:, 0, :, :]
            return lm.astype(np.float32)

        out = []
        last_valid = None
        for item in raw:
            cur = None
            if isinstance(item, np.ndarray):
                if item.ndim == 2 and item.shape[1] == 2 and item.shape[0] >= self.stop_idx:
                    cur = item
                elif item.ndim == 3 and item.shape[0] > 0 and item.shape[1] >= self.stop_idx:
                    cur = item[0]
            if cur is None:
                if last_valid is None:
                    continue
                cur = last_valid
            cur = cur.astype(np.float32)
            out.append(cur)
            last_valid = cur

        if not out:
            raise RuntimeError(f"no valid landmarks in {landmark_npz}")
        return np.stack(out, axis=0)

    def _extract_one(self, frame: np.ndarray, landmark: np.ndarray, smoothed_landmark: np.ndarray) -> Optional[np.ndarray]:
        t0 = time.time()
        try:
            trans_frame, trans = warp_img(
                smoothed_landmark[self.STABLE_POINTS, :],
                self.mean_face[self.STABLE_POINTS, :],
                frame,
                self.STD_SIZE,
            )
            self._last_transform = trans
            trans_landmark = trans(landmark)
            patch = cut_patch(
                trans_frame,
                trans_landmark[self.start_idx : self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            if self.convert_gray:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            dt_ms = (time.time() - t0) * 1000.0
            self._proc_ms.append(dt_ms)
            return patch
        except Exception:
            self._dropped_frames += 1
            return None

    def process(self, frame: np.ndarray, frame_idx: int, timestamp_s: float) -> Optional[VideoFrame]:
        if frame_idx >= self.landmarks.shape[0]:
            lm = self.landmarks[-1]
        else:
            lm = self.landmarks[frame_idx]

        self._q_frame.append(frame)
        self._q_landmark.append(lm)
        self._q_idx_ts.append((frame_idx, timestamp_s))

        if len(self._q_frame) < self.window_margin:
            return None

        smoothed = np.mean(np.stack(list(self._q_landmark), axis=0), axis=0)
        cur_frame = self._q_frame.popleft()
        cur_landmark = self._q_landmark.popleft()
        out_idx, out_ts = self._q_idx_ts.popleft()

        patch = self._extract_one(cur_frame, cur_landmark, smoothed)
        if patch is None:
            return None

        self._emitted_frames += 1
        return VideoFrame(data=patch, index=out_idx, timestamp_s=out_ts)

    def flush(self) -> Iterator[VideoFrame]:
        while self._q_frame:
            cur_frame = self._q_frame.popleft()
            cur_landmark = self._q_landmark.popleft()
            out_idx, out_ts = self._q_idx_ts.popleft()

            if self._last_transform is None:
                smoothed = cur_landmark
                patch = self._extract_one(cur_frame, cur_landmark, smoothed)
            else:
                try:
                    t0 = time.time()
                    trans_frame = apply_transform(self._last_transform, cur_frame, self.STD_SIZE)
                    trans_landmark = self._last_transform(cur_landmark)
                    patch = cut_patch(
                        trans_frame,
                        trans_landmark[self.start_idx : self.stop_idx],
                        self.crop_height // 2,
                        self.crop_width // 2,
                    )
                    if self.convert_gray:
                        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    self._proc_ms.append((time.time() - t0) * 1000.0)
                except Exception:
                    self._dropped_frames += 1
                    patch = None

            if patch is not None:
                self._emitted_frames += 1
                yield VideoFrame(data=patch, index=out_idx, timestamp_s=out_ts)

    @property
    def stats(self) -> MouthRoiStats:
        if self._proc_ms:
            avg_ms = float(np.mean(self._proc_ms))
            max_ms = float(np.max(self._proc_ms))
        else:
            avg_ms = 0.0
            max_ms = 0.0
        return MouthRoiStats(
            emitted_frames=self._emitted_frames,
            dropped_frames=self._dropped_frames,
            avg_process_ms=avg_ms,
            max_process_ms=max_ms,
            feature_shape=(self.crop_height, self.crop_width),
        )


class OnlineMouthRoiStream:
    """Video frame stream -> streaming mouth ROI feature stream."""

    def __init__(
        self,
        video_path: str,
        landmark_npz: str,
        fps: Optional[float] = None,
        realtime: bool = False,
        loop: bool = False,
        mean_face_path: str = "utils/20words_mean_face.npy",
        crop_width: int = 96,
        crop_height: int = 96,
        start_idx: int = 48,
        stop_idx: int = 68,
        window_margin: int = 12,
    ) -> None:
        self.video_stream = VideoFileStream(video_path, fps=fps, realtime=realtime)
        self.loop = bool(loop)
        self.extractor = MouthROIExtractor(
            landmark_npz=landmark_npz,
            mean_face_path=mean_face_path,
            crop_width=crop_width,
            crop_height=crop_height,
            start_idx=start_idx,
            stop_idx=stop_idx,
            window_margin=window_margin,
            convert_gray=True,
        )
        self._emitted_frames = 0

    def __iter__(self) -> Iterator[VideoFrame]:
        self._emitted_frames = 0
        while True:
            for vf in self.video_stream:
                out = self.extractor.process(vf.data, vf.index, vf.timestamp_s)
                if out is not None:
                    self._emitted_frames += 1
                    yield out

            for out in self.extractor.flush():
                self._emitted_frames += 1
                yield out

            if not self.loop:
                break

    @property
    def emitted_frames(self) -> int:
        return self._emitted_frames

    @property
    def stats(self) -> MouthRoiStats:
        return self.extractor.stats
