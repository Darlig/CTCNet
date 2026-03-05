import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import yaml

from nichang.models.ctcnet import CTCNet
from nichang.videomodels import VideoModel

from .audio_stream import AudioChunk
from .video_stream import VideoFrame


@dataclass
class EngineStats:
    processed_chunks: int
    emitted_chunks: int
    inference_avg_ms: float
    inference_p95_ms: float


@dataclass
class OutputChunk:
    data: np.ndarray
    sample_rate: int
    start_sample: int
    end_sample: int


class StreamingSeparator:
    """Chunk-by-chunk streaming wrapper around offline CTCNet + VideoModel."""

    def __init__(
        self,
        conf_path: str,
        checkpoint_path: str,
        device: str = "auto",
        window_ms: int = 2000,
        lookahead_ms: int = 0,
    ) -> None:
        self.conf_path = conf_path
        self.checkpoint_path = checkpoint_path
        self.window_ms = int(window_ms)
        self.lookahead_ms = int(lookahead_ms)

        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        conf["audionet"].update({"n_src": 1})

        self.sample_rate = int(conf["data"]["sample_rate"])
        self.window_samples = max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))
        self.lookahead_samples = max(0, int(round(self.sample_rate * self.lookahead_ms / 1000.0)))

        model_device = self._resolve_device(device)
        self.device = model_device

        self.audiomodel = CTCNet(sample_rate=self.sample_rate, **conf["audionet"])
        state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        self.audiomodel.load_state_dict(state)

        self.videomodel = VideoModel(**conf["videonet"])

        self.audiomodel.to(self.device).eval()
        self.videomodel.to(self.device).eval()

        self._audio_segments: Deque[Tuple[int, int, np.ndarray]] = deque()
        self._video_frames: Deque[VideoFrame] = deque()

        self._processed_chunks = 0
        self._emitted_chunks = 0
        self._latency_ms: List[float] = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _append_audio(self, chunk: AudioChunk) -> None:
        self._audio_segments.append((chunk.start_sample, chunk.end_sample, chunk.data.astype(np.float32)))

        keep_from = chunk.end_sample - (self.window_samples + self.lookahead_samples + 2 * chunk.chunk_samples)
        while self._audio_segments and self._audio_segments[0][1] < keep_from:
            self._audio_segments.popleft()

    def _append_video(self, frames: List[VideoFrame], chunk_end_ts: float) -> None:
        for frame in frames:
            self._video_frames.append(frame)

        min_keep_ts = chunk_end_ts - ((self.window_samples + self.lookahead_samples) / self.sample_rate + 2.0)
        while self._video_frames and self._video_frames[0].timestamp_s < min_keep_ts:
            self._video_frames.popleft()

    def _slice_audio(self, start_sample: int, end_sample: int) -> np.ndarray:
        out = np.zeros(max(0, end_sample - start_sample), dtype=np.float32)
        if out.size == 0:
            return out

        for seg_start, seg_end, seg_data in self._audio_segments:
            inter_start = max(seg_start, start_sample)
            inter_end = min(seg_end, end_sample)
            if inter_end <= inter_start:
                continue
            dst0 = inter_start - start_sample
            src0 = inter_start - seg_start
            length = inter_end - inter_start
            out[dst0 : dst0 + length] = seg_data[src0 : src0 + length]
        return out

    def _slice_video(self, start_ts: float, end_ts: float) -> Optional[np.ndarray]:
        frames = [f.data for f in self._video_frames if start_ts <= f.timestamp_s < end_ts]
        if not frames and self._video_frames:
            older = [f.data for f in self._video_frames if f.timestamp_s < end_ts]
            if older:
                frames = [older[-1]]

        if not frames:
            return None

        arr = np.stack(frames, axis=0).astype(np.float32)
        return arr

    def process(self, chunk: AudioChunk, aligned_video_frames: List[VideoFrame]) -> Optional[OutputChunk]:
        self._processed_chunks += 1

        self._append_audio(chunk)
        self._append_video(aligned_video_frames, chunk.end_timestamp_s)

        # Warmup until enough context is available.
        need_start = max(0, chunk.end_sample - self.lookahead_samples - self.window_samples)
        need_end = chunk.end_sample - self.lookahead_samples
        if need_end <= need_start:
            return None

        audio_window = self._slice_audio(need_start, need_end)
        if audio_window.shape[0] < self.window_samples:
            return None

        video_start_ts = need_start / self.sample_rate
        video_end_ts = need_end / self.sample_rate
        video_window = self._slice_video(video_start_ts, video_end_ts)
        if video_window is None:
            return None

        hop = chunk.hop_samples
        t0 = time.time()
        with torch.no_grad():
            wav_t = torch.from_numpy(audio_window).to(self.device).unsqueeze(0).unsqueeze(0)
            mouth_t = torch.from_numpy(video_window).to(self.device).unsqueeze(0).unsqueeze(1)
            mouth_emb = self.videomodel(mouth_t.float())
            est = self.audiomodel(wav_t, mouth_emb).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        dt_ms = (time.time() - t0) * 1000.0
        self._latency_ms.append(dt_ms)

        est = est[-audio_window.shape[0] :]
        out = est[-hop:]

        out_start = need_end - hop
        out_end = need_end
        self._emitted_chunks += 1
        return OutputChunk(data=out, sample_rate=self.sample_rate, start_sample=out_start, end_sample=out_end)

    @property
    def stats(self) -> EngineStats:
        if not self._latency_ms:
            avg = 0.0
            p95 = 0.0
        else:
            lat = np.array(self._latency_ms, dtype=np.float32)
            avg = float(lat.mean())
            p95 = float(np.percentile(lat, 95))

        return EngineStats(
            processed_chunks=self._processed_chunks,
            emitted_chunks=self._emitted_chunks,
            inference_avg_ms=avg,
            inference_p95_ms=p95,
        )
