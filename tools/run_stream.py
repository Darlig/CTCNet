#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming.audio_stream import AudioFileStream
from streaming.engine import StreamingSeparator
from streaming.mouthroi_stream import OnlineMouthRoiStream
from streaming.sink import WavChunkSink
from streaming.sync import AVSynchronizer
from streaming.video_stream import MouthRoiFileStream


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run file-as-stream AV separation")
    p.add_argument("--conf", default="local/vox2_10w_conf_64_64_3_adamw_1e-1_blocks16_pretrain_test.yml")
    p.add_argument("--checkpoint", default="pretrained_model/vox2_best_model.pt")
    p.add_argument("--audio", default="test_videos/interview_visualvoice/interview.wav")
    p.add_argument("--video", default="test_videos/interview_visualvoice/faces/speaker1.mp4")
    p.add_argument("--landmark-npz", default="test_videos/interview_visualvoice/landmark/speaker1.npz")
    p.add_argument("--mouth-npz", default="test_videos/interview_visualvoice/mouthroi/speaker1.npz")
    p.add_argument("--mouthroi-mode", choices=["online", "precomputed"], default="online")
    p.add_argument("--output", default="outputs/stream_spk1.wav")

    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--chunk-ms", type=int, default=80)
    p.add_argument("--hop-ms", type=int, default=80)
    p.add_argument("--window-ms", type=int, default=2000)
    p.add_argument("--lookahead-ms", type=int, default=0)
    p.add_argument("--crossfade-ms", type=int, default=5)

    p.add_argument("--max-video-queue", type=int, default=200)
    p.add_argument("--av-offset-s", type=float, default=0.0)
    p.add_argument("--realtime", type=int, default=1)
    p.add_argument("--loop", type=int, default=0)
    p.add_argument("--max-duration-s", type=float, default=None)
    p.add_argument("--stats-interval-s", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    return p


def log_stats(prefix: str, start_wall: float, audio_stream, video_stream, sync, engine, sink) -> None:
    elapsed = max(1e-6, time.time() - start_wall)
    e = engine.stats
    s = sync.stats
    w = sink.stats
    m = getattr(video_stream, "stats", None)
    if m is not None:
        mouth_str = (
            f"mouth_avg={getattr(m, 'avg_process_ms', 0.0):7.2f}ms "
            f"mouth_max={getattr(m, 'max_process_ms', 0.0):7.2f}ms "
            f"mouth_drop={getattr(m, 'dropped_frames', 0):4d} "
        )
    else:
        mouth_str = "mouth_avg=   0.00ms mouth_max=   0.00ms mouth_drop=   0 "
    print(
        f"[{prefix}] elapsed={elapsed:7.2f}s "
        f"audio_chunks={audio_stream.emitted_chunks:6d} ({audio_stream.emitted_chunks/elapsed:6.2f}/s) "
        f"video_frames={video_stream.emitted_frames:6d} ({video_stream.emitted_frames/elapsed:6.2f}/s) "
        f"{mouth_str}"
        f"infer_avg={e.inference_avg_ms:7.2f}ms p95={e.inference_p95_ms:7.2f}ms "
        f"vq_avg={s.queue_avg:6.2f} vq_max={s.queue_max:4d} drops={s.dropped_video_frames:4d} empty_align={s.empty_align_count:4d} "
        f"out_chunks={w.chunks:6d} out_sec={w.written_samples/max(1, engine.sample_rate):7.3f}s"
    )


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    audio_stream = AudioFileStream(
        audio_path=args.audio,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        hop_ms=args.hop_ms,
        realtime=bool(args.realtime),
        loop=bool(args.loop),
        max_duration_s=args.max_duration_s,
    )

    if args.mouthroi_mode == "online":
        video_stream = OnlineMouthRoiStream(
            video_path=args.video,
            landmark_npz=args.landmark_npz,
            fps=args.fps,
            realtime=bool(args.realtime),
            loop=bool(args.loop),
        )
    else:
        video_stream = MouthRoiFileStream(
            npz_path=args.mouth_npz,
            fps=args.fps,
            realtime=bool(args.realtime),
            loop=bool(args.loop),
        )

    sync = AVSynchronizer(
        video_stream=video_stream,
        max_queue_size=args.max_video_queue,
        av_offset_s=args.av_offset_s,
    )

    engine = StreamingSeparator(
        conf_path=args.conf,
        checkpoint_path=args.checkpoint,
        device=args.device,
        window_ms=args.window_ms,
        lookahead_ms=args.lookahead_ms,
    )

    sink = WavChunkSink(args.output, sample_rate=engine.sample_rate, crossfade_ms=args.crossfade_ms)

    start_wall = time.time()
    next_log = start_wall + args.stats_interval_s

    for chunk in audio_stream:
        sync.fill_until(chunk.end_timestamp_s + args.lookahead_ms / 1000.0)
        keep_ts = chunk.end_timestamp_s - (args.window_ms / 1000.0 + 2.0)
        frames = sync.frames_for_interval(chunk.timestamp_s, chunk.end_timestamp_s, min_keep_timestamp_s=keep_ts)

        out = engine.process(chunk, frames)
        if out is not None:
            sink.write(out)

        now = time.time()
        if now >= next_log:
            log_stats("stream", start_wall, audio_stream, video_stream, sync, engine, sink)
            next_log = now + args.stats_interval_s

    sink.finalize()
    log_stats("done", start_wall, audio_stream, video_stream, sync, engine, sink)
    print(f"output_wav={args.output}")


if __name__ == "__main__":
    main()
