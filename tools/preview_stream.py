#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming.audio_stream import AudioFileStream
from streaming.video_stream import MouthRoiFileStream, VideoFileStream


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Preview file-as-stream audio/video rates")
    p.add_argument("--audio", required=True)
    p.add_argument("--video", default=None, help="raw video path for frame stream preview")
    p.add_argument("--mouth-npz", default=None, help="npz path for mouth ROI stream preview")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-ms", type=int, default=80)
    p.add_argument("--hop-ms", type=int, default=80)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--realtime", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()

    audio_stream = AudioFileStream(
        audio_path=args.audio,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        hop_ms=args.hop_ms,
        realtime=bool(args.realtime),
        max_duration_s=args.seconds,
    )

    if args.mouth_npz:
        video_stream = MouthRoiFileStream(args.mouth_npz, fps=args.fps, realtime=bool(args.realtime))
        video_kind = "mouth_npz"
    elif args.video:
        video_stream = VideoFileStream(args.video, fps=args.fps, realtime=bool(args.realtime))
        video_kind = "video"
    else:
        raise ValueError("either --video or --mouth-npz is required")

    v_iter = iter(video_stream)

    start = time.time()
    a_count = 0
    v_count = 0

    for chunk in audio_stream:
        a_count += 1

        # Pull video up to current audio time for quick rate preview.
        while True:
            try:
                frame = next(v_iter)
            except StopIteration:
                frame = None
            if frame is None:
                break
            if frame.timestamp_s <= chunk.end_timestamp_s:
                v_count += 1
                continue
            break

        now = time.time()
        elapsed = max(1e-6, now - start)
        if a_count % 20 == 0:
            print(
                f"t={chunk.end_timestamp_s:7.3f}s "
                f"audio_rate={a_count/elapsed:6.2f} chunks/s "
                f"video_rate={v_count/elapsed:6.2f} fps "
                f"chunk_idx={chunk.index:5d}"
            )

    elapsed = max(1e-6, time.time() - start)
    print(
        "preview_done "
        f"mode={video_kind} "
        f"elapsed={elapsed:.2f}s "
        f"audio_chunks={a_count} ({a_count/elapsed:.2f}/s) "
        f"video_frames={v_count} ({v_count/elapsed:.2f}/s)"
    )


if __name__ == "__main__":
    main()
