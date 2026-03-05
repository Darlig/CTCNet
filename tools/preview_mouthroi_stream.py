#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming.mouthroi_stream import OnlineMouthRoiStream


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Preview streaming mouth ROI extraction")
    p.add_argument("--video", default="test_videos/interview_visualvoice/faces/speaker1.mp4")
    p.add_argument("--landmark-npz", default="test_videos/interview_visualvoice/landmark/speaker1.npz")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--realtime", type=int, default=1)
    p.add_argument("--window-margin", type=int, default=12)
    return p


def main() -> None:
    args = build_parser().parse_args()

    stream = OnlineMouthRoiStream(
        video_path=args.video,
        landmark_npz=args.landmark_npz,
        fps=args.fps,
        realtime=bool(args.realtime),
        loop=False,
        window_margin=args.window_margin,
    )

    start = time.time()
    count = 0
    first_shape = None

    for feat in stream:
        count += 1
        if first_shape is None:
            first_shape = feat.data.shape

        elapsed = max(1e-6, time.time() - start)
        if count % 25 == 0:
            stats = stream.stats
            print(
                f"t={feat.timestamp_s:7.3f}s feat_rate={count/elapsed:6.2f}/s "
                f"mouth_avg={stats.avg_process_ms:7.2f}ms mouth_max={stats.max_process_ms:7.2f}ms "
                f"drops={stats.dropped_frames:4d}"
            )

        if feat.timestamp_s >= args.seconds:
            break

    elapsed = max(1e-6, time.time() - start)
    stats = stream.stats
    print(
        "mouth_preview_done "
        f"elapsed={elapsed:.2f}s "
        f"emitted={count} ({count/elapsed:.2f}/s) "
        f"shape={first_shape} "
        f"avg_ms={stats.avg_process_ms:.2f} "
        f"max_ms={stats.max_process_ms:.2f} "
        f"drops={stats.dropped_frames}"
    )


if __name__ == "__main__":
    main()
