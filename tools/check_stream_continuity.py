#!/usr/bin/env python
import argparse

import numpy as np
import soundfile as sf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Check simple continuity metrics for streaming output wav")
    p.add_argument("--wav", required=True)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--jump-threshold", type=float, default=0.4)
    return p


def zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = np.signbit(x)
    return float(np.mean(s[1:] != s[:-1]))


def main() -> None:
    args = build_parser().parse_args()
    wav, sr = sf.read(args.wav, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    if sr != args.sample_rate:
        raise ValueError(f"sample rate mismatch: wav={sr}, expected={args.sample_rate}")

    duration = wav.shape[0] / sr
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    zcr = zero_crossing_rate(wav)

    if wav.size >= 2:
        jumps = np.abs(wav[1:] - wav[:-1])
        max_jump = float(np.max(jumps))
        p99_jump = float(np.percentile(jumps, 99))
        jump_count = int(np.sum(jumps > args.jump_threshold))
    else:
        max_jump = 0.0
        p99_jump = 0.0
        jump_count = 0

    print(
        f"wav={args.wav} duration_s={duration:.3f} peak={peak:.4f} rms={rms:.4f} zcr={zcr:.5f} "
        f"max_jump={max_jump:.4f} p99_jump={p99_jump:.4f} jump_count(>{args.jump_threshold})={jump_count}"
    )


if __name__ == "__main__":
    main()
