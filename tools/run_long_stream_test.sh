#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-precomputed}"  # precomputed | online

COMMON_ARGS=(
  --conf local/vox2_10w_conf_64_64_3_adamw_1e-1_blocks16_pretrain_test.yml
  --checkpoint pretrained_model/vox2_best_model.pt
  --audio test_videos/interview_visualvoice/interview.wav
  --output outputs/stream_long_5min.wav
  --sample-rate 16000
  --fps 25
  --chunk-ms 320
  --hop-ms 320
  --window-ms 2000
  --lookahead-ms 0
  --max-video-queue 200
  --realtime 0
  --loop 1
  --max-duration-s 300
  --stats-interval-s 5
)

if [[ "$MODE" == "online" ]]; then
  python tools/run_stream.py \
    "${COMMON_ARGS[@]}" \
    --mouthroi-mode online \
    --video test_videos/interview_visualvoice/faces/speaker1.mp4 \
    --landmark-npz test_videos/interview_visualvoice/landmark/speaker1.npz
else
  python tools/run_stream.py \
    "${COMMON_ARGS[@]}" \
    --mouthroi-mode precomputed \
    --mouth-npz test_videos/interview_visualvoice/mouthroi/speaker1.npz
fi
