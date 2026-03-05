# Streaming Usage (v2)

## Environment
```bash
source /work1/weiyang/environment/miniconda3/etc/profile.d/conda.sh
conda activate /work1/weiyang/environment/miniconda3/envs/test_ctcnet_numpy1
```

## 1) Preview audio/video stream rates
```bash
python tools/preview_stream.py \
  --audio test_videos/interview_visualvoice/interview.wav \
  --mouth-npz test_videos/interview_visualvoice/mouthroi/speaker1.npz \
  --sample-rate 16000 \
  --fps 25 \
  --chunk-ms 80 \
  --hop-ms 80 \
  --seconds 10 \
  --realtime 1
```

## 2) Preview online mouthroi extraction (M3)
```bash
python tools/preview_mouthroi_stream.py \
  --video test_videos/interview_visualvoice/faces/speaker1.mp4 \
  --landmark-npz test_videos/interview_visualvoice/landmark/speaker1.npz \
  --fps 25 \
  --seconds 10 \
  --realtime 1
```
Logs include `feat_rate`, `mouth_avg`, `mouth_max`, `drops`, and output `shape`.

## 3) End-to-end streaming demo (default Mode B online)
```bash
bash run_stream_demo.sh
```
Default output: `outputs/stream_spk1.wav`

## 4) Run with explicit mode
- Online mouthroi mode (default):
```bash
python tools/run_stream.py \
  --mouthroi-mode online \
  --audio test_videos/interview_visualvoice/interview.wav \
  --video test_videos/interview_visualvoice/faces/speaker1.mp4 \
  --landmark-npz test_videos/interview_visualvoice/landmark/speaker1.npz \
  --output outputs/stream_online.wav \
  --sample-rate 16000 --fps 25 --chunk-ms 320 --hop-ms 320 --window-ms 2000 --realtime 0
```
- Precomputed mouthroi mode (debug/compare):
```bash
python tools/run_stream.py \
  --mouthroi-mode precomputed \
  --audio test_videos/interview_visualvoice/interview.wav \
  --mouth-npz test_videos/interview_visualvoice/mouthroi/speaker1.npz \
  --output outputs/stream_precomputed.wav \
  --sample-rate 16000 --fps 25 --chunk-ms 320 --hop-ms 320 --window-ms 2000 --realtime 0
```

## 5) 5-minute stability test
```bash
bash tools/run_long_stream_test.sh
```
- Default long-test mode is `precomputed` (faster runtime).
- Optional online long-test:
```bash
bash tools/run_long_stream_test.sh online
```

## 6) Continuity check
```bash
python tools/check_stream_continuity.py --wav outputs/stream_spk1.wav --sample-rate 16000
```

## Runtime Metrics
`tools/run_stream.py` prints:
- `audio_chunks` and chunk rate
- `video_frames` and fps
- `mouth_avg` / `mouth_max` / `mouth_drop`
- `infer_avg` / `p95`
- queue `vq_avg` / `vq_max`
- sync drop/empty-align counts
- output wav written seconds

## Known Limitations
- Online mouthroi currently uses provided landmarks (`landmark/*.npz`) with streaming crop from video frames.
- Full online face detection + landmark inference is available in offline prep scripts, but not yet inside `run_stream.py` real-time loop.
