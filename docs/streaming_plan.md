# Streaming Plan (v2)

## Baseline I/O
- Offline baseline command: `bash run_demo.sh`
- Audio: `test_videos/interview_visualvoice/interview.wav`, mono, `16kHz`, `17.536s`
- Raw face video (for online mouthroi): `test_videos/interview_visualvoice/faces/speaker1.mp4`, `25fps`, `438` frames
- Landmark source (for online streaming extractor): `test_videos/interview_visualvoice/landmark/speaker1.npz`
- Precomputed mouthroi (Mode A): `test_videos/interview_visualvoice/mouthroi/speaker1.npz`
- Output: `outputs/*.wav`

## Milestones (v2)
- [x] M0 Offline baseline reproducible
  - [x] `bash run_demo.sh` passed in target conda env
- [x] M1 File-as-stream data sources
  - [x] `AudioFileStream` + `VideoFileStream`
  - [x] `tools/preview_stream.py`
- [x] M2 Streaming separator loop with Mode A
  - [x] `tools/run_stream.py --mouthroi-mode precomputed`
- [x] M3 Streaming mouthroi extractor (core)
  - [x] `streaming/mouthroi_stream.py` (`MouthROIExtractor`, `OnlineMouthRoiStream`)
  - [x] `tools/preview_mouthroi_stream.py`
- [x] M4 End-to-end default online mouthroi (Mode B)
  - [x] `run_stream_demo.sh` default uses `--mouthroi-mode online`
  - [x] Keep debug/compare mode `--mouthroi-mode precomputed`
- [x] M5 Stability + metrics
  - [x] 5-min simulated streaming test: `bash tools/run_long_stream_test.sh`
  - [x] continuity check: `tools/check_stream_continuity.py`
- [x] M6 Documentation and usability
  - [x] `docs/streaming_usage.md` updated with v2 commands and limitations

## Implemented Components
- `streaming/audio_stream.py`: timestamped audio chunks with `chunk_ms/hop_ms/realtime/loop`
- `streaming/video_stream.py`: raw frame stream and precomputed mouthroi stream
- `streaming/mouthroi_stream.py`: incremental mouthroi extraction from `frame + landmark`
- `streaming/sync.py`: A/V timestamp alignment, bounded queue, drop stats
- `streaming/engine.py`: stateful model wrapper with rolling context
- `streaming/sink.py`: wav sink + crossfade stitching
- `tools/run_stream.py`: unified entry, default `--mouthroi-mode online`

## Validation Commands
- Baseline:
  - `source /work1/weiyang/environment/miniconda3/etc/profile.d/conda.sh && conda activate /work1/weiyang/environment/miniconda3/envs/test_ctcnet_numpy1 && bash run_demo.sh`
- Preview stream:
  - `python tools/preview_stream.py --audio test_videos/interview_visualvoice/interview.wav --mouth-npz test_videos/interview_visualvoice/mouthroi/speaker1.npz --sample-rate 16000 --fps 25 --chunk-ms 80 --hop-ms 80 --seconds 10 --realtime 1`
- Preview online mouthroi:
  - `python tools/preview_mouthroi_stream.py --video test_videos/interview_visualvoice/faces/speaker1.mp4 --landmark-npz test_videos/interview_visualvoice/landmark/speaker1.npz --fps 25 --seconds 10 --realtime 1`
- End-to-end default online:
  - `bash run_stream_demo.sh`
- End-to-end precomputed compare:
  - `python tools/run_stream.py --mouthroi-mode precomputed --audio test_videos/interview_visualvoice/interview.wav --mouth-npz test_videos/interview_visualvoice/mouthroi/speaker1.npz --output outputs/stream_precomputed.wav --sample-rate 16000 --fps 25 --chunk-ms 320 --hop-ms 320 --window-ms 2000 --realtime 0`
- 5-min stability:
  - `bash tools/run_long_stream_test.sh`
- Continuity:
  - `python tools/check_stream_continuity.py --wav outputs/stream_long_5min.wav --sample-rate 16000`

## Assumptions / Limitations
- Current online mouthroi mode consumes existing landmark stream (`landmark/*.npz`) and performs streaming crop on incoming frames.
- Online face detection + landmark inference is not yet inserted into `run_stream.py` main path.
- `tools/run_long_stream_test.sh` defaults to `precomputed` mode for practical runtime; optional `online` mode: `bash tools/run_long_stream_test.sh online`.
