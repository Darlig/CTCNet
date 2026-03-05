import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from nichang.videomodels import VideoModel
from nichang.models.ctcnet import CTCNet
from nichang.datas.transform import get_preprocessing_pipelines


def load_jsonl(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON at line {line_no}: {e}\nLINE={line[:200]}") from e


def safe_stem(p: str) -> str:
    # s00024_... .wav -> s00024_...
    return Path(p).stem


def main(conf: dict, jsonl_path: str, out_dir: str, model_path: Optional[str]):
    # ---- build models ----
    conf["exp_dir"] = os.path.join("exp", conf["log"]["exp_name"])
    conf["audionet"].update({"n_src": 1})

    sample_rate = conf["data"]["sample_rate"]
    audiomodel = CTCNet(sample_rate=sample_rate, **conf["audionet"])
    videomodel = VideoModel(**conf["videonet"])

    # default ckpt behavior (keep your old logic, but allow override)
    if model_path is None:
        # 你原脚本里最后强行用的是这个
        model_path = "pretrained_model/vox2_best_model.pt"

    ckpt_obj = torch.load(model_path, map_location="cpu")
    # 兼容：有的存的是 {'state_dict': ...}，有的直接就是 state_dict
    state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    audiomodel.load_state_dict(state_dict, strict=True)

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    os.makedirs(out_dir, exist_ok=True)

    # preprocessing pipeline (与你原来一致)
    mouth_preproc = get_preprocessing_pipelines()["val"]

    with torch.no_grad():
        for line_no, item in load_jsonl(jsonl_path):
            audio_path = item.get("audio")
            spk1_npz = item.get("spk1")
            spk2_npz = item.get("spk2")

            if not audio_path or not spk1_npz or not spk2_npz:
                raise RuntimeError(
                    f"Missing keys at line {line_no}. Need audio/spk1/spk2. Got: {list(item.keys())}"
                )

            # ---- load audio ----
            mix, sr = sf.read(audio_path, dtype="float32")
            if sr != sample_rate:
                raise RuntimeError(
                    f"Sample rate mismatch at line {line_no}: wav sr={sr}, conf sample_rate={sample_rate}\n"
                    f"audio={audio_path}"
                )

            mix_t = torch.from_numpy(mix).to(model_device)

            base = safe_stem(audio_path)

            # ---- run two targets ----
            for idx, npz_path in [(1, spk1_npz), (2, spk2_npz)]:
                npz = np.load(npz_path)
                if "data" not in npz:
                    raise RuntimeError(f'NPZ missing key "data" at line {line_no}: {npz_path}')

                mouth = mouth_preproc(npz["data"])  # numpy
                mouth_t = torch.from_numpy(mouth).to(model_device)

                mouth_emb = videomodel(mouth_t.unsqueeze(0).unsqueeze(1).float())
                est_sources = audiomodel(mix_t[None, None], mouth_emb)

                # save
                out_path = os.path.join(out_dir, f"{base}__spk{idx}.wav")
                sf.write(out_path, est_sources.squeeze(0).squeeze(0).detach().cpu().numpy(), sample_rate)

            print(f"[OK] line {line_no}: {audio_path} -> {base}__spk1.wav, {base}__spk2.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--conf_dir",
        default="exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/conf.yml",
        help="Path to yaml config"
    )
    parser.add_argument("--jsonl", required=True, help="Input jsonl, one sample per line")
    parser.add_argument("--out_dir", required=True, help="Directory to save separated wavs")
    parser.add_argument("--model_path", default=None, help="Override checkpoint path (optional)")

    args = parser.parse_args()

    with open(args.conf_dir, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    main(conf, jsonl_path=args.jsonl, out_dir=args.out_dir, model_path=args.model_path)
