#!/usr/bin/env python3
import argparse
import itertools
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


def read_kaldi_text(path: Path) -> Dict[str, str]:
    """Read Kaldi-style text: `uttid <space> transcript`.

    We keep the transcript as raw text so we can compute either word-level WER
    (split by whitespace) or char-level CER (split into characters).
    """
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 0:
                continue
            utt = parts[0]
            txt = parts[1] if len(parts) == 2 else ""
            m[utt] = txt
    return m


def _is_punct_char(ch: str) -> bool:
    # Unicode punctuation categories start with 'P'
    return unicodedata.category(ch).startswith("P")


def normalize_and_split(text: str, unit: str, do_lower: bool, remove_punct: bool) -> List[str]:
    """Normalize and split transcript into units.

    unit:
      - 'word': split by whitespace (WER)
      - 'char': split into characters after removing whitespace (CER)
    
    IMPORTANT: We avoid regex like \W which would treat CJK as non-word and delete it.
    """
    if do_lower:
        # safe for ASCII; doesn't harm CJK
        text = text.lower()

    if unit == "word":
        toks = text.split()
        if remove_punct:
            # strip leading/trailing punctuation for each token; drop tokens that become empty
            cleaned = []
            for t in toks:
                # strip punct chars at both ends
                i, j = 0, len(t)
                while i < j and _is_punct_char(t[i]):
                    i += 1
                while j > i and _is_punct_char(t[j - 1]):
                    j -= 1
                t2 = t[i:j]
                if t2:
                    cleaned.append(t2)
            toks = cleaned
        return toks

    if unit == "char":
        # remove all whitespace characters
        chars = [c for c in text if not c.isspace()]
        if remove_punct:
            chars = [c for c in chars if not _is_punct_char(c)]
        return chars

    raise ValueError(f"Unsupported unit: {unit}")


def edit_counts(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Return (S, D, I, N) using Levenshtein alignment counts.
    N = len(ref)
    """
    n = len(ref)
    m = len(hyp)
    # dp cost
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # backtrace op: 0=ok/sub, 1=del, 2=ins
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost_sub = dp[i - 1][j - 1]  # correct
            else:
                cost_sub = dp[i - 1][j - 1] + 1  # substitution

            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1

            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                bt[i][j] = 0
            elif best == cost_del:
                bt[i][j] = 1
            else:
                bt[i][j] = 2

    # backtrace counts
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 0:
            # match or substitution
            if i > 0 and j > 0:
                if ref[i - 1] != hyp[j - 1]:
                    S += 1
                i -= 1
                j -= 1
            else:
                # edge (shouldn't happen)
                if i > 0:
                    D += 1
                    i -= 1
                elif j > 0:
                    I += 1
                    j -= 1
        elif op == 1:
            D += 1
            i -= 1
        elif op == 2:
            I += 1
            j -= 1
        else:
            # origin
            break

    return S, D, I, n


def best_perm_for_utt(
    ref_spks: List[List[str]],
    hyp_chs: List[List[str]],
) -> Tuple[Tuple[int, ...], Tuple[int, int, int, int]]:
    """
    Try all permutations: assign hyp channel k -> ref speaker perm[k].
    Return best perm and summed counts (S,D,I,N).
    """
    K = len(ref_spks)
    assert len(hyp_chs) == K

    best_perm = None
    best_cost = None
    best_counts = None

    for perm in itertools.permutations(range(K)):
        S = D = I = N = 0
        for ch_idx, spk_idx in enumerate(perm):
            s, d, ins, n = edit_counts(ref_spks[spk_idx], hyp_chs[ch_idx])
            S += s
            D += d
            I += ins
            N += n
        cost = S + D + I
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_counts = (S, D, I, N)

    assert best_perm is not None and best_counts is not None
    return best_perm, best_counts


def main():
    ap = argparse.ArgumentParser(
        description="Oracle (PIT) WER for 2mix/Nmix. Also writes oracle-aligned hyp files for Kaldi compute-wer."
    )
    ap.add_argument("--refs", nargs="+", required=True, help="ref text files per speaker: ref_spk1.txt ref_spk2.txt ...")
    ap.add_argument("--hyps", nargs="+", required=True, help="hyp text files per separated channel: hyp_ch1.txt hyp_ch2.txt ...")
    ap.add_argument("--out_dir", type=Path, default=Path("oracle_out"), help="output dir for oracle-aligned hyp files")
    ap.add_argument("--unit", choices=["word", "char"], default="word", help="scoring unit: word=W​ER (whitespace tokens), char=C​ER (per-character)")
    ap.add_argument("--lower", action="store_true", help="lowercase before scoring")
    ap.add_argument("--remove_punct", action="store_true", help="remove punctuation before scoring (Unicode-safe)")
    ap.add_argument("--write_mapping", action="store_true", help="write per-utt best permutation mapping")
    args = ap.parse_args()

    if len(args.refs) != len(args.hyps):
        raise SystemExit(f"--refs ({len(args.refs)}) and --hyps ({len(args.hyps)}) must have same length")

    K = len(args.refs)
    refs = [read_kaldi_text(Path(p)) for p in args.refs]
    hyps = [read_kaldi_text(Path(p)) for p in args.hyps]

    # utt set intersection
    utts = set(refs[0].keys())
    for r in refs[1:]:
        utts &= set(r.keys())
    for h in hyps:
        utts &= set(h.keys())

    if not utts:
        raise SystemExit("No common uttids across all ref/hyp files.")

    utts = sorted(utts)

    # prepare output hyp per speaker (oracle-aligned)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_hyp_spk = [open(args.out_dir / f"hyp_oracle_spk{i+1}.txt", "w", encoding="utf-8") for i in range(K)]
    map_f = open(args.out_dir / "oracle_mapping.tsv", "w", encoding="utf-8") if args.write_mapping else None
    if map_f:
        map_f.write("utt\t" + "\t".join([f"ch{c+1}->spk?" for c in range(K)]) + "\n")

    totS = totD = totI = totN = 0

    # Macro scoring: average per-utt error rate (unweighted by utt length)
    sum_utt_err_rate = 0.0
    num_utts = 0

    for utt in utts:
        ref_spks = []
        hyp_chs = []
        for i in range(K):
            rt = normalize_and_split(refs[i][utt], unit=args.unit, do_lower=args.lower, remove_punct=args.remove_punct)
            ht = normalize_and_split(hyps[i][utt], unit=args.unit, do_lower=args.lower, remove_punct=args.remove_punct)
            ref_spks.append(rt)
            hyp_chs.append(ht)

        perm, (S, D, I, N) = best_perm_for_utt(ref_spks, hyp_chs)

        totS += S
        totD += D
        totI += I
        totN += N

        # Macro: per-utt error rate. If N==0 (empty ref), fall back to denom=1 to avoid div-by-zero.
        denom = N if N > 0 else 1
        sum_utt_err_rate += (S + D + I) / denom
        num_utts += 1

        # Write oracle-aligned hyp per speaker: ref speaker j gets hyp from the channel that mapped to j
        # perm maps: channel -> speaker
        spk_to_ch = [None] * K
        for ch_idx, spk_idx in enumerate(perm):
            spk_to_ch[spk_idx] = ch_idx

        for spk_idx in range(K):
            ch_idx = spk_to_ch[spk_idx]
            assert ch_idx is not None
            txt = hyps[ch_idx][utt]  # write ORIGINAL text (before normalize)
            out_hyp_spk[spk_idx].write(utt + (" " + txt if txt else "") + "\n")

        if map_f:
            map_f.write(utt + "\t" + "\t".join([str(x + 1) for x in perm]) + "\n")

    for f in out_hyp_spk:
        f.close()
    if map_f:
        map_f.close()

    micro = (totS + totD + totI) / max(1, totN) * 100.0
    macro = (sum_utt_err_rate / max(1, num_utts)) * 100.0

    label = "ORACLE_WER" if args.unit == "word" else "ORACLE_CER"
    print(f"{label} {micro:.2f}% [ {totS+totD+totI} / {totN}, {totI} ins, {totD} del, {totS} sub ]")
    print(f"{label}_MACRO {macro:.2f}% [ avg over {num_utts} utts ]")
    print(f"[OUT] wrote oracle hyp files to: {args.out_dir}")


if __name__ == "__main__":
    main()