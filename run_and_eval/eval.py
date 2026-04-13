#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize hop=4 results for GPT experiments.

Expected filename patterns (per model & method), e.g.:

  - GPT-4o baseline:
      results-gpt-4o-baseline_r1_hop4.jsonl
  - GPT-4o SCOS-RR:
      results-gpt-4o-scos-rr_r1_hop4.jsonl
  - GPT-4o SCOS-TS:
      results-gpt-4o-scos-ts_r1_hop4.jsonl
  - GPT-4o SC-CoT:
      results-gpt-4o-sccot_r1_hop4.jsonl
  - GPT-4o SC-Align:
      results-gpt-4o-scalign_r1_hop4.jsonl

Similarly for:
  - GPT-4o-mini:
      results-gpt-4o-mini-baseline_r1_hop4.jsonl
      results-gpt-4o-mini-scos-rr_r1_hop4.jsonl
      results-gpt-4o-mini-scos-ts_r1_hop4.jsonl
      results-gpt-4o-mini-sccot_r1_hop4.jsonl
      results-gpt-4o-mini-scalign_r1_hop4.jsonl

  - GPT-3.5-turbo:
      results-gpt-3.5-turbo-baseline_r1_hop4.jsonl
      results-gpt-3.5-turbo-scos-rr_r1_hop4.jsonl
      results-gpt-3.5-turbo-scos-ts_r1_hop4.jsonl
      results-gpt-3.5-turbo-sccot_r1_hop4.jsonl
      results-gpt-3.5-turbo-scalign_r1_hop4.jsonl

Extra (NLI ablation for GPT-4o-mini, hop3):
  - GPT-4o-mini SC-Align NLI-A:
      results-gpt-4o-mini-scalign-nliA-hop3_r1_hop3.jsonl
  - GPT-4o-mini SC-Align NLI-B:
      results-gpt-4o-mini-scalign-nliB-hop3_r1_hop3.jsonl
  - GPT-4o-mini SC-Align NLI-C:
      results-gpt-4o-mini-scalign-nliC-hop3_r1_hop3.jsonl

Each group may contain multiple runs (r1, r2, ...).
"""

import os
import glob
import json
import numpy as np
from typing import List, Tuple, Dict, Optional


def load_metrics_from_file(path: str) -> Tuple[int, int, List[float]]:
    """
    Load metrics from a single jsonl file.

    Only samples with Status == "ok" are counted.

    Returns:
        num_ok:    number of valid samples
        num_corr:  number of Correct == 1
        scores:    list of alignment scores
    """
    num_ok = 0
    num_corr = 0
    scores: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("Status") != "ok":
                continue

            num_ok += 1

            c = obj.get("Correct", None)
            try:
                if c is not None and int(c) == 1:
                    num_corr += 1
            except Exception:
                pass

            sc = obj.get("Score", None)
            if isinstance(sc, (int, float)):
                scores.append(float(sc))

    return num_ok, num_corr, scores


def summarize_group(pattern: str, group_name: str) -> Optional[Dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[{group_name}] No matched files: {pattern}")
        return None

    print(f"\n========== Group: {group_name} ==========")
    print(f"Matched {len(files)} run files:")
    for fp in files:
        print(f"  - {os.path.basename(fp)}")

    run_accs = []
    run_scores_mean = []
    run_scores_std = []

    all_ok = 0
    all_corr = 0
    all_scores: List[float] = []

    print("\n[Per-run statistics]")
    print(f"{'Run':<4}{'File':<55}{'ok':>8}{'acc_run':>12}{'Score_mean':>14}{'Score_std':>14}")

    for idx, fp in enumerate(files, start=1):
        fname = os.path.basename(fp)

        # parse run id from _rK_
        run_id = idx
        if "_r" in fname and "_hop" in fname:
            try:
                mid = fname.split("_r", 1)[1]
                run_id = int(mid.split("_hop", 1)[0])
            except Exception:
                pass

        num_ok, num_corr, scores = load_metrics_from_file(fp)

        acc_run = num_corr / num_ok if num_ok > 0 else float("nan")

        if scores:
            s_mean = float(np.mean(scores))
            s_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        else:
            s_mean = float("nan")
            s_std = float("nan")

        run_accs.append(acc_run)
        run_scores_mean.append(s_mean)
        run_scores_std.append(s_std)

        all_ok += num_ok
        all_corr += num_corr
        all_scores.extend(scores)

        print(f"{run_id:<4}{fname:<55}{num_ok:>8}{acc_run:>12.4f}{s_mean:>14.4f}{s_std:>14.4f}")

    # ----- Group-level statistics -----
    valid_run_accs = [a for a in run_accs if not np.isnan(a)]

    macro_acc_mean = float(np.mean(valid_run_accs)) if valid_run_accs else float("nan")
    macro_acc_std = (
        float(np.std(valid_run_accs, ddof=1)) if len(valid_run_accs) > 1 else 0.0
    )

    micro_acc = all_corr / all_ok if all_ok > 0 else float("nan")

    score_mean_all = float(np.mean(all_scores)) if all_scores else float("nan")
    score_std_all = (
        float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
    )

    print("\n[Group-level statistics]")
    print(f"Total ok samples:        {all_ok}")
    print(f"Total correct:          {all_corr}")
    print(f"Micro accuracy:         {micro_acc:.4f}")
    print(f"Macro accuracy:         mean={macro_acc_mean:.4f}, std={macro_acc_std:.4f}")
    print(f"Alignment Score (all):  mean={score_mean_all:.4f}, std={score_std_all:.4f}")
    print("=========================================\n")

    return {
        "group": group_name,
        "total_ok": all_ok,
        "micro_acc": micro_acc,
        "macro_acc_mean": macro_acc_mean,
        "macro_acc_std": macro_acc_std,
        "score_mean": score_mean_all,
        "score_std": score_std_all,
    }


def main():
    groups: Dict[str, str] = {
        # GPT-4o-mini (hop4)
        "GPT-4o-mini baseline (hop4)": "results-gpt-4o-mini-baseline_r*_hop4.jsonl",
        "GPT-4o-mini SCOS-RR (hop4)":  "results-gpt-4o-mini-scos-rr_r*_hop4.jsonl",
        "GPT-4o-mini SCOS-TS (hop4)":  "results-gpt-4o-mini-scos-ts_r*_hop4.jsonl",
        "GPT-4o-mini SC-CoT (hop4)":   "results-gpt-4o-mini-sccot_r*_hop4.jsonl",
        "GPT-4o-mini SC-Align (hop4)": "results-gpt-4o-mini-scalign_r*_hop4.jsonl",

        # GPT-4o (hop4)
        "GPT-4o baseline (hop4)": "results-gpt-4o-baseline_r*_hop4.jsonl",
        "GPT-4o SCOS-RR (hop4)":  "results-gpt-4o-scos-rr_r*_hop4.jsonl",
        "GPT-4o SCOS-TS (hop4)":  "results-gpt-4o-scos-ts_r*_hop4.jsonl",
        "GPT-4o SC-CoT (hop4)":   "results-gpt-4o-sccot_r*_hop4.jsonl",
        "GPT-4o SC-Align (hop4)": "results-gpt-4o-scalign_r*_hop4.jsonl",

        # GPT-3.5-turbo (hop4)
        "GPT-3.5 baseline (hop4)": "results-gpt-3.5-turbo-baseline_r*_hop4.jsonl",
        "GPT-3.5 SCOS-RR (hop4)":  "results-gpt-3.5-turbo-scos-rr_r*_hop4.jsonl",
        "GPT-3.5 SCOS-TS (hop4)":  "results-gpt-3.5-turbo-scos-ts_r*_hop4.jsonl",
        "GPT-3.5 SC-CoT (hop4)":   "results-gpt-3.5-turbo-sccot_r*_hop4.jsonl",
        "GPT-3.5 SC-Align (hop4)": "results-gpt-3.5-turbo-scalign_r*_hop4.jsonl",

        # ===== Added: GPT-4o-mini SC-Align NLI ablation (hop3) =====
        "GPT-4o-mini SC-Align NLI-A (hop3)": "results-gpt-4o-mini-scalign-nliA-hop3_r*_hop3.jsonl",
        "GPT-4o-mini SC-Align NLI-B (hop3)": "results-gpt-4o-mini-scalign-nliB-hop3_r*_hop3.jsonl",
        "GPT-4o-mini SC-Align NLI-C (hop3)": "results-gpt-4o-mini-scalign-nliC-hop3_r*_hop3.jsonl",
    }

    all_results = []

    for group_name, pattern in groups.items():
        res = summarize_group(pattern, group_name)
        if res is not None:
            all_results.append(res)

    if not all_results:
        print("No valid groups found.")
        return

    print("\n########## Overall Comparison (hop4 / hop3-mixed) ##########")
    header = (
        f"{'Group':<32}"
        f"{'Total_ok':>10}"
        f"{'Micro_acc':>12}"
        f"{'Macro_acc':>22}"
        f"{'Score_mean':>16}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['group']:<32}"
            f"{r['total_ok']:>10d}"
            f"{r['micro_acc']:>12.4f}"
            f"{r['macro_acc_mean']:>12.4f}±{r['macro_acc_std']:<8.4f}"
            f"{r['score_mean']:>8.4f}±{r['score_std']:<8.4f}"
        )

    print("###############################################\n")


if __name__ == "__main__":
    main()
