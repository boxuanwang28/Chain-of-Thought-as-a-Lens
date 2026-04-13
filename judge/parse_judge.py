#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize judge results for SC-Align vs baseline (CoT).

Input: JSONL produced by `llm-judge.py` (or previous `judge_scalign_vs_baseline.py`).

We compute:
- Mean / std of logic & readability scores for baseline vs SC-Align.
- Pairwise win/tie counts for each dimension.

Compatible with records like:
{
  "Hop": 3,
  "Idx": 23,
  "ChainMapping": {"chain1": "scalign", "chain2": "baseline"},
  "BaselineSCStrategy": "scot",
  "SCAlignSCStrategy": "scot_align",
  "BaselineChain": "...",
  "SCAlignChain": "...",
  "JudgeResult": {
    "chain1_logic": 7,
    "chain2_logic": 2,
    "chain1_readability": 8,
    "chain2_readability": 3,
    "better_logic": "chain1",
    "better_readability": "chain1",
    "explanation": "..."
  }
}

Also supports records with parse_error + raw_response (wrapped in ```json fences).
"""

import json
import argparse
from typing import List, Dict, Any
import numpy as np


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data.append(obj)
    return data


def normalize_role(raw: str) -> str:
    """
    Map strings in ChainMapping to the standard roles:
      baseline / scalign
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    if s in {"baseline", "cot", "cot_baseline"}:
        return "baseline"
    if s in {"scalign", "sc-align", "scot_align"}:
        return "scalign"
    return s  # Fallback: keep the original value.


def summarize_judge(records: List[Dict[str, Any]]) -> None:
    base_logic_scores: List[float] = []
    base_read_scores: List[float] = []
    sca_logic_scores: List[float] = []
    sca_read_scores: List[float] = []

    logic_wins = {"baseline": 0, "scalign": 0, "tie": 0}
    read_wins = {"baseline": 0, "scalign": 0, "tie": 0}

    valid_cnt = 0

    for rec in records:
        jr = rec.get("JudgeResult", {})
        if not isinstance(jr, dict):
            continue

        # ========= Key handling: parse parse_error + raw_response =========
        if jr.get("parse_error"):
            raw = jr.get("raw_response")
            if isinstance(raw, str):
                text = raw.strip()
                # Remove ```json / ``` wrappers.
                if text.startswith("```"):
                    first_brace = text.find("{")
                    last_brace = text.rfind("}")
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        text = text[first_brace:last_brace + 1]
                try:
                    parsed = json.loads(text)
                    jr = parsed  # Replace the original jr with the parsed dict.
                except Exception:
                    # Skip this record if parsing still fails.
                    continue
            else:
                # No raw_response is available.
                continue
        # ==========================================================

        mapping = rec.get("ChainMapping", {})
        c1_role_raw = mapping.get("chain1")
        c2_role_raw = mapping.get("chain2")
        c1_role = normalize_role(c1_role_raw)
        c2_role = normalize_role(c2_role_raw)

        # Scores
        try:
            c1_logic = float(jr["chain1_logic"])
            c2_logic = float(jr["chain2_logic"])
            c1_read = float(jr["chain1_readability"])
            c2_read = float(jr["chain2_readability"])
        except Exception:
            continue

        if c1_role not in {"baseline", "scalign"} or c2_role not in {"baseline", "scalign"}:
            # Unexpected mapping; skip it.
            continue

        valid_cnt += 1

        # === Map chain1/chain2 scores back to baseline / scalign ===
        if c1_role == "baseline" and c2_role == "scalign":
            base_logic_scores.append(c1_logic)
            base_read_scores.append(c1_read)
            sca_logic_scores.append(c2_logic)
            sca_read_scores.append(c2_read)
        elif c1_role == "scalign" and c2_role == "baseline":
            base_logic_scores.append(c2_logic)
            base_read_scores.append(c2_read)
            sca_logic_scores.append(c1_logic)
            sca_read_scores.append(c1_read)
        else:
            # Skip baseline/baseline or scalign/scalign pairs.
            continue

        # === Pairwise winner (logic) ===
        raw_logic = jr.get("better_logic")  # "chain1"/"chain2"/"tie" or other labels.
        if raw_logic == "chain1":
            winner = c1_role               # baseline or scalign
        elif raw_logic == "chain2":
            winner = c2_role
        elif raw_logic in {"baseline", "scalign", "tie"}:
            winner = raw_logic
        else:
            # Treat unknown labels as ties.
            winner = "tie"

        if winner in logic_wins:
            logic_wins[winner] += 1

        # === Pairwise winner (readability) ===
        raw_read = jr.get("better_readability")
        if raw_read == "chain1":
            winner_r = c1_role
        elif raw_read == "chain2":
            winner_r = c2_role
        elif raw_read in {"baseline", "scalign", "tie"}:
            winner_r = raw_read
        else:
            winner_r = "tie"

        if winner_r in read_wins:
            read_wins[winner_r] += 1

    if valid_cnt == 0:
        print("No valid judge records found.")
        return

    def mean_std(arr: List[float]):
        if not arr:
            return float("nan"), float("nan")
        arr_np = np.array(arr, dtype=float)
        return float(arr_np.mean()), float(arr_np.std(ddof=1)) if len(arr_np) > 1 else 0.0

    base_logic_mean, base_logic_std = mean_std(base_logic_scores)
    base_read_mean, base_read_std = mean_std(base_read_scores)
    sca_logic_mean, sca_logic_std = mean_std(sca_logic_scores)
    sca_read_mean, sca_read_std = mean_std(sca_read_scores)

    print("========== Judge Summary ==========")
    print(f"Total valid judged pairs: {valid_cnt}\n")

    print(">> Logic (1-10)")
    print(f"Baseline:  mean={base_logic_mean:.3f}, std={base_logic_std:.3f}")
    print(f"SC-Align:  mean={sca_logic_mean:.3f}, std={sca_logic_std:.3f}\n")

    print(">> Readability (1-10)")
    print(f"Baseline:  mean={base_read_mean:.3f}, std={base_read_std:.3f}")
    print(f"SC-Align:  mean={sca_read_mean:.3f}, std={sca_read_std:.3f}\n")

    def pct(x: int) -> str:
        return f"{x} ({x / valid_cnt * 100:.1f}%)"

    print(">> Pairwise preference (logic)")
    print(f"Baseline better : {pct(logic_wins['baseline'])}")
    print(f"SC-Align better : {pct(logic_wins['scalign'])}")
    print(f"Tie             : {pct(logic_wins['tie'])}\n")

    print(">> Pairwise preference (readability)")
    print(f"Baseline better : {pct(read_wins['baseline'])}")
    print(f"SC-Align better : {pct(read_wins['scalign'])}")
    print(f"Tie             : {pct(read_wins['tie'])}")
    print("===================================")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize LLM judge results for SC-Align vs baseline."
    )
    parser.add_argument(
        "--judge-jsonl",
        type=str,
        required=True,
        help="Path to judge result jsonl file produced by llm-judge.py",
    )
    args = parser.parse_args()

    print(f"[Info] Loading judge results from: {args.judge_jsonl}")
    records = load_jsonl(args.judge_jsonl)
    summarize_judge(records)


if __name__ == "__main__":
    main()
