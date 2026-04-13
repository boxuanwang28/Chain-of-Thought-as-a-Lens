#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use GPT-4o-mini as a judge to compare SC-Align vs baseline reasoning chains.

Given:
  - one baseline jsonl file
  - one SC-Align jsonl file

Each line has a structure like:

{
  "Hop": 4,
  "Idx": 2,
  "Status": "ok",
  "SCStrategy": "scot_align",
  "RefQuestion": "...",
  "RefSteps": [...],
  "RefTherefore": "...",
  "TargetQuestion": "...",
  "TargetOptions": {...},
  "ResultSteps": [...],
  "ResultTherefore": "...",
  ...
}

We match records by (Hop, Idx) and ask GPT-4o-mini to evaluate:

  - Logical completeness
  - Readability

for baseline chain vs SC-Align chain.

Output: a JSONL file where each line contains judge scores and explanation.
"""

import os
import json
import argparse
import random
from typing import Dict, Tuple, Any

from openai import OpenAI


def load_jsonl_to_dict(path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Load a jsonl file and index by (Hop, Idx).
    Only keep Status == 'ok'.
    """
    data: Dict[Tuple[int, int], Dict[str, Any]] = {}
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

            hop = int(obj.get("Hop", 0))
            idx = int(obj.get("Idx", -1))
            key = (hop, idx)
            data[key] = obj
    return data


def build_chain_text(obj: Dict[str, Any]) -> str:
    """
    Build a readable text version of a reasoning chain from a record.
    """
    steps = obj.get("ResultSteps") or obj.get("RefSteps") or []
    therefore = obj.get("ResultTherefore") or obj.get("RefTherefore") or ""
    if isinstance(steps, str):
        steps = [steps]

    lines = []
    for i, s in enumerate(steps, start=1):
        lines.append(f"Step {i}: {s}")
    if therefore:
        lines.append(f"Conclusion: {therefore}")
    return "\n".join(lines)


def build_judge_prompt(
    question: str,
    options: Dict[str, str],
    baseline_chain: str,
    scalign_chain: str,
    chain1_label: str,
) -> str:
    """
    Build a single text prompt for the judge model.
    chain1_label is either 'baseline' or 'scalign', telling us what chain1 represents.
    """
    opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    if chain1_label == "baseline":
        chain1_name = "Baseline"
        chain2_name = "SC-Align"
    else:
        chain1_name = "SC-Align"
        chain2_name = "Baseline"

    prompt = f"""You are an expert evaluator of reasoning chains.

You will be given:
- A multiple-choice question and its options.
- Two reasoning chains that both try to answer this question:
  Chain 1 is produced by **{chain1_name}**.
  Chain 2 is produced by **{chain2_name}**.

You must judge each chain along two dimensions:
1. Logical completeness: Does the chain cover the key reasoning steps needed to justify the answer? Is the causal logic coherent and sufficiently detailed?
2. Readability: Is the chain easy to understand, well-structured, and free of confusing repetition?

For EACH dimension, assign a score from 1 (very poor) to 10 (excellent) to BOTH chains.
Then decide which chain is better on that dimension (or 'tie' if they are comparable).

Return your judgment as a JSON object with the following fields ONLY:

{{
  "chain1_logic": <integer 1-10>,
  "chain2_logic": <integer 1-10>,
  "chain1_readability": <integer 1-10>,
  "chain2_readability": <integer 1-10>,
  "better_logic": "chain1" | "chain2" | "tie",
  "better_readability": "chain1" | "chain2" | "tie",
  "explanation": "<brief explanation in 1-3 sentences>"
}}

Make sure the output is valid JSON (no comments, no trailing commas).

Question:
{question}

Options:
{opts_str}

--- Chain 1 ---
{baseline_chain if chain1_label == "baseline" else scalign_chain}

--- Chain 2 ---
{scalign_chain if chain1_label == "baseline" else baseline_chain}
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Use GPT-4o-mini to judge SC-Align vs baseline reasoning chains."
    )
    parser.add_argument(
        "--baseline-jsonl",
        type=str,
        required=True,
        help="Path to baseline jsonl file (e.g., SC-CoT / baseline).",
    )
    parser.add_argument(
        "--scalign-jsonl",
        type=str,
        required=True,
        help="Path to SC-Align jsonl file.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        required=True,
        help="Output jsonl file for judge results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional: max number of (Hop,Idx) pairs to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chain order shuffling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name (default: gpt-4o-mini).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please export it before running this script.")

    client = OpenAI(api_key=api_key)

    print(f"[Info] Loading baseline from: {args.baseline_jsonl}")
    baseline_data = load_jsonl_to_dict(args.baseline_jsonl)

    print(f"[Info] Loading SC-Align from: {args.scalign_jsonl}")
    scalign_data = load_jsonl_to_dict(args.scalign_jsonl)

    # intersection of keys
    common_keys = sorted(set(baseline_data.keys()) & set(scalign_data.keys()))
    print(f"[Info] Found {len(common_keys)} common (Hop,Idx) pairs.")

    if args.max_samples is not None:
        common_keys = common_keys[: args.max_samples]
        print(f"[Info] Limiting to first {len(common_keys)} samples due to --max-samples.")

    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    out_f = open(args.out_jsonl, "w", encoding="utf-8")

    for i, key in enumerate(common_keys, start=1):
        hop, idx = key
        base_rec = baseline_data[key]
        sca_rec = scalign_data[key]

        # Build question + options
        q = sca_rec.get("TargetQuestion") or base_rec.get("TargetQuestion") or ""
        options = sca_rec.get("TargetOptions") or base_rec.get("TargetOptions") or {}
        if not isinstance(options, dict):
            options = {}

        baseline_chain_text = build_chain_text(base_rec)
        scalign_chain_text = build_chain_text(sca_rec)

        # Randomly decide which goes first
        chain1_label = random.choice(["baseline", "scalign"])

        prompt = build_judge_prompt(
            question=q,
            options=options,
            baseline_chain=baseline_chain_text,
            scalign_chain=scalign_chain_text,
            chain1_label=chain1_label,
        )

        print(f"[{i}/{len(common_keys)}] Judging Hop={hop}, Idx={idx}, chain1={chain1_label}...")

        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict but fair evaluator of reasoning quality.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        text = completion.choices[0].message.content.strip()

        try:
            judge_obj = json.loads(text)
        except Exception:
            # In case of formatting problems, store raw text
            judge_obj = {
                "parse_error": True,
                "raw_response": text,
            }

        out_record = {
            "Hop": hop,
            "Idx": idx,
            "JudgeModel": args.model,
            "ChainMapping": {
                "chain1": chain1_label,           # "baseline" or "scalign"
                "chain2": "scalign" if chain1_label == "baseline" else "baseline",
            },
            "BaselineSCStrategy": base_rec.get("SCStrategy"),
            "SCAlignSCStrategy": sca_rec.get("SCStrategy"),
            "TargetQuestion": q,
            "TargetOptions": options,
            "BaselineChain": baseline_chain_text,
            "SCAlignChain": scalign_chain_text,
            "JudgeResult": judge_obj,
        }

        out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print(f"[Done] Saved judge results to: {args.out_jsonl}")
    

if __name__ == "__main__":
    main()
