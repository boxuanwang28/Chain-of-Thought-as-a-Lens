#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_ref_chains_from_hf.py — Generate 1/2/3/4-hop reference chains from real HuggingFace datasets (ScienceQA-style).

Updates:
- Fixes KeyError: '"because_steps"' caused by braces in the JSON example.
- Fixes an extra right brace in an f-string.
- Adds 4-hop support; --hop all now generates 1/2/3/4-hop chains.
"""

import os
import re
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional


# ------------------------------
# Provider Abstraction
# ------------------------------
@dataclass
class LLMConfig:
    provider: str  # "openai" | "hf"
    model: str
    temperature: float = 0.6
    max_tokens: int = 640
    top_p: float = 0.9
    device: Optional[str] = None
    api_key: Optional[str] = None
    hf_token: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._gen_max = min(self.cfg.max_tokens, 512)
        if cfg.provider == "openai":
            from openai import OpenAI
            if cfg.api_key:
                os.environ["OPENAI_API_KEY"] = cfg.api_key
                self.client = OpenAI(api_key=cfg.api_key)
            else:
                self.client = OpenAI()
            self.pipe = None
        elif cfg.provider == "hf":
            from transformers import pipeline
            if cfg.hf_token:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
                try:
                    from huggingface_hub import login as hf_login
                    hf_login(token=cfg.hf_token, add_to_git_credential=False)
                except Exception:
                    # Token login failure is not fatal; the pipeline can still use local/cached models.
                    pass
            self.client = None
            self.pipe = pipeline(
                "text-generation",
                model=cfg.model,
                tokenizer=cfg.model,
                device_map="auto" if (cfg.device in [None, "auto"]) else None,
                model_kwargs={"torch_dtype": "auto"},
            )
        else:
            raise ValueError("provider must be openai or hf")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        sys_msg = system_prompt or "You are a helpful assistant that writes concise, structured reasoning chains."
        if self.cfg.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self._gen_max,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Simple LLaMA/Qwen-style chat template; does not rely on tokenizer.chat_template.
            full_prompt = f"[INST]{sys_msg}\n\n{user_prompt} [/INST]"
            outs = self.pipe(
                full_prompt,
                max_new_tokens=self._gen_max,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
            )
            return outs[0]["generated_text"][len(full_prompt):].strip()


# ------------------------------
# Load real dataset from HuggingFace
# ------------------------------
def load_hf_examples(name: str, split: str, limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset

    name = name.lower()
    rows: List[Dict] = []

    if name in ("arc_easy", "arc_challenge"):
        ds = load_dataset("ai2_arc", "ARC-Easy" if name == "arc_easy" else "ARC-Challenge", split=split)
        for ex in ds:
            q = ex["question"].strip()
            choices = ex["choices"]
            ans = ex["answerKey"].strip().upper()
            opt_map = {l.upper(): t.strip() for l, t in zip(choices["label"], choices["text"]) if l.upper() in "ABCD"}
            if len(opt_map) == 4 and ans in opt_map:
                rows.append({"question": q, "context": "", "options": opt_map, "answer": ans})

    elif name == "sciq":
        ds = load_dataset("sciq", split=split)
        for ex in ds:
            q = ex["question"].strip()
            ctx = ex["support"].strip()
            ans = ex["correct_answer"].strip()
            opts = [ans, ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            letters = ["A", "B", "C", "D"]
            random.shuffle(opts)
            options = {l: t for l, t in zip(letters, opts)}
            ansL = next(l for l, t in options.items() if t == ans)
            rows.append({"question": q, "context": ctx, "options": options, "answer": ansL})

    elif name == "csqa":
        ds = load_dataset("commonsense_qa", split=split)
        for ex in ds:
            q = ex["question"].strip()
            ans = ex["answerKey"].upper()
            choices = {c["label"].upper(): c["text"].strip() for c in ex["choices"]}
            if len(choices) == 5:
                # Keep only A-D and drop E.
                choices = dict(list(choices.items())[:4])
            if ans in choices:
                rows.append({"question": q, "context": "", "options": choices, "answer": ans})
    else:
        raise ValueError("Unsupported dataset name")

    if limit:
        rows = rows[:limit]
    return rows


# ------------------------------
# Build prompt (fixed f-string version)
# ------------------------------
def build_ref_prompt_from_example(ex: Dict, hop: int) -> str:
    q, ctx, opts = ex["question"], ex.get("context", ""), ex["options"]
    ansL, ansT = ex["answer"], ex["options"][ex["answer"]]

    return (
        f"You are given a multiple-choice science question. "
        f"Write a concise reasoning chain that supports the correct answer.\n"
        f"Return ONLY one XML-wrapped JSON object between <json> and </json> tags.\n"
        f"JSON keys: 'because_steps' (array of strings with EXACTLY {hop} steps) and 'therefore' (string).\n"
        "No extra text outside the <json> block. No code fences.\n\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n"
        f"Options: (A) {opts['A']} (B) {opts['B']} (C) {opts['C']} (D) {opts['D']}\n"
        f"Answer: {ansL} ({ansT})\n\n"
        "Constraints:\n"
        f"- 'because_steps' MUST contain exactly {hop} items.\n"
        "- Each step ONE short declarative sentence; keep entities consistent across steps.\n"
        "- Steps must be thematically coherent and non-redundant.\n"
        "- 'therefore' must directly support the given answer.\n\n"
        "Output format example (FORMAT ONLY):\n"
        "<json>{\"because_steps\":[\"step 1\",\"step 2\"],\"therefore\":\"final conclusion\"}</json>\n"
    )


# ------------------------------
# Parse model output
# ------------------------------
XML_JSON_RE = re.compile(r"<json>([\s\S]*?)</json>", re.I)


def parse_ref_only(text: str, hop: int) -> Optional[Dict]:
    m = XML_JSON_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict) and "because_steps" in data and "therefore" in data:
                steps = data["because_steps"]
                if isinstance(steps, list) and len(steps) == hop:
                    return {"steps": steps, "therefore": data["therefore"]}
        except Exception:
            return None
    return None


# ------------------------------
# Generation loop
# ------------------------------
def generate_from_dataset(client, base_rows, hop, n, writer, flush_every=1, verbose=False, debug_dir=None):
    rng = random.Random(42 + hop)
    written = 0
    for idx, ex in enumerate(base_rows[:n]):
        prompt = build_ref_prompt_from_example(ex, hop)
        for attempt in range(8):
            try:
                text = client.complete(prompt)
            except Exception as e:
                if verbose:
                    print(f"[hop{hop}] request error: {e}")
                time.sleep(1.2)
                continue
            ref = parse_ref_only(text, hop)
            if ref:
                rec = {
                    "id": f"{hop}-{idx:05d}",
                    "hop": hop,
                    "question": ex["question"],
                    "context": ex.get("context", ""),
                    "options": ex["options"],
                    "answer": ex["answer"],
                    "ref_chain": ref,
                }
                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written % flush_every == 0:
                    writer.flush()
                break
            else:
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    fail_path = os.path.join(
                        debug_dir,
                        f"fail_hop{hop}_idx{idx}_try{attempt}.txt",
                    )
                    with open(fail_path, "w", encoding="utf-8") as f:
                        f.write(text)
                if verbose:
                    print(f"[hop{hop}] parse fail item{idx} attempt {attempt + 1}")
        time.sleep(rng.uniform(0.05, 0.2))
    writer.flush()
    return written


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["arc_easy", "arc_challenge", "sciq", "csqa"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--provider", choices=["openai", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--hf-token", default=None)
    # Add 4-hop support.
    ap.add_argument("--hop", choices=["1", "2", "3", "4", "all"], default="all")
    ap.add_argument("--per-hop", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="ref_from_hf.jsonl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    client = LLMClient(
        LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            hf_token=args.hf_token,
        )
    )
    rows = load_hf_examples(args.dataset, args.split, args.limit)

    # Keep append mode so 1/2/3/4-hop chains can be written into the same JSONL file.
    with open(args.out, "a", encoding="utf-8") as wf:
        if args.hop == "all":
            for hop in (1, 2, 3, 4):
                print(f"[Hop {hop}] start: need={args.per_hop} (dataset size={len(rows)})")
                generate_from_dataset(
                    client,
                    rows,
                    hop,
                    args.per_hop,
                    wf,
                    verbose=args.verbose,
                    debug_dir=args.debug_dir,
                )
        else:
            hop = int(args.hop)
            print(f"[Hop {hop}] start: need={args.per_hop} (dataset size={len(rows)})")
            generate_from_dataset(
                client,
                rows,
                hop,
                args.per_hop,
                wf,
                verbose=args.verbose,
                debug_dir=args.debug_dir,
            )


if __name__ == "__main__":
    main()
