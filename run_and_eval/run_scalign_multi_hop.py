#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run 1/2/3/4-hop experiments in one script while minimizing parse failures:
- Prefer the Hugging Face chat_template (tokenizer.apply_chat_template) when available; otherwise fall back to a simple template.
- Add a "</json>" stopping criterion (StoppingCriteria) for local models.
- Apply robust post-parsing fixes: alias key handling, string-to-list conversion, and step truncation/padding.
- Metric: normalized JSD, with Score = 1 - SSD.
- Support repeated runs: --runs N writes <out_prefix>_r{1..N}_hop{1,2,3,4}.jsonl.
- Added: placeholder detection, full RawModelOutput logging for all branches, and accuracy computation directly from the model answer field.
- Added: resume support (--resume), which continues from the largest existing Idx in the output file.
- Added: SCOS (Semantic Consistency Optimization Sampling):
    * --scos-mode {none, rr, ts}
        - rr: use redundancy rate (number of consecutive repeated steps) as the metric and choose the minimum.
        - ts: use thematic shift (1 - average cosine similarity between adjacent steps) as the metric and choose the minimum.
    * --scos-k: sample k reasoning chains per question.
- Added: Self-Consistency CoT (SC-CoT):
    * --scot-k: number of self-consistency samples (>1 enables majority voting over answers).
- Added: Self-Consistent SCOS (SCOS + SC-CoT):
    * When --scot-k>1 and --scos-mode!=none and --scos-k>1 are all set:
        - First run SC-CoT majority voting over the answers from K chains.
        - Then select the best chain within the majority-answer subset using the SCOS metric (RR/TS).
- Added: SCOT + alignment-based chain selection (scot_align):
    * When --scot-k>1 and --align-select are set:
        - Sample K chains and use SC-CoT majority voting to determine the answer.
        - Within the majority-answer subset, choose the chain with the highest Alignment Score (JSD-based) as the final output.
- Added: SCOS-AL (alignment-based SCOS):
    * When --scos-al and --scos-k>1 are set:
        - Sample scos_k chains for the same question.
        - Compute the Alignment Score for each chain.
        - Do not perform majority voting; directly choose the chain with the highest Alignment Score as the final output.
"""

import os
import re
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------- Parse <json>{...}</json> ----------
XML_JSON_RE = re.compile(r"<json>([\s\S]*?)</json>", re.I)
CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n([\s\S]*?)```", re.M)
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")

def _try_json_loads(blob: str) -> Optional[Dict]:
    """Try multiple cleanup passes to recover approximate JSON."""
    if not blob:
        return None
    candidates = [
        blob,
        blob.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'"),
        blob.replace("\n", " ").replace("\t", " "),
        re.sub(r",\s*([}\]])", r"\1", blob),   # Remove trailing commas.
        blob.replace("'", '"'),
    ]
    for s in candidates:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def _pick_best_json(objs: List[Dict]) -> Optional[Dict]:
    """Pick the candidate that best matches the expected structure."""
    if not objs:
        return None
    scored = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        score = 0
        if "because_steps" in o: score += 2
        if "therefore" in o: score += 2
        for k in ("steps", "because", "chain", "rationale"):
            if k in o: score += 1
        for k in ("conclusion", "final", "answer", "result"):
            if k in o: score += 1
        scored.append((score, o))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def _balanced_json_slices(s: str, max_chunk_len: int = 100_000):
    """
    Scan a string for all substrings enclosed by balanced braces (approximate JSON blocks).
    Return an iterator over those substrings. Uses an O(n) stack scan instead of recursive regex.
    """
    if not s:
        return
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    chunk = s[start:i+1]
                    if len(chunk) <= max_chunk_len:
                        yield chunk
                    start = -1

def extract_json_obj(text: str) -> Optional[Dict]:
    """Robustly extract the first JSON object from model output whenever possible."""
    if not text:
        return None

    # 1) Prefer <json>...</json>.
    m = XML_JSON_RE.search(text)
    if m:
        blob = m.group(1).strip()
        obj = _try_json_loads(blob)
        if obj is not None:
            return obj
        # If parsing inside the tags fails, scan the blob for multiple JSON candidates.
        objs = []
        for chunk in _balanced_json_slices(blob):
            o = _try_json_loads(chunk)
            if o: objs.append(o)
        picked = _pick_best_json(objs)
        if picked is not None:
            return picked

    # 2) Fall back to a ```...``` code block.
    m = CODE_FENCE_RE.search(text)
    text2 = m.group(1) if m else text

    # 3) Scan the full text or code block for multiple JSON candidates.
    objs = []
    for chunk in _balanced_json_slices(text2):
        o = _try_json_loads(chunk)
        if o: objs.append(o)
    picked = _pick_best_json(objs)
    if picked is not None:
        return picked

    # 4) Final fallback: take the first {...} block.
    m = JSON_BLOCK_RE.search(text2)
    if not m:
        return None
    blob = m.group(0)
    return _try_json_loads(blob)

# ---------- Output normalization ----------
STEP_SPLIT_RE = re.compile(r"(?:^|\n|\r|\t|•|- |\d+\.\s+|[;；])+")

def normalize_chain(obj: Dict, L: int) -> Optional[Dict]:
    """
    Normalize outputs to: {"because_steps": List[str] with length==L, "therefore": str, "answer": "A|B|C|D"}.
    """
    if not isinstance(obj, dict):
        return None

    # Fallback aliases for because_steps.
    if "because_steps" not in obj:
        if "steps" in obj:
            obj["because_steps"] = obj.get("steps")
        elif "because" in obj:
            obj["because_steps"] = obj.get("because")
        else:
            for k in ("because", "chain", "rationale"):
                if k in obj:
                    obj["because_steps"] = obj[k]
                    break

    # Fallback aliases for therefore.
    if "therefore" not in obj:
        for k in ("conclusion", "final", "answer_text", "result"):
            if k in obj:
                obj["therefore"] = obj[k]
                break

    # Fallback aliases for answer.
    ans = obj.get("answer", None)
    if ans is None:
        for k in ("pred", "choice", "final_answer", "option"):
            if k in obj:
                ans = obj.get(k)
                break

    # Normalize steps to list[str].
    steps = obj.get("because_steps", [])
    if isinstance(steps, str):
        cand = [s.strip() for s in STEP_SPLIT_RE.split(steps) if s.strip()]
        steps = cand if cand else [steps.strip()]
    if not isinstance(steps, list):
        steps = [str(steps)]
    steps = [str(s).strip() for s in steps if str(s).strip()]

    # Align the number of steps: truncate extras or pad missing ones.
    if len(steps) > L:
        steps = steps[:L]
    while len(steps) < L:
        if len(steps) == 0:
            steps.append("Derive key fact from the question.")
        else:
            steps.append(steps[-1])

    there = str(obj.get("therefore", "")).strip()
    if not there:
        there = "Therefore, choose the best option."

    # Normalize answer: keep the first uppercase character and only accept A-D.
    answer = (str(ans).strip().upper() if ans is not None else "")
    answer = answer[:1] if answer else ""
    if answer not in {"A", "B", "C", "D"}:
        answer = ""  # Leave invalid answers empty; they will be rejected later.

    return {"because_steps": steps, "therefore": there, "answer": answer}

# ---------- Placeholder detection ----------
def looks_like_placeholder(steps: List[str], therefore: str) -> bool:
    text = " ".join([*(steps or []), therefore or ""]).lower()
    bad_tokens = ["step 1", "step1", "step 2", "step2", "final", "placeholder"]
    return any(bt in text for bt in bad_tokens)

# ---------- Dataset example -> question structure ----------
def dataset_example_to_article(name: str, ex: Dict) -> Dict:
    """
    Return: {question, context, options{A..D}, answer}
    """
    name = name.lower()
    if name in ("arc_easy", "arc_challenge"):
        q = ex.get("question", "").strip()
        ch = ex.get("choices", {})
        labels, texts = ch.get("label", []), ch.get("text", [])
        mp = {str(l).upper(): str(t).strip() for l, t in zip(labels, texts)}
        ansL = str(ex.get("answerKey", "")).upper().strip()
        return {
            "question": q,
            "context": "",
            "options": {k: mp.get(k, "") for k in ["A", "B", "C", "D"]},
            "answer": ansL if ansL in {"A","B","C","D"} else ""
        }
    elif name == "sciq":
        q = ex.get("question", "").strip()
        sup = ex.get("support", "").strip()
        ans_txt = ex.get("correct_answer", "").strip()
        ds = [ex.get("distractor1", ""), ex.get("distractor2", ""), ex.get("distractor3", "")]
        opts = [ans_txt] + [d for d in ds if d]
        letters = ["A", "B", "C", "D"]
        random.shuffle(opts)
        options = {L: T for L, T in zip(letters, opts)}
        # Recover the correct option letter.
        goldL = ""
        for L, T in options.items():
            if T.strip().lower() == ans_txt.strip().lower():
                goldL = L
                break
        return {"question": q, "context": sup, "options": options, "answer": goldL}
    elif name == "csqa":
        q = ex.get("question", "").strip()
        ch = ex.get("choices", [])
        mp = {c.get("label", "").upper(): c.get("text", "").strip() for c in ch}
        ansL = str(ex.get("answerKey", "")).upper().strip()
        return {
            "question": q,
            "context": "",
            "options": {k: mp.get(k, "") for k in ["A", "B", "C", "D"]},
            "answer": ansL if ansL in {"A","B","C","D"} else ""
        }
    else:
        raise ValueError("dataset not supported")

# ---------- Stop generation once "</json>" is seen ----------
class SubstringStopper(StoppingCriteria):
    def __init__(self, stop_str: str, tokenizer: AutoTokenizer):
        super().__init__()
        self.stop_str = stop_str
        self.tokenizer = tokenizer
        self.window = max(16, len(stop_str) * 3)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        tail = seq[-self.window:] if len(seq) > self.window else seq
        txt = self.tokenizer.decode(tail, skip_special_tokens=True)
        return self.stop_str.lower() in txt.lower()

# ---------- LLM client ----------
class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_tokens: int = 640,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.model_lower = (model or "").lower()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        if provider == "openai":
            from openai import OpenAI
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            self.client = OpenAI()
            self.pipe = None
            self.tokenizer = None
        elif provider == "hf":
            self.client = None
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=model,
                device_map="auto",
                model_kwargs={"torch_dtype": "auto"},
                trust_remote_code=True,
            )
            self.tokenizer = self.pipe.tokenizer
            if self.pipe.model.config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
                self.pipe.model.config.eos_token_id = self.tokenizer.eos_token_id
            if self.pipe.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                self.pipe.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            raise ValueError("provider must be 'openai' or 'hf'")

    def _truncate_to_json_end(self, text: str) -> str:
        low = (text or "").lower()
        pos = low.find("</json>")
        if pos != -1:
            return text[: pos + len("</json>")]
        return text

    def _build_chat(self, system_prompt: str, user_prompt: str) -> str:
        tok = self.tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]\n"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            # Use the Responses API for o* reasoning models.
            is_reasoning = self.model_lower.startswith("o")
            if is_reasoning:
                hard_instructions = (
                    "You must output EXACTLY ONE block in the following format and NOTHING ELSE:\n"
                    '<json>{"because_steps": ["STEP 1", "STEP 2", "..."], "therefore": "FINAL SENTENCE", "answer": "A"}</json>\n'
                    "The field 'answer' MUST be one of A, B, C, or D.\n"
                    "Do not add explanations, prefaces, or any text before or after the block."
                )
                merged = (
                    system_prompt.strip() + "\n\n" +
                    hard_instructions + "\n\n" +
                    user_prompt.strip()
                )

                resp = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": merged}],
                    reasoning={"effort": "medium"},
                    max_output_tokens=self.max_tokens,
                )
                text = getattr(resp, "output_text", None)
                if not text:
                    try:
                        parts = []
                        for item in resp.output:
                            if hasattr(item, "content"):
                                for c in item.content:
                                    if getattr(c, "type", "") in ("output_text", "text"):
                                        val = getattr(getattr(c, "text", ""), "value", None)
                                        if not val and hasattr(c, "content"):
                                            val = getattr(c.content, "text", None)
                                        parts.append(val or "")
                        text = "".join(parts).strip() if parts else ""
                    except Exception:
                        text = ""
                text = (text or "").strip()
                text = self._truncate_to_json_end(text)  # Truncate consistently at </json>.
                return text

            # Other OpenAI models: use Chat Completions.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=["</json>"],  # Helps non-reasoning models stop cleanly.
            )
            text = resp.choices[0].message.content.strip()
            text = self._truncate_to_json_end(text)  # Truncate consistently at </json>.
            return text

        # Hugging Face branch: chat_template + stopping criteria.
        full = self._build_chat(system_prompt, user_prompt)
        stopper = StoppingCriteriaList([SubstringStopper("</json>", self.tokenizer)])
        do_sample_flag = (self.temperature > 0.0) or (self.top_p < 1.0)
        outs = self.pipe(
            full,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=do_sample_flag,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            stopping_criteria=stopper,
            return_full_text=False,
        )
        text = outs[0]["generated_text"].strip()
        text = self._truncate_to_json_end(text)
        return text

# ---------- Build the CoT prompt (example=reference, target=test question) ----------
def build_cot_prompt_with_ref_example(ref_sample: Dict, target_item: Dict, L_steps: int) -> Tuple[str, str]:
    rq = ref_sample.get("question", "")
    rctx = ref_sample.get("context", "")
    ropts = ref_sample.get("options", {})
    ransL = ref_sample.get("answer", "")
    rsteps = (ref_sample.get("ref_chain", {}) or {}).get("steps") or (ref_sample.get("ref_chain", {}) or {}).get("because_steps") or []
    rthere = (ref_sample.get("ref_chain", {}) or {}).get("therefore", "")

    tq = target_item["question"]
    tctx = target_item.get("context", "")
    topts = target_item["options"]

    # Example JSON (includes an answer field for format guidance).
    ref_json_block = {
        "because_steps": [str(s).strip() for s in rsteps[:L_steps]],
        "therefore": str(rthere).strip(),
        "answer": ransL if ransL in {"A","B","C","D"} else "A"
    }

    sys = "You are a careful reasoning assistant. Output ONLY one <json>{...}</json> block, no extra text."
    user = (
        "Here is a worked example of multiple-choice reasoning with a concise chain.\n\n"
        f"Example Question: {rq}\n"
        f"Example Context: {rctx}\n"
        f"Example Options: (A) {ropts.get('A','')} (B) {ropts.get('B','')} (C) {ropts.get('C','')} (D) {ropts.get('D','')}\n"
        f"Example Answer: {ransL}\n"
        "Example Chain (strict JSON):\n"
        f"<json>{json.dumps(ref_json_block, ensure_ascii=False)}</json>\n\n"
        "Now solve the NEW problem in the SAME STYLE and EXACTLY the SAME NUMBER OF STEPS.\n"
        f"- Produce EXACTLY {L_steps} steps in 'because_steps'.\n"
        "- Keep each step to ONE short sentence; keep entities consistent; avoid hedging.\n"
        "- Output ONLY one <json>{\"because_steps\":[\"...\"], \"therefore\":\"...\", \"answer\":\"A\"}</json> block. No extra text.\n"
        "- The field 'answer' MUST be one of A, B, C, or D and MUST be a single letter (e.g., \"A\").\n\n"
        f"Question: {tq}\n"
        f"Context: {tctx}\n"
        f"Options: (A) {topts.get('A','')} (B) {topts.get('B','')} (C) {topts.get('C','')} (D) {topts.get('D','')}\n"
    )
    return sys, user

# ---------- NLI entropy (roberta-large-mnli) ----------
class SemanticEntropyCalculator:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        id2label = {int(k): v for k, v in self.model.config.id2label.items()} if hasattr(self.model.config, "id2label") else self.model.config.id2label
        self.entail_idx = [k for k, v in id2label.items() if v.lower() == "entailment"][0]
        self.neutral_idx = [k for k, v in id2label.items() if v.lower() == "neutral"][0]
        self.contra_idx = [k for k, v in id2label.items() if v.lower() == "contradiction"][0]

    @torch.no_grad()
    def entropy(self, s1: str, s2: str) -> float:
        inputs = self.tokenizer.encode_plus(s2, s1, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        p = np.array([probs[self.entail_idx], probs[self.neutral_idx], probs[self.contra_idx]], dtype=np.float64)
        p = np.clip(p / (p.sum() + 1e-12), 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    @torch.no_grad()
    def probs(self, s1: str, s2: str) -> np.ndarray:
        inputs = self.tokenizer.encode_plus(s2, s1, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        p = np.array([probs[self.entail_idx], probs[self.neutral_idx], probs[self.contra_idx]], dtype=np.float64)
        p = np.clip(p / (p.sum() + 1e-12), 1.0e-12, 1.0)
        return p

def build_upper_tri_entropy_distribution(segs: List[str], calc: SemanticEntropyCalculator) -> np.ndarray:
    L = len(segs)
    assert L >= 2
    vals = []
    for i in range(L):
        for j in range(i + 1, L):
            vals.append(calc.entropy(segs[i], segs[j]))
    H = np.array(vals, dtype=np.float64)
    s = float(H.sum())
    if s <= 1e-12:
        H[:] = 1.0 / max(len(H), 1)
    else:
        H /= s
    return H

# ---------- SCOS: sentence embeddings + RR / TS metrics ----------
class SentenceEmbedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        attn_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        masked = last_hidden * attn_mask
        summed = masked.sum(dim=1)
        counts = attn_mask.sum(dim=1).clamp(min=1e-9)
        embs = summed / counts
        return embs.detach().cpu().numpy()

def scos_redundancy_metric(steps: List[str]) -> float:
    """RR: number of consecutively repeated identical steps."""
    if not steps:
        return 0.0
    cnt = 0
    for i in range(len(steps) - 1):
        if steps[i].strip().lower() == steps[i+1].strip().lower():
            cnt += 1
    return float(cnt)

def scos_thematic_shift_metric(steps: List[str], embedder: Optional[SentenceEmbedder]) -> float:
    """TS: 1 - the average cosine similarity between adjacent steps."""
    if embedder is None or len(steps) < 2:
        return 0.0
    embs = embedder.encode(steps)  # [L, H]
    sims = []
    for i in range(len(steps) - 1):
        a = embs[i]
        b = embs[i+1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        sims.append(float(np.dot(a, b) / denom))
    if not sims:
        return 0.0
    return float(1.0 - np.mean(sims))

# ---------- JSD (normalized to [0,1]) ----------
def _kl_pq(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def jsd_norm(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * _kl_pq(p, m) + 0.5 * _kl_pq(q, m)
    return float(np.clip(jsd / np.log(2.0), 0.0, 1.0))

def ssd_and_score(H_ref: np.ndarray, H_res: np.ndarray) -> Tuple[float, float]:
    ssd = jsd_norm(H_ref, H_res)
    return ssd, 1.0 - ssd

# ---------- Alignment Score computation ----------
def compute_alignment_score(
    ref_steps: List[str],
    ref_there: str,
    res_steps: List[str],
    res_there: str,
    base_L: int,
    nli_calc: SemanticEntropyCalculator,
) -> Tuple[float, float]:
    """
    Return: (SSD, Score), where Score = 1 - SSD.
    - base_L == 1: compute JSD directly over the NLI probability distributions.
    - base_L >= 2: build segment entropy distributions H_ref and H_res, then compute JSD.
    """
    if base_L == 1:
        p_ref = nli_calc.probs(ref_steps[0], ref_there)
        p_res = nli_calc.probs(res_steps[0], res_there)
        ssd = jsd_norm(p_ref, p_res)
        score = 1.0 - ssd
        return ssd, score
    else:
        ref_segs_aug = ref_steps + [ref_there]
        res_segs_aug = res_steps + [res_there]
        H_ref = build_upper_tri_entropy_distribution(ref_segs_aug, nli_calc)
        H_res = build_upper_tri_entropy_distribution(res_segs_aug, nli_calc)
        return ssd_and_score(H_ref, H_res)

# ---------- Simple resume helper: read the maximum Idx in a file ----------
def find_max_idx_in_file(path: str) -> int:
    """
    Return the maximum Idx already present in the JSONL file; return -1 if the file is missing or empty.
    Idx values are considered regardless of whether Status is ok.
    """
    if not os.path.exists(path):
        return -1
    max_idx = -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "Idx" in obj:
                    try:
                        i = int(obj["Idx"])
                        if i > max_idx:
                            max_idx = i
                    except Exception:
                        pass
    except Exception:
        return -1
    return max_idx

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-all", required=True, help="JSONL containing all hop levels (each line should include hop and ref_chain.steps/therefore).")
    ap.add_argument("--dataset", required=True, choices=["arc_easy", "arc_challenge", "sciq", "csqa"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--llm-provider", choices=["openai", "hf"], required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--nli-model", default="roberta-large-mnli")
    ap.add_argument("--out-prefix", required=True, help="Output prefix. The script writes <prefix>_hop{1,2,3,4}.jsonl or, for repeated runs, <prefix>_rK_hopH.jsonl.")
    ap.add_argument("--limit-per-hop", type=int, default=256, help="Maximum number of examples to process for each hop.")
    ap.add_argument("--start-per-hop", type=int, default=0, help="Starting index for each hop (useful for manual resume).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=1, help="Number of full repeated runs; each run writes a full set of hop{1,2,3,4} files.")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep interval between examples, in seconds.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from the largest existing Idx in the output file.")
    # --- SCOS arguments ---
    ap.add_argument("--scos-mode", choices=["none", "rr", "ts"], default="none",
                    help="SCOS selection mode: none=off; rr=redundancy rate; ts=thematic shift.")
    ap.add_argument("--scos-k", type=int, default=1, help="Number of reasoning chains sampled per question for SCOS (k).")
    ap.add_argument("--scos-embed-model", default="sentence-transformers/all-mpnet-base-v2",
                    help="Sentence embedding model used for SCOS in TS mode.")
    # --- Self-Consistency CoT arguments ---
    ap.add_argument("--scot-k", type=int, default=1,
                    help="Number of Self-Consistency CoT samples; >1 enables answer voting.")
    # --- Alignment-based chain selection arguments ---
    ap.add_argument("--align-select", action="store_true",
                    help="In SCOT mode (scot-k>1), first determine the answer by SC-CoT majority vote, then choose the chain with the highest Alignment Score within the majority-answer subset.")
    # --- SCOS-AL: directly choose the best chain by Alignment Score among k samples ---
    ap.add_argument("--scos-al", action="store_true",
                    help="SCOS-AL mode: sample scos-k chains per question and directly choose the one with the highest Alignment Score as the final output (no majority voting).")
    # --- Choose the hop level(s) to run (1/2/3/4/all) ---
    ap.add_argument("--hop", choices=["1", "2", "3", "4", "all"], default="all",
                    help="Hop level(s) to evaluate (1/2/3/4/all). Default: all.")
    args = ap.parse_args()

    # Read the reference file and bucket records by hop (1/2/3/4).
    buckets = {1: [], 2: [], 3: [], 4: []}
    with open(args.jsonl-all if hasattr(args, "jsonl-all") else args.jsonl_all, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                rec = json.loads(l)
            except Exception:
                continue
            try:
                h = int(rec.get("hop", 0))
            except Exception:
                h = 0
            if h in buckets:
                buckets[h].append(rec)

    # --- Determine the strategy: baseline / scos / scot / scos_scot / scot_align / scos_al ---
    use_scot_raw = args.scot_k > 1
    use_scos_raw = (args.scos_mode != "none" and args.scos_k > 1)
    use_align_select = bool(args.align_select and use_scot_raw)
    use_scos_al = bool(args.scos_al and args.scos_k > 1)

    if use_scot_raw and use_scos_raw and not use_scos_al:
        strategy = "scos_scot"    # Self-Consistent SCOS
        use_scot = True
        use_scos = True
    elif use_align_select:
        strategy = "scot_align"   # SCOT + alignment-based chain selection.
        use_scot = True
        use_scos = False
    elif use_scot_raw:
        strategy = "scot"
        use_scot = True
        use_scos = False
    elif use_scos_al:
        strategy = "scos_al"      # Directly pick the best chain by Alignment Score among K samples.
        use_scot = False
        use_scos = False
    elif use_scos_raw:
        strategy = "scos"
        use_scot = False
        use_scos = True
    else:
        strategy = "baseline"
        use_scot = False
        use_scos = False

    # Core components
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Use a slightly higher temperature for sampling-based strategies.
    base_temp = 0.4
    if strategy in {"scos", "scot", "scos_scot", "scot_align", "scos_al"}:
        if (use_scos and args.scos_k > 1) or (use_scot and args.scot_k > 1) or (strategy == "scos_al" and args.scos_k > 1):
            base_temp = 0.7

    llm = LLMClient(
        args.llm_provider,
        args.llm_model,
        temperature=base_temp,
        top_p=0.9,
        max_tokens=640,
        api_key=args.openai_api_key,
    )
    nli_calc = SemanticEntropyCalculator(model_name=args.nli_model)

    # Falcon-specific stricter prompt control to avoid outputs like "answer": "A|B|C|D".
    is_falcon = (args.llm_provider == "hf") and ("falcon" in args.llm_model.lower())

    # Sentence embeddings required by SCOS.
    embedder = None
    if use_scos and args.scos_mode == "ts":
        try:
            embedder = SentenceEmbedder(args.scos_embed_model)
        except Exception as e:
            print(f"[WARN] Failed to init SentenceEmbedder({args.scos_embed_model}): {e}")
            embedder = None

    # Dataset cache
    ds_cache = {}
    def sample_target(rng_local: random.Random):
        if args.dataset not in ds_cache:
            subset = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}.get(args.dataset, None)
            ds_cache[args.dataset] = load_dataset("ai2_arc", subset, split=args.split) if subset \
                else load_dataset("sciq" if args.dataset == "sciq" else "commonsense_qa", split=args.split)
        ds = ds_cache[args.dataset]
        ex = ds[int(rng_local.random() * len(ds))]
        return dataset_example_to_article(args.dataset, ex)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # Resolve the list of hop levels to run.
    if args.hop == "all":
        hops_to_run = (1, 2, 3, 4)
    else:
        hops_to_run = (int(args.hop),)

    # Repeated full runs
    for run_idx in range(args.runs):
        seed_i = args.seed + run_idx
        random.seed(seed_i)
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
        try:
            from transformers import set_seed
            set_seed(seed_i)
        except Exception:
            pass
        rng = random.Random(seed_i)

        out_prefix_run = f"{args.out_prefix}_r{run_idx+1}" if args.runs > 1 else args.out_prefix
        if args.verbose:
            print(
                f"=== Run {run_idx+1}/{args.runs} | seed={seed_i} | strategy={strategy} | "
                f"SCOS={args.scos_mode}, scos_k={args.scos_k}, use_scos={use_scos} | "
                f"SCOT_k={args.scot_k}, use_scot={use_scot}, align_select={args.align_select}, scos_al={args.scos_al} | "
                f"hops={hops_to_run} ==="
            )

        # Run the pipeline for each hop.
        for hop in hops_to_run:
            refs = buckets.get(hop, [])
            if not refs:
                if args.verbose:
                    print(f"[hop {hop}] no records, skip.")
                continue

            out_path = f"{out_prefix_run}_hop{hop}.jsonl"

            # ---------- Compute the resume start index ----------
            start_base = args.start_per_hop
            if args.resume:
                last_idx = find_max_idx_in_file(out_path)
                resume_from = last_idx + 1
                start = max(start_base, resume_from)
            else:
                start = start_base

            end = min(len(refs), start + args.limit_per_hop)

            if start >= end:
                if args.verbose:
                    if args.resume:
                        print(f"[hop {hop}] nothing to do (resume detected up-to-date). out={out_path}")
                    else:
                        print(f"[hop {hop}] nothing to do in range [{start}:{end}). out={out_path}")
                if args.resume and args.verbose:
                    print(f"[hop {hop}] current file max Idx = {find_max_idx_in_file(out_path)}")
                continue

            wf = open(out_path, "a", encoding="utf-8")

            # Accuracy tracking
            total_ok = 0     # Number of successfully parsed and evaluated samples (Status=ok).
            correct_cnt = 0  # Number of correct predictions.

            if args.verbose:
                info_resume = ""
                if args.resume:
                    info_resume = f" | resume_from={start}"
                print(f"[hop {hop}] total={len(refs)}  run=[{start}:{end})  -> {out_path}{info_resume}")

            # --- Define the single-sample generation function ---
            def run_single_sample(sys_prompt, user_prompt, base_L, target_item):
                attempts = []

                raw = llm.complete(sys_prompt, user_prompt)
                attempts.append(("main", raw))
                data = extract_json_obj(raw)
                norm = normalize_chain(data, base_L) if data else None
                if norm and not norm.get("answer"):
                    norm = None

                # retry1
                if not norm:
                    if is_falcon:
                        hard_user = user_prompt + (
                            "\n\nIMPORTANT: Output ONLY ONE line exactly like:\n"
                            '<json>{"because_steps": ["STEP 1", "STEP 2", "..."], "therefore": "FINAL SENTENCE", "answer": "C"}</json>\n'
                            "The field 'answer' MUST be a single letter among A, B, C, or D (for example: \"A\").\n"
                            "Never output \"A|B|C|D\" or any string that contains all four options as the answer.\n"
                            "No extra text before or after the JSON block."
                        )
                    else:
                        hard_user = user_prompt + (
                            "\n\nIMPORTANT: Output ONLY ONE line exactly like:\n"
                            "<json>{\"because_steps\": [\"...\"], \"therefore\": \"...\", \"answer\": \"A|B|C|D\"}</json>\n"
                            "The field 'answer' MUST be one of A, B, C, or D. No extra text."
                        )
                    raw2 = llm.complete(sys_prompt, hard_user)
                    attempts.append(("retry1", raw2))
                    data2 = extract_json_obj(raw2)
                    norm = normalize_chain(data2, base_L) if data2 else None
                    if norm and not norm.get("answer"):
                        norm = None
                    if norm:
                        raw = raw2

                # retry2
                if not norm:
                    if is_falcon:
                        extreme_user = (
                            "Return ONE block ONLY, strictly:\n"
                            '<json>{"because_steps": ["STEP 1", "STEP 2", "..."], "therefore": "FINAL SENTENCE", "answer": "C"}</json>\n'
                            f"Rules:\n- Provide EXACTLY {base_L} concrete, domain-specific steps for THIS question.\n"
                            "- Each step must mention concrete entities or facts from the question.\n"
                            "- DO NOT output the literals 'STEP 1', 'STEP 2', 'FINAL', 'placeholder', or any generic meta-instructions.\n"
                            "- The field 'answer' MUST be a single letter among A, B, C, or D (e.g., \"A\").\n"
                            "- NEVER output \"A|B|C|D\" or any concatenation of all four options (like \"ABCD\") as the answer.\n"
                            "- No text outside the <json> block.\n\n"
                            f"Question: {target_item['question']}\n"
                            f"Context: {target_item.get('context','')}\n"
                            f"Options: {json.dumps(target_item['options'], ensure_ascii=False)}\n"
                        )
                    else:
                        extreme_user = (
                            "Return ONE block ONLY, strictly:\n"
                            "<json>{\"because_steps\": [\"...\"], \"therefore\": \"...\", \"answer\": \"A|B|C|D\"}</json>\n"
                            f"Rules:\n- Provide EXACTLY {base_L} concrete, domain-specific steps for THIS question.\n"
                            "- DO NOT output the literals 'STEP 1', 'STEP 2', 'FINAL', or placeholders.\n"
                            "- The field 'answer' MUST be one of A, B, C, or D.\n"
                            "- No text outside the <json> block.\n\n"
                            f"Question: {target_item['question']}\n"
                            f"Context: {target_item.get('context','')}\n"
                            f"Options: {json.dumps(target_item['options'], ensure_ascii=False)}\n"
                        )

                    raw3 = llm.complete(sys_prompt, extreme_user)
                    attempts.append(("retry2", raw3))
                    data3 = extract_json_obj(raw3)
                    norm = normalize_chain(data3, base_L) if data3 else None
                    if norm and not norm.get("answer"):
                        norm = None
                    if norm:
                        raw = raw3

                if not norm:
                    return None, raw, attempts, "no_json_or_missing_keys_or_invalid_answer"

                res_steps = norm["because_steps"]
                there_res = norm["therefore"]
                model_answer = norm.get("answer", "")

                # Reject placeholder outputs.
                if looks_like_placeholder(res_steps, there_res):
                    return None, raw, attempts, "placeholder_output"

                # Validate the answer field.
                if model_answer not in {"A", "B", "C", "D"}:
                    return None, raw, attempts, "invalid_answer"

                return norm, raw, attempts, None

            # ---------- Main loop ----------
            for idx in range(start, end):
                ref = refs[idx]

                # Read the reference because_steps and therefore fields.
                steps = (ref.get("ref_chain", {}) or {}).get("steps") or (ref.get("ref_chain", {}) or {}).get("because_steps") or []
                steps = [str(s).strip() for s in steps if str(s).strip()]
                there_ref = str((ref.get("ref_chain", {}) or {}).get("therefore", "")).strip()

                base_L = len(steps)  # Original number of because steps (1..4).
                if base_L < 1:
                    rec_out = {
                        "Hop": hop, "Idx": idx, "Status": "skip_L<1",
                        "SCStrategy": strategy
                    }
                    wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                    if args.verbose: print(f"[hop {hop} idx {idx}] skip base_L={base_L}")
                    time.sleep(args.sleep); continue

                # Sample a target test question (including the gold answer).
                target_item = sample_target(rng)

                # Build the prompt.
                sys_prompt, user_prompt = build_cot_prompt_with_ref_example(ref, target_item, base_L)

                # ===== SCOT / SCOS_SCOT / SCOT_ALIGN =====
                if strategy in {"scot", "scos_scot", "scot_align"}:
                    # Number of samples: use max for SCOS+SCOT so both branches have enough candidates.
                    K = args.scot_k if strategy in {"scot", "scot_align"} else max(args.scot_k, args.scos_k)
                    candidates = []   # Each item: {norm, raw, attempts, answer}.
                    first_fail_tuple = None

                    for _ in range(K):
                        norm, raw, attempts, reason = run_single_sample(sys_prompt, user_prompt, base_L, target_item)
                        if norm is None:
                            if first_fail_tuple is None:
                                first_fail_tuple = (raw, attempts, reason)
                            continue
                        ans = norm.get("answer", "")
                        if ans not in {"A", "B", "C", "D"}:
                            if first_fail_tuple is None:
                                first_fail_tuple = (raw, attempts, "invalid_answer_after_norm")
                            continue
                        candidates.append({
                            "norm": norm,
                            "raw": raw,
                            "attempts": attempts,
                            "answer": ans,
                        })

                    if not candidates:
                        if first_fail_tuple is not None:
                            raw0, attempts0, reason0 = first_fail_tuple
                        else:
                            raw0, attempts0, reason0 = "", [], "scot_all_candidates_invalid"
                        rec_out = {
                            "Hop": hop,
                            "Idx": idx,
                            "Status": "parse_fail",
                            "Reason": reason0,
                            "SCStrategy": strategy,
                            "SCOSMode": args.scos_mode,
                            "SCOSK": args.scos_k,
                            "SCOTK": args.scot_k,
                            "RawModelOutput": (raw0 or "")[:2000],
                            "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts0],
                        }
                        wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                        if args.verbose:
                            snippet = (raw0 or "")[:260].replace("\n", " ")
                            print(f"[hop {hop} idx {idx}] parse fail ({strategy}) | reason={reason0} | raw[:260]: {snippet}")
                        time.sleep(args.sleep); continue

                    # Majority-vote answer.
                    vote_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
                    for c in candidates:
                        vote_counts[c["answer"]] += 1
                    voted_answer = max("ABCD", key=lambda x: vote_counts[x])

                    # Keep only candidates with the majority-vote answer.
                    majority_cands = [c for c in candidates if c["answer"] == voted_answer]
                    if not majority_cands:
                        majority_cands = candidates

                    ssd = None
                    score = None

                    if strategy == "scot":
                        # Pure SCOT: pick the first chain in the majority subset.
                        chosen = majority_cands[0]
                        best_metric = 0.0
                    elif strategy == "scot_align":
                        # Within the majority-answer subset, choose the chain with the highest Alignment Score.
                        best_metric = 0.0
                        best_score = None
                        best_ssd = None
                        chosen = None
                        for c in majority_cands:
                            steps_c = c["norm"]["because_steps"]
                            there_c = c["norm"]["therefore"]
                            try:
                                ssd_i, score_i = compute_alignment_score(
                                    steps, there_ref,
                                    steps_c, there_c,
                                    base_L, nli_calc
                                )
                            except Exception:
                                continue
                            if (best_score is None) or (score_i > best_score):
                                best_score = score_i
                                best_ssd = ssd_i
                                chosen = c
                        if chosen is None:
                            chosen = majority_cands[0]
                            best_score = 0.0
                            best_ssd = 0.0
                        ssd = best_ssd
                        score = best_score
                    else:
                        # Self-Consistent SCOS: choose the best chain by the SCOS metric within the majority subset.
                        best_metric = None
                        chosen = None
                        for c in majority_cands:
                            steps_c = c["norm"]["because_steps"]
                            if args.scos_mode == "rr":
                                metric = scos_redundancy_metric(steps_c)
                            elif args.scos_mode == "ts":
                                metric = scos_thematic_shift_metric(steps_c, embedder)
                            else:
                                metric = 0.0
                            if best_metric is None or metric < best_metric:
                                best_metric = metric
                                chosen = c
                        if chosen is None:
                            chosen = majority_cands[0]
                            best_metric = 0.0

                    norm = chosen["norm"]
                    raw = chosen["raw"]
                    attempts = chosen["attempts"]
                    res_steps = norm["because_steps"]
                    there_res = norm["therefore"]
                    model_answer = voted_answer  # Use the voted answer.

                    # === Alignment metric (JSD_norm) ===
                    if ssd is None or score is None:
                        try:
                            ssd, score = compute_alignment_score(
                                steps, there_ref,
                                res_steps, there_res,
                                base_L, nli_calc
                            )
                        except Exception as e:
                            rec_out = {
                                "Hop": hop, "Idx": idx, "Status": "nli_error", "Error": str(e),
                                "SCStrategy": strategy,
                                "RawModelOutput": (raw or "")[:2000],
                                "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts],
                            }
                            wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                            if args.verbose: print(f"[hop {hop} idx {idx}] nli error: {e}")
                            time.sleep(args.sleep); continue

                    # === Accuracy evaluation: use the voted answer ===
                    goldL = target_item.get("answer", "")
                    correct = int(bool(goldL) and (model_answer == goldL))
                    total_ok += 1
                    correct_cnt += correct

                    rec_out = {
                        "Hop": hop,
                        "Idx": idx,
                        "Status": "ok",
                        "SCStrategy": strategy,
                        "L_steps": base_L,
                        "L_effective": base_L + (0 if base_L == 1 else 1),
                        "Dataset": args.dataset,
                        "Split": args.split,
                        "LLMProvider": args.llm_provider,
                        "LLMModel": args.llm_model,
                        "NLIModel": args.nli_model,
                        "SCOSMode": args.scos_mode,
                        "SCOSK": args.scos_k,
                        "SCOSMetric": best_metric if best_metric is not None else 0.0,
                        "SCOTK": args.scot_k,
                        "SCOTVotes": vote_counts,
                        "RefQuestion": ref.get("question", ""),
                        "RefAnswer": ref.get("answer", ""),
                        "RefSteps": steps,
                        "RefTherefore": there_ref,
                        "TargetQuestion": target_item["question"],
                        "TargetOptions": target_item["options"],
                        "GoldAnswer": goldL,
                        "ModelAnswer": model_answer,
                        "Correct": correct,
                        "ResultSteps": res_steps,
                        "ResultTherefore": there_res,
                        "SSD": ssd,
                        "Score": score,
                        "RawModelOutput": (raw or "")[:2000],
                        "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts],
                    }
                    wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                    if args.verbose:
                        extra = ""
                        if strategy == "scos_scot" and best_metric is not None:
                            extra = f" metric={best_metric:.4f}"
                        if strategy == "scot_align":
                            extra = f" AlignScore={score:.4f}"
                        print(
                            f"[hop {hop} idx {idx}] ok({strategy})  L_steps={base_L}  "
                            f"Score={score:.4f}  SSD={ssd:.4f}  Ans={model_answer} Gold={goldL} "
                            f"{'✓' if correct else '✗'}  votes={vote_counts}{extra}"
                        )
                    time.sleep(args.sleep)
                    continue  # Move to the next question.

                # ===== SCOS-AL: choose the best chain by Alignment Score among K samples =====
                if strategy == "scos_al":
                    K = max(args.scos_k, 1)
                    best_norm = None
                    best_raw = ""
                    best_attempts = []
                    best_ssd = None
                    best_score = None
                    first_fail_tuple = None

                    for _ in range(K):
                        norm, raw, attempts, reason = run_single_sample(sys_prompt, user_prompt, base_L, target_item)
                        if norm is None:
                            if first_fail_tuple is None:
                                first_fail_tuple = (raw, attempts, reason)
                            continue

                        res_steps = norm["because_steps"]
                        there_res = norm["therefore"]

                        try:
                            ssd_i, score_i = compute_alignment_score(
                                steps, there_ref,
                                res_steps, there_res,
                                base_L, nli_calc
                            )
                        except Exception as e:
                            if first_fail_tuple is None:
                                first_fail_tuple = (raw, attempts, f"nli_error_during_scos_al: {e}")
                            continue

                        if (best_score is None) or (score_i > best_score):
                            best_score = score_i
                            best_ssd = ssd_i
                            best_norm = norm
                            best_raw = raw
                            best_attempts = attempts

                    if best_norm is None:
                        if first_fail_tuple is not None:
                            raw0, attempts0, reason0 = first_fail_tuple
                        else:
                            raw0, attempts0, reason0 = "", [], "scos_al_all_candidates_invalid"
                        rec_out = {
                            "Hop": hop,
                            "Idx": idx,
                            "Status": "parse_fail",
                            "Reason": reason0,
                            "SCStrategy": strategy,
                            "SCOSMode": args.scos_mode,
                            "SCOSK": args.scos_k,
                            "SCOTK": args.scot_k,
                            "RawModelOutput": (raw0 or "")[:2000],
                            "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts0],
                        }
                        wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                        if args.verbose:
                            snippet = (raw0 or "")[:260].replace("\n", " ")
                            print(f"[hop {hop} idx {idx}] parse fail ({strategy}) | reason={reason0} | raw[:260]: {snippet}")
                        time.sleep(args.sleep); continue

                    # The selected chain.
                    norm = best_norm
                    raw = best_raw
                    attempts = best_attempts
                    res_steps = norm["because_steps"]
                    there_res = norm["therefore"]
                    model_answer = norm.get("answer", "")

                    ssd = best_ssd if best_ssd is not None else 0.0
                    score = best_score if best_score is not None else 0.0

                    # === Accuracy evaluation: directly use the selected chain answer ===
                    goldL = target_item.get("answer", "")
                    correct = int(bool(goldL) and (model_answer == goldL))
                    total_ok += 1
                    correct_cnt += correct

                    rec_out = {
                        "Hop": hop,
                        "Idx": idx,
                        "Status": "ok",
                        "SCStrategy": strategy,
                        "L_steps": base_L,
                        "L_effective": base_L + (0 if base_L == 1 else 1),
                        "Dataset": args.dataset,
                        "Split": args.split,
                        "LLMProvider": args.llm_provider,
                        "LLMModel": args.llm_model,
                        "NLIModel": args.nli_model,
                        "SCOSMode": args.scos_mode,
                        "SCOSK": args.scos_k,
                        "SCOSMetric": 0.0,  # For SCOS-AL, the real selection metric is the Alignment Score.
                        "SCOTK": args.scot_k,
                        "RefQuestion": ref.get("question", ""),
                        "RefAnswer": ref.get("answer", ""),
                        "RefSteps": steps,
                        "RefTherefore": there_ref,
                        "TargetQuestion": target_item["question"],
                        "TargetOptions": target_item["options"],
                        "GoldAnswer": goldL,
                        "ModelAnswer": model_answer,
                        "Correct": correct,
                        "ResultSteps": res_steps,
                        "ResultTherefore": there_res,
                        "SSD": ssd,
                        "Score": score,
                        "RawModelOutput": (raw or "")[:2000],
                        "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts],
                    }
                    wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                    if args.verbose:
                        print(
                            f"[hop {hop} idx {idx}] ok({strategy})  L_steps={base_L}  "
                            f"Score={score:.4f}  SSD={ssd:.4f}  Ans={model_answer} Gold={goldL} "
                            f"{'✓' if correct else '✗'}  (K={K} candidates)"
                        )
                    time.sleep(args.sleep)
                    continue  # Move to the next question.

                # ===== Baseline / pure SCOS =====
                best_norm = None
                best_raw = ""
                best_attempts = []
                best_metric = None
                first_fail_tuple = None

                K = args.scos_k if use_scos else 1

                for _ in range(K):
                    norm, raw, attempts, reason = run_single_sample(sys_prompt, user_prompt, base_L, target_item)
                    if norm is None:
                        if first_fail_tuple is None:
                            first_fail_tuple = (raw, attempts, reason)
                        continue

                    # Compute the SCOS metric if it is enabled.
                    if use_scos:
                        if args.scos_mode == "rr":
                            metric = scos_redundancy_metric(norm["because_steps"])
                        elif args.scos_mode == "ts":
                            metric = scos_thematic_shift_metric(norm["because_steps"], embedder)
                        else:
                            metric = 0.0
                    else:
                        metric = 0.0  # baseline

                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_norm = norm
                        best_raw = raw
                        best_attempts = attempts

                    # Baseline: the first valid chain is enough.
                    if not use_scos:
                        break

                # If all candidates fail.
                if best_norm is None:
                    if first_fail_tuple is not None:
                        raw0, attempts0, reason0 = first_fail_tuple
                    else:
                        raw0, attempts0, reason0 = "", [], "scos_all_candidates_invalid"
                    rec_out = {
                        "Hop": hop,
                        "Idx": idx,
                        "Status": "parse_fail",
                        "Reason": reason0,
                        "SCStrategy": strategy,
                        "SCOSMode": args.scos_mode,
                        "SCOSK": args.scos_k,
                        "SCOTK": args.scot_k,
                        "RawModelOutput": (raw0 or "")[:2000],
                        "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts0],
                    }
                    wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                    if args.verbose:
                        snippet = (raw0 or "")[:260].replace("\n", " ")
                        print(f"[hop {hop} idx {idx}] parse fail ({strategy}) | reason={reason0} | raw[:260]: {snippet}")
                    time.sleep(args.sleep); continue

                # Use the best chain selected by SCOS or the baseline strategy.
                norm = best_norm
                raw = best_raw
                attempts = best_attempts

                res_steps = norm["because_steps"]
                there_res = norm["therefore"]
                model_answer = norm.get("answer", "")

                # === Alignment metric (JSD_norm) ===
                try:
                    ssd, score = compute_alignment_score(
                        steps, there_ref,
                        res_steps, there_res,
                        base_L, nli_calc
                    )
                except Exception as e:
                    rec_out = {
                        "Hop": hop, "Idx": idx, "Status": "nli_error", "Error": str(e),
                        "SCStrategy": strategy,
                        "RawModelOutput": (raw or "")[:2000],
                        "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts],
                    }
                    wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                    if args.verbose: print(f"[hop {hop} idx {idx}] nli error: {e}")
                    time.sleep(args.sleep); continue

                # === Accuracy evaluation: directly use the model answer ===
                goldL = target_item.get("answer", "")
                correct = int(bool(goldL) and (model_answer == goldL))
                total_ok += 1
                correct_cnt += correct

                # Write a successful record.
                rec_out = {
                    "Hop": hop,
                    "Idx": idx,
                    "Status": "ok",
                    "SCStrategy": strategy,
                    "L_steps": base_L,
                    "L_effective": base_L + (0 if base_L == 1 else 1),
                    "Dataset": args.dataset,
                    "Split": args.split,
                    "LLMProvider": args.llm_provider,
                    "LLMModel": args.llm_model,
                    "NLIModel": args.nli_model,
                    "SCOSMode": args.scos_mode,
                    "SCOSK": args.scos_k,
                    "SCOSMetric": best_metric if best_metric is not None else 0.0,
                    "SCOTK": args.scot_k,
                    "RefQuestion": ref.get("question", ""),
                    "RefAnswer": ref.get("answer", ""),
                    "RefSteps": steps,
                    "RefTherefore": there_ref,
                    "TargetQuestion": target_item["question"],
                    "TargetOptions": target_item["options"],
                    "GoldAnswer": goldL,
                    "ModelAnswer": model_answer,
                    "Correct": correct,
                    "ResultSteps": res_steps,
                    "ResultTherefore": there_res,
                    "SSD": ssd,
                    "Score": score,
                    "RawModelOutput": (raw or "")[:2000],
                    "AllRawOutputs": [{"Attempt": a, "Text": (t or "")[:2000]} for a, t in attempts],
                }
                wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); wf.flush()
                if args.verbose:
                    extra = ""
                    if use_scos and best_metric is not None:
                        extra = f"  SCOS({args.scos_mode}) metric={best_metric:.4f}"
                    print(
                        f"[hop {hop} idx {idx}] ok({strategy})  L_steps={base_L}  "
                        f"Score={score:.4f}  SSD={ssd:.4f}  Ans={model_answer} Gold={goldL} "
                        f"{'✓' if correct else '✗'}{extra}"
                    )

                time.sleep(args.sleep)

            wf.close()

            # Summarize accuracy at the hop level.
            if total_ok > 0:
                acc = correct_cnt / max(total_ok, 1)
                print(f"[hop {hop}] Accuracy ({strategy}): {correct_cnt}/{total_ok} = {acc:.4f}")
            else:
                print(f"[hop {hop}] Accuracy ({strategy}): N/A (no ok records)")

            if args.verbose:
                print(f"[hop {hop}] done -> {out_path}")

        if args.verbose:
            print(f"=== Run {run_idx+1}/{args.runs} done ===")

    print("All hops done.")

if __name__ == "__main__":
    main()
