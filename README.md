# SC-Align Code Package

This package reorganizes the project into three folders:

```text
scalign_code_package/
├── make_ref/
│   └── make_ref.py
├── run_and_eval/
│   ├── run_scalign_multi_hop.py
│   └── eval.py
├── judge/
│   ├── llm_judge.py
│   └── parse_judge.py
└── requirements.txt
```

## 1. Folder overview

### `make_ref/`
Generates reference reasoning chains from Hugging Face datasets.

- `make_ref.py`: build 1/2/3/4-hop reference chains for datasets such as ARC, SciQ, and CSQA.

### `run_and_eval/`
Runs the baselines, SCOS (also referred to as ACSS in the paper), SC-CoT, and SC-Align methods, and evaluates their performance.

- `run_scalign_multi_hop.py`: main experiment runner. Supports OpenAI API models and local Hugging Face models, including baseline, SCOS/ACSS, SC-CoT, and SC-Align settings.
- `eval.py`: summarizes experiment results from generated JSONL files.

### `judge/`
Runs LLM-as-a-judge evaluation and parses the judge outputs.

- `llm_judge.py`: compares baseline reasoning chains and SC-Align reasoning chains with an LLM judge.
- `parse_judge.py`: summarizes judge scores and pairwise wins.

## 2. Environment setup

Use Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

For OpenAI-based scripts, set your API key before running:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 3. Quick usage

### A. Generate reference chains

```bash
python make_ref/make_ref.py \
  --dataset arc_challenge \
  --split validation \
  --provider openai \
  --model gpt-4o-mini \
  --hop all \
  --per-hop 256 \
  --out arcC_ref.jsonl
```

### B. Run baselines / SC-Align methods

Baseline example:

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --out-prefix results-gpt-4o-mini-baseline
```

SC-CoT example:

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --scot-k 5 \
  --out-prefix results-gpt-4o-mini-sccot
```

SCOS / ACSS examples:

SCOS-RR (ACSS-RR in the paper):

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --scos-mode rr \
  --scos-k 5 \
  --out-prefix results-gpt-4o-mini-scos-rr
```

SCOS-TS (ACSS-TS in the paper):

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --scos-mode ts \
  --scos-k 5 \
  --out-prefix results-gpt-4o-mini-scos-ts
```

SC-Align example:

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --scot-k 5 \
  --align-select \
  --out-prefix results-gpt-4o-mini-scalign
```

## Running with OpenAI API or Local Hugging Face Models

The main script supports both OpenAI API models and local/open-source models through Hugging Face.

### OpenAI API example

```bash
python run_and_eval/run_scalign_multi_hop.py \
  --jsonl-all arcC_ref.jsonl \
  --dataset arc_challenge \
  --split test \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --openai-api-key YOUR_API_KEY \
  --scot-k 5 \
  --align-select \
  --out-prefix results-gpt4omini-scalign

### C. Evaluate JSONL result files

```bash
python run_and_eval/eval.py
```

Update the filename patterns inside `eval.py` if your output names differ from the built-in patterns.

### D. Run LLM-as-a-judge evaluation

```bash
python judge/llm_judge.py \
  --baseline-jsonl results-gpt-4o-mini-sccot_r1_hop4.jsonl \
  --scalign-jsonl results-gpt-4o-mini-scalign_r1_hop4.jsonl \
  --out-jsonl judge_results.jsonl
```

### E. Parse judge outputs

```bash
python judge/parse_judge.py --judge-jsonl judge_results.jsonl
```

## 4. Method naming note

- In the code, `SCOS` refers to Semantic Consistency Optimization Sampling.
- In the paper, the same family may be referred to as `ACSS`.
- The two SCOS / ACSS variants included here are:
  - `SCOS-RR` / `ACSS-RR`: selection based on the redundancy-rate style metric.
  - `SCOS-TS` / `ACSS-TS`: selection based on the thematic-shift style metric.

