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
Runs the baselines / SC-Align methods and evaluates their performance.

- `run_scalign_multi_hop.py`: main experiment runner. Supports OpenAI API models and local Hugging Face models.
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

## 4. Notes

- All Chinese comments/docstrings in the provided scripts were converted to English.
- `judge/llm_judge.py` now reads the OpenAI key from `OPENAI_API_KEY` instead of a hard-coded value.
- A small output-path print bug in `llm_judge.py` was also fixed.
