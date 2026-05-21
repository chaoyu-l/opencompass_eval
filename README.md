# opencompass_eval

Tests how the **general capability** of four LLMs changes — each base model vs
its 8 LoRA fine-tuned variants — on MMLU, BBH, HumanEval and IFEval.

| Model               | Type           | Run script                 |
|---------------------|----------------|----------------------------|
| DREAM-7B            | diffusion      | `run_dream_7b_general.sh`  |
| LLaDA-8B            | diffusion      | `run_llada_8b_general.sh`  |
| Llama-3-8B-Instruct | autoregressive | `run_llama3_8b_general.sh` |
| Qwen2.5-7B-Instruct | autoregressive | `run_qwen25_7b_general.sh` |

## 1. Install (once)

```bash
./install.sh
```

This creates the `opencompass` conda env, installs `requirements_trace.txt`
plus this package, and fetches the nltk data IFEval needs. `requirements_trace.txt`
is the verified, pinned dependency set — the only requirements file used.

## 2. Run

Open each `run_*.sh`, edit the **SETTINGS block** at the top, then just run it —
no command-line arguments needed:

```bash
./run_dream_7b_general.sh
./run_llada_8b_general.sh
./run_llama3_8b_general.sh
./run_qwen25_7b_general.sh
```

The SETTINGS block of each script holds:

- `BASE_MODEL_PATH` — the base model directory.
- `LORA_PATH` — a directory with 8 adapter subfolders `0/` … `7/`; set it to
  `""` to evaluate the base model only.
- `BATCH_SIZE` — inference batch size (default 16; an H200 handles 32–64).
- `CONDA_ENV` — the conda env from `install.sh` (default `opencompass`).

Run them one at a time (each is a long job). To resume a previous run, append
`-r latest` (add `-m eval` to only re-score), e.g.
`./run_dream_7b_general.sh -r latest`.

## 3. Results

Each run writes to `outputs/<model>_general/<timestamp>/`: `predictions/` (raw
outputs), `results/` (per-dataset scores), `summary/` (the score table). The
summary gives one score per benchmark × model — MMLU's 57 subjects and BBH's 27
tasks are aggregated automatically, so each run yields a 4-benchmark × 9-model
grid.

---

- Benchmark data goes under `./data/` (auto-downloaded on first run).
- Subjective eval (FollowBench / AlpacaEval-2): see `eval_*_subjective.py`.
- Implementation details and gotchas: see `CLAUDE.md`.
- Built on [OpenCompass](https://github.com/open-compass/opencompass) (Apache-2.0).
