"""LLaDA-8B — General ability + Instruction following (gen-based).

Benchmarks (full sets): MMLU, BBH, HumanEval, IFEval

Usage:
    BASE_MODEL_PATH=/path/to/llada-8b \\
    [LORA_PATH=/path/to/lora_dir] \\
    python run.py eval_llada_8b_general.py

Run inside the verified `opencompass` conda env — see requirements_trace.txt.
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.llada_8b import models
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import \
        humaneval_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# -----------------------------------------------------------------------------
# Per-task canvas (= max_out_len; for diffusion also the sampling-step budget).
# Identical across the 4 models (Dream / LLaDA / Qwen2.5 / Llama3) so the
# generation length budget is the same. Matched against dataset['abbr']
# (case-insensitive substring); first matching key wins. Sources:
#   - Dream eval/eval_dream_gen.sh; LLaDA EVAL.md (HumanEval 512)
#   - MMLU uses the simple_evals CoT template ("Think step by step ...
#     ANSWER: X"); the postprocessor strictly requires the ANSWER line, so 512
#     is needed (256 truncates chains-of-thought before the ANSWER line).
# -----------------------------------------------------------------------------
CANVAS_RULES = [
    ('mmlu',      512),
    ('bbh',       512),
    ('humaneval', 512),
    ('ifeval',    256),
]


def _resolve_canvas(abbr: str) -> int:
    a = abbr.lower()
    for key, n in CANVAS_RULES:
        if key in a:
            return n
    raise ValueError(
        f"No canvas configured for dataset abbr={abbr!r}. "
        f"Add an entry to CANVAS_RULES in {__file__}."
    )


# Full benchmark suite: MMLU (57 subjects), BBH (27 tasks), HumanEval, IFEval.
datasets = [*mmlu_datasets, *bbh_datasets, *humaneval_datasets, *ifeval_datasets]

# Apply per-task canvas. block_length is set equal to the canvas (gen_length):
# this is LLaDA-8B-Instruct's official setting in ML-GSAI/LLaDA EVAL.md — every
# non-math benchmark uses pure diffusion (block == gen_length). Semi-AR
# sampling (block < canvas) is documented there only for GSM8K/Math, where
# EVAL.md notes it helps math but "reduces accuracy elsewhere".
for _d in datasets:
    _canvas = _resolve_canvas(_d['abbr'])
    _inf = _d['infer_cfg']['inferencer']
    _inf['max_out_len']  = _canvas
    _inf['block_length'] = _canvas

# mmengine dumps the config dict to a .py file and reloads it; functions can't
# be serialized, so any top-level helper must be deleted before dump.
del _resolve_canvas, _d, _canvas, _inf

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1, num_split=1),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=OpenICLInferTask)),
)

# Collapse the 57 MMLU subjects and 27 BBH tasks into one score each, so the
# summary table has 4 rows per model instead of 86.
summarizer = dict(
    dataset_abbrs=['mmlu', 'bbh', 'openai_humaneval', 'IFEval'],
    summary_groups=[*mmlu_summary_groups, *bbh_summary_groups],
)

work_dir = './outputs/llada_8b_general'
