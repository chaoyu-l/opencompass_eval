"""LLaMA-3-8B-Instruct — General ability + Instruction following (gen-based).

Benchmarks (full sets): MMLU, BBH, HumanEval, IFEval

Usage:
    BASE_MODEL_PATH=/path/to/llama-3-8b-instruct \\
    [LORA_PATH=/path/to/lora_dir] \\
    python run.py eval_llama3_8b_general.py

Run inside the verified `opencompass` conda env — see requirements_trace.txt.
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.llama3_8b import models
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
# Per-task max_out_len — identical across the 4 models (Llama3 / Qwen2.5 /
# DREAM / LLaDA) so AR and diffusion share the same generation length budget.
# See eval_dream_7b_general.py for rationale and sources. Matched against
# dataset['abbr'] (case-insensitive substring); first matching key wins.
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
        f"No max_out_len configured for dataset abbr={abbr!r}. "
        f"Add an entry to CANVAS_RULES in {__file__}."
    )


# Full benchmark suite: MMLU (57 subjects), BBH (27 tasks), HumanEval, IFEval.
datasets = [*mmlu_datasets, *bbh_datasets, *humaneval_datasets, *ifeval_datasets]

# Apply per-task max_out_len; fail fast if any dataset has no matching rule.
for _d in datasets:
    _d['infer_cfg']['inferencer']['max_out_len'] = _resolve_canvas(_d['abbr'])

# mmengine dumps the config dict to a .py file and reloads it; functions can't
# be serialized, so any top-level helper must be deleted before dump.
del _resolve_canvas, _d

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

work_dir = './outputs/llama3_8b_general'
