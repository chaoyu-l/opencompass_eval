"""LLaMA-3-8B-Instruct — Subjective evaluation (FollowBench + AlpacaEval 2).

Usage:  python run.py eval_llama3_8b_subjective.py
"""
# =============================================================================
# >>>  在这里填写你的 OpenAI API 配置  <<<
# =============================================================================
OPENAI_API_KEY = 'sk-xxx'  # 替换为你的 API Key
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'
# =============================================================================

from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.llama3_8b import models
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval \
        import followbench_llmeval_datasets
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 \
        import alpacav2_datasets

from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

datasets = [*followbench_llmeval_datasets, *alpacav2_datasets]

judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-turbo',
    key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=4096,
    batch_size=8,
    temperature=0,
)]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=4,
                task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner,
                     models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=4,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = './outputs/llama3_8b_subjective'
