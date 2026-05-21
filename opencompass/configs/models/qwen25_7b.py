from opencompass.models import HuggingFacewithChatTemplateResized


# NOTE: `import os` is intentionally placed *inside* the function bodies. This
# file is consumed by mmengine's lazy-import mechanism (see config.py:1081),
# which turns top-level `import os` into a proxy object — calling
# `os.environ.get()` on that proxy raises RuntimeError. Imports inside
# function bodies are NOT lazy-rewritten, so they get the real module.
def _read_env(name: str, required: bool = False, default: str = '') -> str:
    import os
    val = os.environ.get(name, default)
    if required and not val:
        raise EnvironmentError(
            f'Environment variable {name} is required. Set it before running, e.g.\n'
            f'  Bash:       {name}=/path/to/dir python run.py eval_<model>_general.py\n'
            f'  PowerShell: $env:{name}="C:/path/to/dir"; python run.py eval_<model>_general.py'
        )
    return val


BASE_PATH = _read_env('BASE_MODEL_PATH', required=True)
LORA_BASE = _read_env('LORA_PATH')  # optional: leave unset to skip LoRA variants
# Inference batch size; override via EVAL_BATCH_SIZE (the run_*.sh scripts set
# it). Default 4 is safe on a ~24 GB GPU; an H200 can go much higher.
BATCH_SIZE = int(_read_env('EVAL_BATCH_SIZE', default='4'))

# NOTE: Using HuggingFacewithChatTemplateResized (NOT HuggingFacewithChatTemplate)
# is required when LORA_PATH is set. The Trace LoRA was trained after
# `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)`
# (Trace_Results/inference/infer_single.py:246), which gives 151672 for
# Qwen2.5-7B-Instruct (len(tok)=151665). The vanilla HF wrapper skips that
# resize, so the base keeps the config-default 152064 and PEFT load fails
# with `size mismatch for ... embed_tokens.weight ([151672, 3584] vs
# [152064, 3584])`. Resized=True replicates the training-time resize.
_common = dict(
    type=HuggingFacewithChatTemplateResized,
    resize_to_multiple_of=8,
    path=BASE_PATH,
    tokenizer_path=BASE_PATH,
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    ),
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True,
        torch_dtype='bfloat16',               # HF-native string; aligned across 4 models
    ),
    generation_kwargs=dict(do_sample=False),  # 显式 greedy 解码
    max_out_len=512,                          # default; per-task override in eval script
    # Aligned with Trace's max_prompt_len=1024 (decoupled from gen canvas).
    max_seq_len=1024,
    batch_size=BATCH_SIZE,
    stop_words=['<|im_end|>', '<|endoftext|>'],  # Qwen2.5 chat 终止符
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [dict(abbr='qwen2.5-7b-instruct', **_common)]

if LORA_BASE:
    for i in range(8):
        models.append(dict(
            abbr=f'qwen2.5-7b-lora-{i}',
            peft_path=f'{LORA_BASE}/{i}',
            **_common,
        ))
