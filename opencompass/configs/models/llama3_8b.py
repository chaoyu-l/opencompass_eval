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

# Why HuggingFacewithChatTemplateResized (NOT HuggingFaceCausalLM, NOT vanilla
# HuggingFacewithChatTemplate):
#   1. Chat template (vs raw text): The Trace_Results LoRA was trained with
#      prompts wrapped by tokenizer.apply_chat_template(messages=[{user}],
#      add_generation_prompt=True) — see Trace_Results/utils/data/data_collator.py:363-373.
#      HF*ChatTemplate applies the same wrapping at eval time, keeping the
#      input distribution aligned with training. HuggingFaceCausalLM would
#      feed raw text, which is OUT-of-distribution for a chat-format-trained
#      adapter and silently tanks scores. This also matches the official
#      OpenCompass reference config at opencompass/configs/models/hf_llama/
#      hf_llama3_8b_instruct.py.
#   2. Resized embeddings (vs vanilla HF*ChatTemplate): Trace's training/
#      inference path always runs
#      `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)`
#      before LoRA loading (Trace_Results/inference/infer_single.py:246). The
#      LoRA's saved embed_tokens/lm_head therefore have shape
#      [ceil(len(tok)/8)*8, hidden]. The vanilla HF wrapper skips this
#      resize, so PEFT load fails with `size mismatch`. Resized replicates
#      the training-time resize.
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
    stop_words=['<|end_of_text|>', '<|eot_id|>'],  # Llama3 chat 终止符
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [dict(abbr='llama-3-8b-instruct', **_common)]

if LORA_BASE:
    for i in range(8):
        models.append(dict(
            abbr=f'llama-3-8b-lora-{i}',
            peft_path=f'{LORA_BASE}/{i}',
            **_common,
        ))
