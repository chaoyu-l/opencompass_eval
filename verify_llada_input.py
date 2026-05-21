"""Verify that OpenCompass `LLaDAModel` builds the same input as Trace's
`DataCollator.diffusion_call` LLaDA inference branch.

Runs both code paths against a small batch of sample prompts using the real
LLaDA-8B-Instruct tokenizer (no 8B weights loaded) and prints token IDs +
decoded strings for visual confirmation, plus an equality assertion.
"""
import os
import sys
import torch
from transformers import AutoTokenizer

LLADA_PATH = os.environ.get(
    'BASE_MODEL_PATH',
    '/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/llada-8b-instruct',
)

# ---- import the new OpenCompass build-inputs logic (without loading model) ----
# We replicate it here as a free function over a tokenizer so we don't have to
# instantiate the heavy LLaDAModel.
def opencompass_build_inputs(tokenizer, inputs, max_prompt_len, pad_id):
    ids_list = []
    for raw in inputs:
        text = raw if isinstance(raw, str) else str(raw)
        msgs = [{'role': 'user', 'content': text.strip()}]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
        )
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))

    max_len = max(t.numel() for t in ids_list)
    B = len(ids_list)
    padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, ids_t in enumerate(ids_list):
        l = ids_t.numel()
        padded_ids[i, max_len - l:] = ids_t
        padded_attn[i, max_len - l:] = 1
    return padded_ids, padded_attn


# ---- import Trace's DataCollator path ----
TRACE_DIR = '/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results'
sys.path.insert(0, TRACE_DIR)
from utils.data.data_collator import DataCollator  # noqa: E402


def main():
    print(f'[INFO] tokenizer path: {LLADA_PATH}')
    tokenizer = AutoTokenizer.from_pretrained(
        LLADA_PATH, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pad_id = tokenizer.pad_token_id

    # ---- sample prompts (mix of short / long, no DEFAULT_SYSTEM_PROMPT) ----
    sample_prompts = [
        'What is 2 + 2?',
        ('Question: Janet has 3 apples. She buys 5 more. How many does she '
         "have?\nLet's think step by step\nAnswer:"),
    ]

    # ---- OpenCompass new path ----
    max_prompt_len = 1024
    oc_ids, oc_attn = opencompass_build_inputs(
        tokenizer, sample_prompts, max_prompt_len=max_prompt_len, pad_id=pad_id)

    # ---- Trace path: DataCollator.diffusion_call (LLaDA inference branch) ----
    collator = DataCollator(
        tokenizer=tokenizer,
        model=None,
        padding='longest',
        max_prompt_len=max_prompt_len,
        max_ans_len=128,        # not used in inference branch
        pad_to_multiple_of=1,
        inference=True,
        is_diffusion=True,
        model_family='llada',
    )
    batch = [{'prompt': p, 'answer': ''} for p in sample_prompts]
    trace_out = collator(batch)
    tr_ids = trace_out['input_ids']
    tr_attn = trace_out['attention_mask']

    # ---- Print both ----
    for i, p in enumerate(sample_prompts):
        print(f'\n========== sample[{i}] ==========')
        print(f'raw prompt: {p!r}')
        print(f'OC  shape={tuple(oc_ids[i].shape)}  '
              f'attn_sum={int(oc_attn[i].sum())}')
        print(f'TR  shape={tuple(tr_ids[i].shape)}  '
              f'attn_sum={int(tr_attn[i].sum())}')
        # show first 60 ids
        print(f'OC ids[:60]: {oc_ids[i][:60].tolist()}')
        print(f'TR ids[:60]: {tr_ids[i][:60].tolist()}')
        print(f'OC tail decode: '
              f'{tokenizer.decode(oc_ids[i][int(oc_attn[i].sum() == 0):][-60:])!r}')
        print(f'TR tail decode: '
              f'{tokenizer.decode(tr_ids[i][int(tr_attn[i].sum() == 0):][-60:])!r}')

    # ---- Assertions ----
    assert oc_ids.shape == tr_ids.shape, (
        f'shape mismatch: OC={oc_ids.shape} TR={tr_ids.shape}')
    assert torch.equal(oc_ids, tr_ids), 'input_ids tensor differs!'
    assert torch.equal(oc_attn, tr_attn), 'attention_mask tensor differs!'

    # ---- Show one fully decoded prompt to confirm chat-template format ----
    print('\n========== full decoded chat-templated prompt (sample 0) ==========')
    full_decode = tokenizer.decode(oc_ids[0][oc_attn[0] == 1])
    print(repr(full_decode))

    print('\n[OK] OpenCompass LLaDA inputs == Trace DataCollator inputs '
          '(input_ids and attention_mask match exactly).')


if __name__ == '__main__':
    main()
