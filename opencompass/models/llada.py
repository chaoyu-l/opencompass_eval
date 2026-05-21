import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS


# --------------- LLaDA generation helpers (from Trace_experiments) ---------------

def _add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise.clamp_min(1e-10))) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps,
                     device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def _get_mask_id(model, tokenizer) -> int:
    if getattr(tokenizer, 'mask_token_id', None) is not None:
        return int(tokenizer.mask_token_id)
    if getattr(model.config, 'mask_token_id', None) is not None:
        return int(model.config.mask_token_id)
    raise ValueError(
        'Cannot find mask_token_id in tokenizer or model config. '
        f'Tokenizer special tokens: {tokenizer.special_tokens_map}'
    )


@torch.no_grad()
def llada_generate(
    model,
    tokenizer,
    prompt_input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: int = 128,
    steps: int = 64,
    temperature: float = 0.0,
    block_length: int = 128,
    remasking: str = 'low_confidence',
):
    """Block-wise iterative demasking generation for LLaDA."""
    device = prompt_input_ids.device
    B, P = prompt_input_ids.shape
    gen_length = max_new_tokens
    mask_id = _get_mask_id(model, tokenizer)

    x = torch.full((B, P + gen_length), mask_id,
                    dtype=torch.long, device=device)
    x[:, :P] = prompt_input_ids.clone()

    full_attention_mask = None
    if attention_mask is not None:
        gen_mask = torch.ones((B, gen_length),
                              dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=-1)

    if gen_length % block_length != 0:
        block_length = gen_length
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks
        if steps == 0:
            steps = num_blocks
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start_idx = P + num_block * block_length
        end_idx = P + (num_block + 1) * block_length

        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = _get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            outputs = model(x, attention_mask=full_attention_mask)
            logits = outputs.logits

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1,
                                 index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((B, x.shape[1]), device=device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, end_idx:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=device)
            for j in range(B):
                current_k = num_transfer_tokens[j, i].item()
                if current_k > 0:
                    _, select_index = torch.topk(confidence[j], k=current_k)
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x


# --------------- OpenCompass model wrapper ---------------

@MODELS.register_module()
class LLaDAModel(BaseModel):
    """OpenCompass wrapper for the LLaDA diffusion language model.

    Uses iterative block-wise demasking for text generation.

    Args:
        path (str): Path to LLaDA model checkpoint.
        tokenizer_path (str): Path to tokenizer (defaults to ``path``).
        max_seq_len (int): Maximum input sequence length.
        block_length (int): Default block length when the eval script does not
            supply a per-task value. Per-task block_length is plumbed through
            GenInferencer.block_length → generate(block_length=...) and
            overrides this default.
        temperature (float): Gumbel noise temperature (0 = greedy).
        remasking (str): Remasking strategy, 'low_confidence' or 'random'.
        model_kwargs (dict): Extra kwargs for ``from_pretrained``.
        tokenizer_kwargs (dict): Extra kwargs for tokenizer loading.
        peft_path (str): Path to PEFT/LoRA adapter (merged at load time).
        peft_merge (bool): Whether to merge_and_unload LoRA weights.
        batch_size (int): Batch size for generation (unused by wrapper,
            consumed by OpenCompass runner).
    """

    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 2048,
        max_prompt_len: Optional[int] = None,
        block_length: int = 128,
        temperature: float = 0.0,
        remasking: str = 'low_confidence',
        model_kwargs: Optional[Dict] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        peft_path: Optional[str] = None,
        peft_merge: bool = True,
        meta_template: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            generation_kwargs=generation_kwargs or {},
        )
        self.block_length = block_length
        self.temperature = temperature
        self.remasking = remasking
        # Trace's separate prompt cap (their max_prompt_len, decoupled from
        # max_ans_len). None → fall back to max_seq_len at generate() time,
        # matching Dream/Qwen/Llama3 wrapper convention.
        self.max_prompt_len = max_prompt_len

        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs or {})
        self._load_model(path, model_kwargs or {}, peft_path, peft_merge)

    def _load_tokenizer(self, path: str, tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, **tokenizer_kwargs)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.bos_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise RuntimeError(
                'LLaDA wrapper requires a tokenizer with apply_chat_template; '
                'this checkpoint is chat-template-trained.'
            )

    def _load_model(self, path: str, model_kwargs: dict,
                    peft_path: Optional[str], peft_merge: bool):
        from transformers import AutoConfig, AutoModelForCausalLM

        model_kwargs = model_kwargs.copy()
        model_kwargs.setdefault('trust_remote_code', True)
        model_kwargs.setdefault('torch_dtype', torch.bfloat16)

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, config=config,
                attn_implementation='flash_attention_2', **model_kwargs)
        except (ValueError, ImportError, TypeError):
            self.model = AutoModelForCausalLM.from_pretrained(
                path, config=config, **model_kwargs)

        self.model.resize_token_embeddings(
            int(8 * math.ceil(len(self.tokenizer) / 8.0)))

        if self.tokenizer.eos_token_id is not None:
            self.model.config.end_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, peft_path, is_trainable=False)
            if peft_merge:
                self.model = self.model.merge_and_unload()

        if not hasattr(self.model, 'prepare_inputs_for_generation'):
            self.model.prepare_inputs_for_generation = (
                lambda input_ids, **kw: {'input_ids': input_ids})

        self.model.eval()

    def _build_inputs_trace_style(
        self, inputs: List[str], max_out_len: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Mirrors Trace's `DataCollator.diffusion_call` LLaDA inference branch
        (`Trace_Results/utils/data/data_collator.py:446-490`):

          1) wrap each input as a single user-role chat message;
          2) apply_chat_template(..., tokenize=True, add_generation_prompt=True)
             — produces the LLaDA-Instruct format
             ``<bos><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}
             <|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n``;
          3) per-sample left-truncate to max_prompt_len (preserves the
             assistant generation header);
          4) batch left-pad with pad_id; attention_mask 1 on real tokens.

        Note: OpenCompass dataset configs already render the full prompt
        (few-shot + question), so we feed the raw string in as the user
        message — no Trace-side system-prompt detection.
        """
        if self.max_prompt_len is not None:
            max_prompt_len = int(self.max_prompt_len)
        else:
            # Match Trace: prompt cap == max_prompt_len (mapped here to
            # max_seq_len), decoupled from the diffusion canvas. Same convention
            # as Dream/Qwen/Llama3 wrappers — Trace sets max_prompt_len=1024
            # with max_ans_len=512 fully decoupled.
            max_prompt_len = int(self.max_seq_len)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                'Tokenizer has no pad_token_id and no eos_token_id; '
                'cannot left-pad LLaDA inputs.'
            )

        ids_list: List[torch.Tensor] = []
        for raw in inputs:
            text = raw if isinstance(raw, str) else str(raw)
            msgs = [{'role': 'user', 'content': text.strip()}]

            prompt_ids = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
            )

            if len(prompt_ids) > max_prompt_len:
                # Trace: keep tail (assistant header preserved).
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

        device = self.model.device
        return padded_ids.to(device), padded_attn.to(device)

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 block_length: Optional[int] = None,
                 **kwargs) -> List[str]:
        input_ids, attention_mask = self._build_inputs_trace_style(
            inputs, max_out_len)

        # Per-task canvas: steps == max_out_len. The eval scripts override
        # max_out_len per dataset so dream and llada share the same canvas.
        # Per-task block_length: when GenInferencer's `block_length` field is
        # set, it lands here as a kwarg and overrides the constructor default.
        bl = block_length if block_length is not None else self.block_length
        generate_ids = llada_generate(
            self.model,
            self.tokenizer,
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_out_len,
            steps=max_out_len,
            temperature=self.temperature,
            block_length=bl,
            remasking=self.remasking,
        )

        prompt_len = input_ids.shape[1]
        results = []
        for i in range(len(inputs)):
            gen_ids = generate_ids[i, prompt_len:]
            txt = self.tokenizer.decode(
                gen_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            stop_list = ['<|eot_id|>']
            if self.tokenizer.eos_token:
                stop_list.append(self.tokenizer.eos_token)
            for st in stop_list:
                if st and st in txt:
                    txt = txt.split(st)[0]
            results.append(txt.strip())
        return results

    def get_ppl(self, inputs, mask_length=None):
        raise NotImplementedError(
            'LLaDA is a diffusion model; PPL-based evaluation is not '
            'applicable. Use gen-based evaluation instead.')

    def get_ppl_tokenwise(self, inputs, mask_length=None):
        raise NotImplementedError(
            'LLaDA is a diffusion model; PPL-based evaluation is not '
            'applicable. Use gen-based evaluation instead.')

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
