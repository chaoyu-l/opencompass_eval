import math
from typing import Dict, List, Optional

import torch

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS


@MODELS.register_module()
class DREAMModel(BaseModel):
    """OpenCompass wrapper for the DREAM diffusion language model.

    DREAM's iterative demasking logic is shipped inside the model checkpoint
    (``model.diffusion_generate``), so this wrapper simply delegates to it.

    Generation policy (mirrors Trace_Results inference):
      * Prompt is wrapped via tokenizer.apply_chat_template(..., add_generation_prompt=True)
        before tokenize/pad — Dream is chat-trained, raw text is OOD.
      * Sampling alg = 'maskgit_plus' (top-1 confidence-based; Trace's
        mechanism-aligned choice, not Dream's library default).
      * steps == max_out_len: one diffusion step per generated token on average.
        Per-task canvas (and therefore steps) is set in the eval script via
        infer_cfg.inferencer.max_out_len.

    Args:
        path (str): Path to DREAM model checkpoint.
        tokenizer_path (str): Path to tokenizer (defaults to ``path``).
        max_seq_len (int): Maximum input sequence length.
        temperature (float): Sampling temperature (0 = greedy).
        model_kwargs (dict): Extra kwargs for ``from_pretrained``.
        tokenizer_kwargs (dict): Extra kwargs for tokenizer loading.
        peft_path (str): Path to PEFT/LoRA adapter (merged at load time).
        peft_merge (bool): Whether to merge_and_unload LoRA weights.
        batch_size (int): Batch size (consumed by OpenCompass runner).
    """

    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 2048,
        temperature: float = 0.0,
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
        self.temperature = temperature

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

    def _load_model(self, path: str, model_kwargs: dict,
                    peft_path: Optional[str], peft_merge: bool):
        from transformers import AutoConfig, AutoModel

        model_kwargs = model_kwargs.copy()
        model_kwargs.setdefault('trust_remote_code', True)
        model_kwargs.setdefault('torch_dtype', torch.bfloat16)

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        try:
            self.model = AutoModel.from_pretrained(
                path, config=config,
                attn_implementation='flash_attention_2', **model_kwargs)
        except (ValueError, ImportError, TypeError):
            self.model = AutoModel.from_pretrained(
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

        self.model.eval()

        # Dream supports torch.compile (LLaDA does not — its custom forward is
        # incompatible with CUDA graphs, see Trace_Results infer_single.py:288).
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
            except Exception:
                pass

    @staticmethod
    def _split_system_user(prompt_text: str):
        # Trace's helper checks for a project-specific DEFAULT_SYSTEM_PROMPT
        # prefix to split (system, user). OpenCompass prompts don't carry that
        # prefix, so the split degenerates to (None, prompt.strip()).
        return None, (prompt_text or '').strip()

    @staticmethod
    def _strip_trailing_assistant(text: str) -> str:
        # Some prompt templates leave an "Assistant:" suffix; drop it before
        # apply_chat_template so the assistant header isn't duplicated.
        t = (text or '').rstrip()
        for suf in ['Assistant:', 'assistant:', 'ASSISTANT:']:
            if t.endswith(suf):
                return t[:-len(suf)].rstrip()
        return t

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        # Mirror Trace_Results data_collator.py:493-541 — apply Dream chat
        # template with generation prompt, left-pad to batch max. This keeps
        # the input distribution aligned with how Dream was instruction-tuned.
        pad_id = (self.tokenizer.pad_token_id
                  if self.tokenizer.pad_token_id is not None
                  else self.tokenizer.eos_token_id)

        ids_list = []
        for p in inputs:
            sys_txt, user_txt = self._split_system_user(p)
            user_txt = self._strip_trailing_assistant(user_txt)
            messages = []
            if sys_txt:
                messages.append({'role': 'system', 'content': sys_txt})
            messages.append({'role': 'user', 'content': user_txt})

            prompt_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
            # Left-truncate to keep the assistant header at the tail.
            if (self.max_seq_len is not None
                    and len(prompt_ids) > self.max_seq_len):
                prompt_ids = prompt_ids[-self.max_seq_len:]
            ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))

        max_len = max(t.numel() for t in ids_list)
        B = len(ids_list)
        padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        padded_attn = torch.zeros((B, max_len), dtype=torch.long)
        for i, ids_t in enumerate(ids_list):
            l = ids_t.numel()
            padded_ids[i, max_len - l:] = ids_t
            padded_attn[i, max_len - l:] = 1

        input_ids = padded_ids.to(self.model.device)
        attention_mask = padded_attn.to(self.model.device)

        # Per-task canvas: steps == max_out_len. The eval scripts override
        # max_out_len per dataset (see eval_dream_7b_general.py).
        out = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_out_len,
            steps=max_out_len,
            temperature=self.temperature,
            alg='maskgit_plus',
            return_dict_in_generate=True,
            output_history=False,
        )
        generate_ids = out.sequences if hasattr(out, 'sequences') else out

        prompt_len = input_ids.shape[1]
        results = []
        for i in range(len(inputs)):
            gen_ids = generate_ids[i, prompt_len:]
            txt = self.tokenizer.decode(
                gen_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            stop_list = []
            if self.tokenizer.eos_token:
                stop_list.append(self.tokenizer.eos_token)
            stop_list += ['<|im_end|>', '<|endoftext|>']
            for st in stop_list:
                if st and st in txt:
                    txt = txt.split(st)[0]
            results.append(txt.strip())
        return results

    def get_ppl(self, inputs, mask_length=None):
        raise NotImplementedError(
            'DREAM is a diffusion model; PPL-based evaluation is not '
            'applicable. Use gen-based evaluation instead.')

    def get_ppl_tokenwise(self, inputs, mask_length=None):
        raise NotImplementedError(
            'DREAM is a diffusion model; PPL-based evaluation is not '
            'applicable. Use gen-based evaluation instead.')

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
