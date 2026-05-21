import math
from typing import Optional

from mmengine.device import is_npu_available

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFacewithChatTemplate, _set_model_kwargs_torch_dtype)
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFacewithChatTemplateResized(HuggingFacewithChatTemplate):
    """HF chat-template wrapper that resizes token embeddings to
    ``ceil(len(tokenizer) / N) * N`` BEFORE attaching a PEFT adapter.

    This mirrors the training-time call
    ``model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=N)``
    used in Trace_Results/inference/infer_single.py:246. Without it, a LoRA
    saved with ``modules_to_save=['embed_tokens', 'lm_head']`` against a
    resized base will fail to load on a fresh base, e.g. for Qwen2.5-7B:
    LoRA shape is [151672, hidden] (=ceil(151665/8)*8) but the unresized
    base is [152064, hidden] (config-default padding), giving a
    ``size mismatch`` in ``PeftModel.from_pretrained``.

    Args:
        resize_to_multiple_of (int, optional): pad target. Set to 0/None
            to disable resizing. Defaults to 8 to match Trace.
    """

    def __init__(self, resize_to_multiple_of: Optional[int] = 8, **kwargs):
        self.resize_to_multiple_of = resize_to_multiple_of
        super().__init__(**kwargs)

    def _load_model(self, path: str, kwargs: dict,
                    peft_path: Optional[str] = None,
                    peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        model_kwargs = dict(device_map='auto', trust_remote_code=True)
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if self.resize_to_multiple_of:
            N = int(self.resize_to_multiple_of)
            target = int(N * math.ceil(len(self.tokenizer) / float(N)))
            cur = self.model.get_input_embeddings().weight.shape[0]
            if cur != target:
                self.logger.info(
                    f'resize_token_embeddings: {cur} -> {target} '
                    f'(len(tokenizer)={len(self.tokenizer)}, multiple_of={N})')
                self.model.resize_token_embeddings(target)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False
