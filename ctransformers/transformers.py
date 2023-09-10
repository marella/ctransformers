try:
    import torch
except ImportError:
    raise ImportError(
        "Could not import `torch` package. "
        "Please install it using: pip install transformers[torch]"
    )

try:
    import transformers
except ImportError:
    raise ImportError(
        "Could not import `transformers` package. "
        "Please install it using: pip install transformers"
    )

from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TensorType,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)
from transformers.modeling_outputs import CausalLMOutput

from .llm import LLM


class CTransformersConfig(PretrainedConfig):
    pass


class CTransformersModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, llm: LLM):
        for name in [
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ]:
            if getattr(config, name, None) is None:
                value = getattr(llm, name, None)
                setattr(config, name, value)
        super().__init__(config)
        self._llm = llm
        MODEL_FOR_CAUSAL_LM_MAPPING.register("ctransformers", CTransformersModel)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        llm = self._llm
        logits = []
        for tokens in input_ids:
            tokens = tokens.tolist()
            tokens = llm.prepare_inputs_for_generation(tokens)
            llm.eval(tokens)
            logits.append(torch.tensor(llm.logits).reshape([1, -1]))
        logits = torch.stack(logits)
        if not return_dict:
            return (logits,)
        return CausalLMOutput(logits=logits)


class CTransformersTokenizer(PreTrainedTokenizer):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def vocab_size(self) -> int:
        return self._llm.vocab_size

    @property
    def bos_token_id(self) -> int:
        return self._llm.bos_token_id

    @property
    def bos_token(self) -> str:
        return self._llm.detokenize(self._llm.bos_token_id) or "<s>"

    @property
    def eos_token_id(self) -> int:
        return self._llm.eos_token_id

    @property
    def eos_token(self) -> str:
        return self._llm.detokenize(self._llm.eos_token_id) or "</s>"

    @property
    def pad_token_id(self) -> int:
        return self._llm.pad_token_id

    @property
    def pad_token(self) -> str:
        return self._llm.detokenize(self._llm.pad_token_id) or "</s>"

    @property
    def all_special_ids(self) -> List[int]:
        return [self.eos_token_id]

    def _encode_plus(
        self,
        text: Union[str, List[int]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(text, str):
            input_ids = self._llm.tokenize(text)
        elif (
            isinstance(text, (list, tuple))
            and len(text) > 0
            and isinstance(text[0], int)
        ):
            input_ids = text
        else:
            raise ValueError(
                f"Input {text} is not valid. Should be a string or a list/tuple of integers."
            )
        return self.prepare_for_model(
            input_ids,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
        )

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in self.all_special_ids]
        return self._llm.detokenize(token_ids)

    def _convert_token_to_id(self, token: str) -> int:
        return self._llm.tokenize(token, add_bos_token=False)[0]

    def _convert_id_to_token(self, index: int) -> str:
        return self._llm.detokenize(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)
