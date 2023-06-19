try:
    import transformers
except ImportError:
    raise ImportError(
        "Could not import `transformers` package. "
        "Please install it with `pip install transformers`"
    )

try:
    import torch
except ImportError:
    raise ImportError(
        "Could not import `torch` package. "
        "Please install it from https://pytorch.org/get-started/locally/"
    )

import copy
from typing import List, Optional, Union

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer

from .llm import get, LLM


class Model:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.config = llm.config
        self.config.vocab_size = llm.vocab_size
        self.config.pad_token_id = None
        self.device = None
        self._past = []

    def prepare_inputs_for_generation(self):
        raise NotImplementedError()

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        streamer: Optional[BaseStreamer] = None,
        **kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        llm, config = self._llm, self.config
        reset = config.reset

        assert "input_ids" not in kwargs, "TODO"
        assert inputs.shape[0] == 1, "Batch size must be 1."

        if kwargs.get("temperature") == 0.0:
            kwargs["temperature"] = 0.5
        for k in [
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "max_new_tokens",
        ]:
            kwargs[k] = kwargs.get(k, getattr(config, k))
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        else:
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)

        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        output_scores = (
            output_scores
            if output_scores is not None
            else generation_config.output_scores
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else generation_config.return_dict_in_generate
        )

        scores = [] if return_dict_in_generate and output_scores else None
        tokens = inputs.flatten().tolist()
        n_past = len(self._past)
        if n_past > 0 and tokens[:n_past] == self._past:
            tokens = tokens[n_past:]
            reset = False

        if reset:
            llm.reset()

        if tokens:
            llm.eval(tokens)
        count = 0
        while count < generation_config.max_new_tokens:
            logits = llm.logits
            logits_tensor = torch.tensor(logits, dtype=torch.float).unsqueeze(0)
            logits_tensor = logits_processor(input_ids=inputs, scores=logits_tensor)
            if return_dict_in_generate and output_scores:
                scores.append(logits_tensor)
            assert isinstance(
                logits_tensor[0][0].item(), float
            ), f"Expected 'float' but got '{type(logits_tensor[0][0].item())}'"
            for i in range(len(logits)):
                logits[i] = logits_tensor[0][i].item()

            token = llm.sample(
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                repetition_penalty=generation_config.repetition_penalty,
            )
            llm.eval([token])

            inputs = torch.concat([inputs, torch.tensor([token]).unsqueeze(0)], dim=-1)
            if stopping_criteria(inputs, scores):
                break

            if llm.is_eos_token(token):
                break

            count += 1

        self._past = inputs.flatten().tolist()

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(sequences=inputs, scores=scores)
        return inputs

    # Added for Microsoft Guidance
    # https://github.com/microsoft/guidance/blob/d6b855aa625677f806fc51ec7238d2a38df594ea/guidance/llms/_transformers.py#L158
    def prepare_inputs_for_generation(self):
        raise NotImplementedError()

    # Added for Microsoft Guidance
    # https://github.com/microsoft/guidance/blob/d6b855aa625677f806fc51ec7238d2a38df594ea/guidance/llms/_transformers.py#L171
    def _update_model_kwargs_for_generation(self):
        raise NotImplementedError()


class Tokenizer:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.vocab_size = llm.vocab_size
        self.eos_token_id = llm.eos_token_id
        self.eos_token = llm.detokenize(self.eos_token_id) or "</s>"  # TODO
        self.max_sequence_length = llm.context_length

    def encode(self, text: str) -> List[int]:
        return self._llm.tokenize(text)

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._llm.detokenize(token_ids)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.decode(ids)
        else:
            return [self.decode(id) for id in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        index = 1 if self._llm.model_type == "llama" else 0
        if tokens is None:
            return None
        elif isinstance(tokens, str):
            return self.encode(tokens)[index]
        else:
            return [self.encode(token)[index] for token in tokens]
