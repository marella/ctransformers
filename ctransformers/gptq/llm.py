try:
    import exllama
except ImportError:
    raise ImportError(
        "Could not import `exllama` package. "
        "Please install it using `pip install ctransformers[gptq]`"
    )

import re
from pathlib import Path
from typing import (
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)

import torch
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

from ..llm import Config, doc, get


class LLM:
    def __init__(
        self,
        model_path: str,
        *,
        config: Optional[Config] = None,
    ):
        """Loads the language model from a local file.

        Args:
            model_path: The path to a model directory.
            config: `Config` object.
        """
        model_path = Path(model_path).resolve()
        config = config or Config()
        self._model_path = model_path
        self._config = config

        files = [
            (f.stat().st_size, f)
            for f in model_path.iterdir()
            if f.is_file() and f.name.endswith(".safetensors")
        ]
        if not files:
            raise ValueError(f"No model file found in directory '{model_path}'")
        model_file = min(files)[1]

        model_config = ExLlamaConfig(str(model_path / "config.json"))
        model_config.model_path = str(model_file)

        model = ExLlama(model_config)
        tokenizer = ExLlamaTokenizer(str(model_path / "tokenizer.model"))
        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        self._model = model
        self._tokenizer = tokenizer
        self._generator = generator

    @property
    def model_path(self) -> str:
        """The path to the model directory."""
        return self._model_path

    @property
    def config(self) -> Config:
        """The config object."""
        return self._config

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token."""
        return self._tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        """The number of tokens in vocabulary."""
        return self._model.config.vocab_size

    @property
    def context_length(self) -> int:
        """The context length of model."""
        return self._model.config.max_seq_len

    def tokenize(self, text: str) -> List[int]:
        """Converts a text into list of tokens.

        Args:
            text: The text to tokenize.

        Returns:
            The list of tokens.
        """
        return self._tokenizer.encode(text)

    def detokenize(
        self,
        tokens: Sequence[int],
        decode: bool = True,
    ) -> Union[str, bytes]:
        """Converts a list of tokens to text.

        Args:
            tokens: The list of tokens.
            decode: Whether to decode the text as UTF-8 string.

        Returns:
            The combined text of all tokens.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)
        return self._tokenizer.decode(tokens)

    def is_eos_token(self, token: int) -> bool:
        """Checks if a token is an end-of-sequence token.

        Args:
            token: The token to check.

        Returns:
            `True` if the token is an end-of-sequence token else `False`.
        """
        return token == self.eos_token_id

    def reset(self) -> None:
        self._generator.reset()

    @doc
    def generate(
        self,
        tokens: Sequence[int],
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        last_n_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        threads: Optional[int] = None,
        reset: Optional[bool] = None,
    ) -> Generator[int, None, None]:
        """Generates new tokens from a list of tokens.

        Args:
            tokens: The list of tokens to generate tokens from.
            {params}

        Returns:
            The generated tokens.
        """
        generator = self._generator
        config = self.config
        top_k = get(top_k, config.top_k)
        top_p = get(top_p, config.top_p)
        temperature = get(temperature, config.temperature)
        repetition_penalty = get(repetition_penalty, config.repetition_penalty)
        last_n_tokens = get(last_n_tokens, config.last_n_tokens)
        reset = get(reset, config.reset)

        if reset:
            self.reset()
        generator.settings.top_k = top_k
        generator.settings.top_p = top_p
        generator.settings.temperature = temperature
        generator.settings.token_repetition_penalty_max = repetition_penalty
        generator.settings.token_repetition_penalty_sustain = last_n_tokens
        generator.settings.token_repetition_penalty_decay = last_n_tokens // 2

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).unsqueeze(0)
        assert tokens.shape[0] == 1
        generator.gen_begin(tokens)
        while True:
            token = generator.gen_single_token()
            token = token[0][0].item()
            if self.is_eos_token(token):
                break
            yield token

    def _stream(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        last_n_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        threads: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        reset: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        generator = self._generator
        config = self.config
        max_new_tokens = get(max_new_tokens, config.max_new_tokens)
        stop = get(stop, config.stop) or []
        if isinstance(stop, str):
            stop = [stop]

        tokens = self.tokenize(prompt)
        max_new_tokens = min(max_new_tokens, self.context_length - tokens.shape[1])

        stop_regex = re.compile("|".join(map(re.escape, stop)))
        count = 0
        length = len(self.detokenize(tokens[0]))
        text = ""
        for token in self.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            last_n_tokens=last_n_tokens,
            seed=seed,
            batch_size=batch_size,
            threads=threads,
            reset=reset,
        ):
            new_text = self.detokenize(generator.sequence_actual[0])[length:]
            length += len(new_text)
            text += new_text

            # https://github.com/abetlen/llama-cpp-python/blob/1a13d76c487df1c8560132d10bda62d6e2f4fa93/llama_cpp/llama.py#L686-L706
            # Check if one of the stop sequences is part of the text.
            # Note that the stop sequence may not always be at the end of text.
            if stop:
                match = stop_regex.search(text)
                if match:
                    text = text[: match.start()]
                    break

            # Avoid sending the longest suffix of text which is also a prefix
            # of a stop sequence, as it can form a stop sequence with the text
            # generated later.
            longest = 0
            for s in stop:
                for i in range(len(s), 0, -1):
                    if text.endswith(s[:i]):
                        longest = max(i, longest)
                        break

            end = len(text) - longest
            if end > 0:
                yield text[:end]
                text = text[end:]

            count += 1
            if count >= max_new_tokens:
                break

        if text:
            yield text

    @doc
    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        last_n_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        threads: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Optional[bool] = None,
        reset: Optional[bool] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Generates text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            {params}

        Returns:
            The generated text.
        """
        config = self.config
        stream = get(stream, config.stream)

        text = self._stream(
            prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            last_n_tokens=last_n_tokens,
            seed=seed,
            batch_size=batch_size,
            threads=threads,
            stop=stop,
            reset=reset,
        )
        if stream:
            return text
        return "".join(text)
