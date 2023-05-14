from dataclasses import dataclass
from functools import partial
from pathlib import Path
from ctypes import (
    CDLL,
    c_bool,
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    POINTER,
)
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)

from .lib import find_library

c_int_p = POINTER(c_int)
llm_p = c_void_p


@dataclass
class Config:
    # sample
    top_k: int = 40
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    last_n_tokens: int = 64
    seed: int = -1

    # eval
    batch_size: int = 8
    threads: int = -1

    # generate
    max_new_tokens: int = 256
    reset: bool = True


def get(*values):
    for value in values:
        if value is not None:
            return value


def load_library(path: Optional[str] = None) -> Any:
    path = find_library(path)
    lib = CDLL(path)

    lib.ctransformers_llm_create.argtypes = [c_char_p, c_char_p]
    lib.ctransformers_llm_create.restype = llm_p

    lib.ctransformers_llm_delete.argtypes = [llm_p]
    lib.ctransformers_llm_delete.restype = None

    lib.ctransformers_llm_tokenize.argtypes = [llm_p, c_char_p, c_int_p]
    lib.ctransformers_llm_tokenize.restype = c_int

    lib.ctransformers_llm_detokenize.argtypes = [llm_p, c_int]
    lib.ctransformers_llm_detokenize.restype = c_char_p

    lib.ctransformers_llm_is_eos_token.argtypes = [llm_p, c_int]
    lib.ctransformers_llm_is_eos_token.restype = c_bool

    lib.ctransformers_llm_batch_eval.argtypes = [llm_p, c_int_p, c_int, c_int]
    lib.ctransformers_llm_batch_eval.restype = c_bool

    lib.ctransformers_llm_sample.argtypes = [
        llm_p,
        c_int,  # top_k
        c_float,  # top_p
        c_float,  # temperature
        c_float,  # repetition_penalty
        c_int,  # last_n_tokens
        c_int,  # seed
    ]
    lib.ctransformers_llm_sample.restype = c_int

    lib.ctransformers_llm_reset.argtypes = [llm_p]
    lib.ctransformers_llm_reset.restype = None

    return lib


class LLM:

    def __init__(
        self,
        model_path: str,
        model_type: str,
        *,
        config: Optional[Config] = None,
        lib: Optional[str] = None,
    ):
        self._model_path = model_path
        self._model_type = model_type
        self._config = config or Config()
        self._llm = None
        self._lib = None

        if not Path(model_path).is_file():
            raise ValueError(f"Model path '{model_path}' doesn't exist.")

        self._lib = load_library(lib)
        self._llm = self._lib.ctransformers_llm_create(model_path.encode(),
                                                       model_type.encode())
        if self._llm is None:
            raise RuntimeError(
                f"Failed to create LLM '{model_type}' from '{model_path}'.")

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def config(self) -> Config:
        return self._config

    def __getattr__(self, name: str) -> Callable:
        lib, llm = self._lib, self._llm
        if name.startswith('ctransformers_llm_') and hasattr(lib, name):
            return partial(getattr(lib, name), llm)
        raise AttributeError(f"'LLM' object has no attribute '{name}'")

    def tokenize(self, text: str) -> List[int]:
        tokens = (c_int * len(text))()
        n_tokens = self.ctransformers_llm_tokenize(text.encode(), tokens)
        return tokens[:n_tokens]

    def detokenize(self, tokens: Union[Sequence[int], int]) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        texts = []
        for token in tokens:
            text = self.ctransformers_llm_detokenize(token)
            texts.append(text.decode())
        return ''.join(texts)

    def is_eos_token(self, token: int) -> bool:
        return self.ctransformers_llm_is_eos_token(token)

    def eval(
        self,
        tokens: Sequence[int],
        *,
        batch_size: Optional[int] = None,
        threads: Optional[int] = None,
    ) -> None:
        config = self.config
        batch_size = get(batch_size, config.batch_size)
        threads = get(threads, config.threads)

        n_tokens = len(tokens)
        tokens = (c_int * n_tokens)(*tokens)
        status = self.ctransformers_llm_batch_eval(tokens, n_tokens,
                                                   batch_size, threads)
        if not status:
            raise RuntimeError('Failed to evaluate tokens.')

    def sample(
        self,
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        last_n_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        config = self.config
        top_k = get(top_k, config.top_k)
        top_p = get(top_p, config.top_p)
        temperature = get(temperature, config.temperature)
        repetition_penalty = get(repetition_penalty, config.repetition_penalty)
        last_n_tokens = get(last_n_tokens, config.last_n_tokens)
        seed = get(seed, config.seed)

        return self.ctransformers_llm_sample(
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            last_n_tokens,
            seed,
        )

    def reset(self) -> None:
        self.ctransformers_llm_reset()

    def __del__(self):
        if self._llm is not None:
            self.ctransformers_llm_delete()

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
        config = self.config
        reset = get(reset, config.reset)

        if reset:
            self.reset()

        self.eval(tokens, batch_size=batch_size, threads=threads)
        while True:
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                last_n_tokens=last_n_tokens,
                seed=seed,
            )
            self.eval([token], batch_size=batch_size, threads=threads)
            if self.is_eos_token(token):
                break
            yield token

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
        reset: Optional[bool] = None,
    ) -> str:
        config = self.config
        max_new_tokens = get(max_new_tokens, config.max_new_tokens)

        tokens = self.tokenize(prompt)

        count = 0
        response = []
        for token in self.generate(
                tokens,
                batch_size=batch_size,
                threads=threads,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                last_n_tokens=last_n_tokens,
                seed=seed,
                reset=reset,
        ):
            response.append(token)
            count += 1
            if count >= max_new_tokens:
                break

        return self.detokenize(response)
