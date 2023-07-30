# [C Transformers](https://github.com/marella/ctransformers) [![PyPI](https://img.shields.io/pypi/v/ctransformers)](https://pypi.org/project/ctransformers/) [![tests](https://github.com/marella/ctransformers/actions/workflows/tests.yml/badge.svg)](https://github.com/marella/ctransformers/actions/workflows/tests.yml) [![build](https://github.com/marella/ctransformers/actions/workflows/build.yml/badge.svg)](https://github.com/marella/ctransformers/actions/workflows/build.yml)

Python bindings for the Transformer models implemented in C/C++ using [GGML](https://github.com/ggerganov/ggml) library.

> Also see [ChatDocs](https://github.com/marella/chatdocs)

- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
  - [Hugging Face Hub](#hugging-face-hub)
  - [LangChain](#langchain)
  - [GPU](#gpu)
- [Documentation](#documentation)
- [License](#license)

## Supported Models

| Models                | Model Type  |
| :-------------------- | ----------- |
| GPT-2                 | `gpt2`      |
| GPT-J, GPT4All-J      | `gptj`      |
| GPT-NeoX, StableLM    | `gpt_neox`  |
| LLaMA, LLaMA 2        | `llama`     |
| MPT                   | `mpt`       |
| Dolly V2              | `dolly-v2`  |
| Replit                | `replit`    |
| StarCoder, StarChat   | `starcoder` |
| Falcon (Experimental) | `falcon`    |

## Installation

```sh
pip install ctransformers
```

## Usage

It provides a unified interface for all models:

```py
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained('/path/to/ggml-gpt-2.bin', model_type='gpt2')

print(llm('AI is going to'))
```

[Run in Google Colab](https://colab.research.google.com/drive/1GMhYMUAv_TyZkpfvUI1NirM8-9mCXQyL)

> **Note:** In order to use LLaMA 2 70B models, the model path or repo name must contain the word `70B`. For example, `llama-2-70b.bin`, `llama-2-70b/ggml-model.bin`, `TheBloke/Llama-2-70B-GGML` etc.

It provides a generator interface for more control:

```py
tokens = llm.tokenize('AI is going to')

for token in llm.generate(tokens):
    print(llm.detokenize(token))
```

It can be used with a custom or Hugging Face tokenizer:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

tokens = tokenizer.encode('AI is going to')

for token in llm.generate(tokens):
    print(tokenizer.decode(token))
```

It also provides access to the low-level C API. See [Documentation](#documentation) section below.

### Hugging Face Hub

It can be used with models hosted on the Hub:

```py
llm = AutoModelForCausalLM.from_pretrained('marella/gpt-2-ggml')
```

If a model repo has multiple model files (`.bin` files), specify a model file using:

```py
llm = AutoModelForCausalLM.from_pretrained('marella/gpt-2-ggml', model_file='ggml-model.bin')
```

It can be used with your own models uploaded on the Hub. For better user experience, upload only one model per repo.

To use it with your own model, add `config.json` file to your model repo specifying the `model_type`:

```json
{
  "model_type": "gpt2"
}
```

You can also specify additional parameters under `task_specific_params.text-generation`.

See [marella/gpt-2-ggml](https://huggingface.co/marella/gpt-2-ggml/blob/main/config.json) for a minimal example and [marella/gpt-2-ggml-example](https://huggingface.co/marella/gpt-2-ggml-example/blob/main/config.json) for a full example.

### LangChain

It is integrated into LangChain. See [LangChain docs](https://python.langchain.com/docs/ecosystem/integrations/ctransformers).

### GPU

> **Note:** Currently only LLaMA and Falcon models have GPU support.

To run some of the model layers on GPU, set the `gpu_layers` parameter:

```py
llm = AutoModelForCausalLM.from_pretrained('/path/to/ggml-llama.bin', model_type='llama', gpu_layers=50)
```

#### CUDA

Make sure you have installed [CUDA 12](https://developer.nvidia.com/cuda-downloads) and latest [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx).

<details>
<summary><strong>Show instructions for CUDA 11</strong></summary><br>

To use with CUDA 11, install the `ctransformers` package using:

```sh
CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers
```

On Windows PowerShell run:

```sh
$env:CT_CUBLAS=1
pip install ctransformers --no-binary ctransformers
```

On Windows Command Prompt run:

```sh
set CT_CUBLAS=1
pip install ctransformers --no-binary ctransformers
```

</details>

[Run in Google Colab](https://colab.research.google.com/drive/1Ihn7iPCYiqlTotpkqa1tOhUIpJBrJ1Tp)

#### Metal

To enable Metal support, install the `ctransformers` package using:

```sh
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

## Documentation

<!-- API_DOCS -->

### Config

| Parameter            | Type        | Description                                              | Default |
| :------------------- | :---------- | :------------------------------------------------------- | :------ |
| `top_k`              | `int`       | The top-k value to use for sampling.                     | `40`    |
| `top_p`              | `float`     | The top-p value to use for sampling.                     | `0.95`  |
| `temperature`        | `float`     | The temperature to use for sampling.                     | `0.8`   |
| `repetition_penalty` | `float`     | The repetition penalty to use for sampling.              | `1.1`   |
| `last_n_tokens`      | `int`       | The number of last tokens to use for repetition penalty. | `64`    |
| `seed`               | `int`       | The seed value to use for sampling tokens.               | `-1`    |
| `max_new_tokens`     | `int`       | The maximum number of new tokens to generate.            | `256`   |
| `stop`               | `List[str]` | A list of sequences to stop generation when encountered. | `None`  |
| `stream`             | `bool`      | Whether to stream the generated text.                    | `False` |
| `reset`              | `bool`      | Whether to reset the model state before generating text. | `True`  |
| `batch_size`         | `int`       | The batch size to use for evaluating tokens.             | `8`     |
| `threads`            | `int`       | The number of threads to use for evaluating tokens.      | `-1`    |
| `context_length`     | `int`       | The maximum context length to use.                       | `-1`    |
| `gpu_layers`         | `int`       | The number of layers to run on GPU.                      | `0`     |

> **Note:** Currently only LLaMA, MPT, Falcon models support the `context_length` parameter and only LLaMA, Falcon models support the `gpu_layers` parameter.

### <kbd>class</kbd> `AutoModelForCausalLM`

---

#### <kbd>classmethod</kbd> `AutoModelForCausalLM.from_pretrained`

```python
from_pretrained(
    model_path_or_repo_id: str,
    model_type: Optional[str] = None,
    model_file: Optional[str] = None,
    config: Optional[ctransformers.hub.AutoConfig] = None,
    lib: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs
) → LLM
```

Loads the language model from a local file or remote repo.

**Args:**

- <b>`model_path_or_repo_id`</b>: The path to a model file or directory or the name of a Hugging Face Hub model repo.
- <b>`model_type`</b>: The model type.
- <b>`model_file`</b>: The name of the model file in repo or directory.
- <b>`config`</b>: `AutoConfig` object.
- <b>`lib`</b>: The path to a shared library or one of `avx2`, `avx`, `basic`.
- <b>`local_files_only`</b>: Whether or not to only look at local files (i.e., do not try to download the model).

**Returns:**
`LLM` object.

### <kbd>class</kbd> `LLM`

### <kbd>method</kbd> `LLM.__init__`

```python
__init__(
    model_path: str,
    model_type: str,
    config: Optional[ctransformers.llm.Config] = None,
    lib: Optional[str] = None
)
```

Loads the language model from a local file.

**Args:**

- <b>`model_path`</b>: The path to a model file.
- <b>`model_type`</b>: The model type.
- <b>`config`</b>: `Config` object.
- <b>`lib`</b>: The path to a shared library or one of `avx2`, `avx`, `basic`.

---

##### <kbd>property</kbd> LLM.config

The config object.

---

##### <kbd>property</kbd> LLM.context_length

The context length of model.

---

##### <kbd>property</kbd> LLM.embeddings

The input embeddings.

---

##### <kbd>property</kbd> LLM.eos_token_id

The end-of-sequence token.

---

##### <kbd>property</kbd> LLM.logits

The unnormalized log probabilities.

---

##### <kbd>property</kbd> LLM.model_path

The path to the model file.

---

##### <kbd>property</kbd> LLM.model_type

The model type.

---

##### <kbd>property</kbd> LLM.vocab_size

The number of tokens in vocabulary.

---

#### <kbd>method</kbd> `LLM.detokenize`

```python
detokenize(tokens: Sequence[int], decode: bool = True) → Union[str, bytes]
```

Converts a list of tokens to text.

**Args:**

- <b>`tokens`</b>: The list of tokens.
- <b>`decode`</b>: Whether to decode the text as UTF-8 string.

**Returns:**
The combined text of all tokens.

---

#### <kbd>method</kbd> `LLM.embed`

```python
embed(
    input: Union[str, Sequence[int]],
    batch_size: Optional[int] = None,
    threads: Optional[int] = None
) → List[float]
```

Computes embeddings for a text or list of tokens.

> **Note:** Currently only LLaMA and Falcon models support embeddings.

**Args:**

- <b>`input`</b>: The input text or list of tokens to get embeddings for.
- <b>`batch_size`</b>: The batch size to use for evaluating tokens. Default: `8`
- <b>`threads`</b>: The number of threads to use for evaluating tokens. Default: `-1`

**Returns:**
The input embeddings.

---

#### <kbd>method</kbd> `LLM.eval`

```python
eval(
    tokens: Sequence[int],
    batch_size: Optional[int] = None,
    threads: Optional[int] = None
) → None
```

Evaluates a list of tokens.

**Args:**

- <b>`tokens`</b>: The list of tokens to evaluate.
- <b>`batch_size`</b>: The batch size to use for evaluating tokens. Default: `8`
- <b>`threads`</b>: The number of threads to use for evaluating tokens. Default: `-1`

---

#### <kbd>method</kbd> `LLM.generate`

```python
generate(
    tokens: Sequence[int],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    last_n_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    threads: Optional[int] = None,
    reset: Optional[bool] = None
) → Generator[int, NoneType, NoneType]
```

Generates new tokens from a list of tokens.

**Args:**

- <b>`tokens`</b>: The list of tokens to generate tokens from.
- <b>`top_k`</b>: The top-k value to use for sampling. Default: `40`
- <b>`top_p`</b>: The top-p value to use for sampling. Default: `0.95`
- <b>`temperature`</b>: The temperature to use for sampling. Default: `0.8`
- <b>`repetition_penalty`</b>: The repetition penalty to use for sampling. Default: `1.1`
- <b>`last_n_tokens`</b>: The number of last tokens to use for repetition penalty. Default: `64`
- <b>`seed`</b>: The seed value to use for sampling tokens. Default: `-1`
- <b>`batch_size`</b>: The batch size to use for evaluating tokens. Default: `8`
- <b>`threads`</b>: The number of threads to use for evaluating tokens. Default: `-1`
- <b>`reset`</b>: Whether to reset the model state before generating text. Default: `True`

**Returns:**
The generated tokens.

---

#### <kbd>method</kbd> `LLM.is_eos_token`

```python
is_eos_token(token: int) → bool
```

Checks if a token is an end-of-sequence token.

**Args:**

- <b>`token`</b>: The token to check.

**Returns:**
`True` if the token is an end-of-sequence token else `False`.

---

#### <kbd>method</kbd> `LLM.reset`

```python
reset() → None
```

Resets the model state.

---

#### <kbd>method</kbd> `LLM.sample`

```python
sample(
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    last_n_tokens: Optional[int] = None,
    seed: Optional[int] = None
) → int
```

Samples a token from the model.

**Args:**

- <b>`top_k`</b>: The top-k value to use for sampling. Default: `40`
- <b>`top_p`</b>: The top-p value to use for sampling. Default: `0.95`
- <b>`temperature`</b>: The temperature to use for sampling. Default: `0.8`
- <b>`repetition_penalty`</b>: The repetition penalty to use for sampling. Default: `1.1`
- <b>`last_n_tokens`</b>: The number of last tokens to use for repetition penalty. Default: `64`
- <b>`seed`</b>: The seed value to use for sampling tokens. Default: `-1`

**Returns:**
The sampled token.

---

#### <kbd>method</kbd> `LLM.tokenize`

```python
tokenize(text: str) → List[int]
```

Converts a text into list of tokens.

**Args:**

- <b>`text`</b>: The text to tokenize.

**Returns:**
The list of tokens.

---

#### <kbd>method</kbd> `LLM.__call__`

```python
__call__(
    prompt: str,
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
    reset: Optional[bool] = None
) → Union[str, Generator[str, NoneType, NoneType]]
```

Generates text from a prompt.

**Args:**

- <b>`prompt`</b>: The prompt to generate text from.
- <b>`max_new_tokens`</b>: The maximum number of new tokens to generate. Default: `256`
- <b>`top_k`</b>: The top-k value to use for sampling. Default: `40`
- <b>`top_p`</b>: The top-p value to use for sampling. Default: `0.95`
- <b>`temperature`</b>: The temperature to use for sampling. Default: `0.8`
- <b>`repetition_penalty`</b>: The repetition penalty to use for sampling. Default: `1.1`
- <b>`last_n_tokens`</b>: The number of last tokens to use for repetition penalty. Default: `64`
- <b>`seed`</b>: The seed value to use for sampling tokens. Default: `-1`
- <b>`batch_size`</b>: The batch size to use for evaluating tokens. Default: `8`
- <b>`threads`</b>: The number of threads to use for evaluating tokens. Default: `-1`
- <b>`stop`</b>: A list of sequences to stop generation when encountered. Default: `None`
- <b>`stream`</b>: Whether to stream the generated text. Default: `False`
- <b>`reset`</b>: Whether to reset the model state before generating text. Default: `True`

**Returns:**
The generated text.

<!-- API_DOCS -->

## License

[MIT](https://github.com/marella/ctransformers/blob/main/LICENSE)
