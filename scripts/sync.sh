#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

if [ -n "$(git status models/ggml --porcelain)" ]; then
  echo "'models/ggml' directory has uncommitted changes."
  exit 1
fi

git submodule update --init --recursive

cp models/submodules/llama.cpp/ggml-cuda.cu models/ggml/
cp models/submodules/llama.cpp/ggml-cuda.h models/ggml/

cp models/submodules/llama.cpp/ggml-metal.h models/ggml/
cp models/submodules/llama.cpp/ggml-metal.m models/ggml/
cp models/submodules/llama.cpp/ggml-metal.metal models/ggml/

cp models/submodules/llama.cpp/ggml-alloc.c models/ggml/
cp models/submodules/llama.cpp/ggml-alloc.h models/ggml/

cp models/submodules/llama.cpp/ggml.c models/ggml/
cp models/submodules/llama.cpp/ggml.h models/ggml/

cp models/submodules/llama.cpp/k_quants.c models/ggml/
cp models/submodules/llama.cpp/k_quants.h models/ggml/

cp models/submodules/llama.cpp/llama.cpp models/ggml/
cp models/submodules/llama.cpp/llama.h models/ggml/

cp models/submodules/ggllm.cpp/libfalcon.cpp models/ggml/
cp models/submodules/ggllm.cpp/libfalcon.h models/ggml/

cp models/submodules/ggllm.cpp/cmpnct_unicode.cpp models/ggml/
cp models/submodules/ggllm.cpp/cmpnct_unicode.h models/ggml/
