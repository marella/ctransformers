#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

cp models/llama.cpp/llama-util.h models/llms/llama/
cp models/llama.cpp/llama.cpp models/llms/llama/
cp models/llama.cpp/llama.h models/llms/llama/
