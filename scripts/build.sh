#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

cmake -B build
cmake --build build --config Release
