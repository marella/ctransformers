#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

cmake -S models -B build
cmake --build build --config Release
