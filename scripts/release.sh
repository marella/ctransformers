#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

trap clean EXIT
clean() {
    rm -r build dist *.egg-info || true
}

clean
python3 setup.py sdist bdist_wheel
twine upload dist/*
