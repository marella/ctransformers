#!/usr/bin/env sh

set -eu
cd "$(dirname "$0")"
cd ..

if [ -n "$(git status --porcelain)" ]; then
    echo "Project has uncommitted changes."
    exit 1
fi

# Prevent scikit-build from adding these to source distribution.
sdist_ignore="ctransformers/lib models/submodules"

trap onexit EXIT

clean() {
    rm -r build dist *.egg-info || true
    rm -r _skbuild MANIFEST.in || true
}

onexit() {
    clean
    git restore $sdist_ignore
}

clean
CT_WHEEL=1 python3 setup.py bdist_wheel

rm -r $sdist_ignore
python3 setup.py sdist

twine upload dist/*
