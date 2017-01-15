#!/bin/bash

# set -x

jupyter-nbconvert --version 2>/dev/null 1>&2 || (echo "jupyter-nbconvert not found" && exit 1)
autopep8 --version 2>/dev/null 1>&2 || (echo "autopep8 not found" && exit 1)

TEMPLATE=./odl_python.tpl
OUT_DIR=../code

for notebook in *.ipynb; do
    BASE=${notebook::-6}  # all except last 6 characters
    jupyter-nbconvert --to python --template="$TEMPLATE" --stdout "$notebook" | autopep8 -aa - > "$OUT_DIR/$BASE.py"
done
