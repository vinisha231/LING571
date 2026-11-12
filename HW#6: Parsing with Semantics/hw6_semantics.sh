#!/bin/bash
# usage: hw6_semantics.sh <input_grammar_filename> <input_sentences_filename> <output_filename>
if [ "$#" -ne 3 ]; then
    echo "usage: $0 <input_grammar_filename> <input_sentences_filename> <output_filename>" >&2
    exit 1
fi
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/mnt/dropbox/26-27/571F/envs/bin/python
[ -x "$PY" ] || PY=python3 #fallback
"$PY" "$DIR/hw6_semantics.py" "$@"