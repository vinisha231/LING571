#!/bin/bash 
# usage: hw4_oracle.sh <input_dependency_parse_filename> <output_dependency_filename> <output_sequence_filename>
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/mnt/dropbox/26-27/571F/envs/bin/python
[ -x "$PY" ] || PY=python3 #fallback
"$PY" "$DIR/hw4_oracle.py" "$@"