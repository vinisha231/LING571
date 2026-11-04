#!/bin/bash
# usage: hw5_predict.sh <train_data> <test_data> <test_labels> <train_seq> <pred_seq>
if [ "$#" -ne 5 ]; then
    echo "usage: $0 <train_data> <test_data> <test_labels> <train_seq> <pred_seq>" >&2
    exit 1
fi
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/mnt/dropbox/26-27/571F/envs/bin/python
[ -x "$PY" ] || PY=python3 #fallback
"$PY" "$DIR/hw5_predict.py" "$@"