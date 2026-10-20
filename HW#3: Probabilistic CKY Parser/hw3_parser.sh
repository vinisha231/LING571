#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/mnt/dropbox/26-27/571F/envs/bin/python
[ -x "$PY" ] || PY=python3 #fallback 
"$PY" "$DIR/hw3_parser.py" "$@"