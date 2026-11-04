#!/bin/bash
D=/dropbox/26-27/571F/hw5
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/mnt/dropbox/26-27/571F/envs/bin/python
[ -x "$PY" ] || PY=python3
printf "%-14s %-10s %-10s %-10s\n" "config" "micro" "macro" "weighted"
for group in none feats lemmas forms pairs dep_labels lookahead structure; do
    if [ "$group" = "none" ]; then ablate=""; else ablate="$group"; fi
    out=$(HW5_ABLATE="$ablate" "$PY" "$DIR/hw5_predict.py" $D/es_train.conllu $D/es_test.conllu \
          $D/es_test.seq.gold /tmp/ablate_$$_train.seq /tmp/ablate_$$_pred.seq 2>/dev/null)
    micro=$(echo "$out" | awk '/Micro/{print $3}')
    macro=$(echo "$out" | awk '/Macro/{print $3}')
    weighted=$(echo "$out" | awk '/Weighted/{print $3}')
    printf "%-14s %-10s %-10s %-10s\n" "${ablate:-full}" "$micro" "$macro" "$weighted"
done
rm -f /tmp/ablate_$$_*