#!/bin/bash
# hw3_run.sh -- run the whole HW #3 pipeline end to end.
#
# Usage: hw3_run.sh <treebank_filename> <output_PCFG_file> \
#                   <test_sentence_filename> <baseline_parse_output_filename> \
#                   <input_PCFG_file> \
#                   <improved_parse_output_filename> \
#                   <baseline_eval> <improved_eval>
#
# Notes:
#  * I improved the PARSER (OOV handling), not the induction, so
#    <input_PCFG_file> is accepted but ignored; the improved parser re-uses
#    <output_PCFG_file>, as the assignment allows.
#  * evalb needs the gold parses, which are not an argument; they default to
#    the course copy and can be overridden with GOLD=... (EVALB_DIR=... too).

if [ $# -ne 8 ]; then
    echo "Usage: $0 <treebank> <output_PCFG> <test_sentences> <baseline_parses_out> \\" >&2
    echo "          <input_PCFG> <improved_parses_out> <baseline_eval> <improved_eval>" >&2
    exit 1
fi

TREEBANK=$1; PCFG_OUT=$2; SENTS=$3; BASE_OUT=$4
PCFG_IN=$5; IMPR_OUT=$6; BASE_EVAL=$7; IMPR_EVAL=$8

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALB_DIR=${EVALB_DIR:-/dropbox/26-27/571F/hw3/tools}
GOLD=${GOLD:-/dropbox/26-27/571F/hw3/data/parses.gold}
PRM="$EVALB_DIR/COLLINS.prm"
[ -f "$PRM" ] || PRM="$EVALB_DIR/Collins.prm"   # slides and handout differ in case

echo "== 1. Grammar induction ==" >&2
"$DIR/hw3_topcfg.sh" "$TREEBANK" "$PCFG_OUT"

echo "== 2. Baseline PCKY parsing ==" >&2
time "$DIR/hw3_parser.sh" "$PCFG_OUT" "$SENTS" "$BASE_OUT"

echo "== 3. Evaluating baseline ==" >&2
"$EVALB_DIR/evalb" -p "$PRM" "$GOLD" "$BASE_OUT" > "$BASE_EVAL"

echo "== 4. Improved PCKY parsing ==" >&2
time "$DIR/hw3_improved_parser.sh" "$PCFG_OUT" "$SENTS" "$IMPR_OUT"

echo "== 5. Evaluating improved parser ==" >&2
"$EVALB_DIR/evalb" -p "$PRM" "$GOLD" "$IMPR_OUT" > "$IMPR_EVAL"