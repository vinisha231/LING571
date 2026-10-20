#!/usr/bin/env python3
"""hw3_parser.py baseline PCKY parser (OOV words fail to parse).
Usage: hw3_parser.py <input_PCFG_file> <test_sentence_filename> <output_parse_filename>"""
import sys
from hw3_pcky import PCFG, exact_lex_fn, parse_file

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: hw3_parser.py <input_PCFG_file> <test_sentence_filename> "
                 "<output_parse_filename>")
    grammar = PCFG(sys.argv[1])
    n_ok, n = parse_file(grammar, exact_lex_fn(grammar), sys.argv[2], sys.argv[3])
    sys.stderr.write("Baseline: parsed %d / %d sentences\n" % (n_ok, n))

if __name__ == "__main__":
    main()