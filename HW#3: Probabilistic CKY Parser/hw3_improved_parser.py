#!/usr/bin/env python3
"""
hw3_improved_parser.py PCKY with better out-of-vocabulary (OOV) handling.
Usage: hw3_improved_parser.py <input_PCFG_file> <test_sentence_filename> <output_parse_filename>
The improvement is the coverage since the baseline cannot parse any sentence containing a word not seen in training. 
Here the lexical lookup is changed (The CKY itself is untouched):
1. exact match, as in the baseline;
2. otherwise a case-insensitive match ('Arriving' vs 'arriving'), using the best probability seen for that word under any casing;
3. otherwise the word is truly OOV: propose every open-class tag with the same probability (1.0), so the grammar decides which tag 
fits. 
"Open class" is estimated from the PCFG alone: a pre-terminal is open-class if it rewrites to >= MIN_DISTINCT distinct words.
Nothing here looks at the test data.
"""
import sys
from collections import defaultdict
from hw3_pcky import PCFG, parse_file
MIN_DISTINCT = 10 #distinct word types a tag needs to count as open-class

def make_lex_fn(grammar, min_distinct=MIN_DISTINCT):
    #case-insensitive index: lowercase word -> {tag: best logprob}
    folded = defaultdict(dict)
    types_per_tag = defaultdict(int)
    for word, tags in grammar.lexicon.items():
        for tag, lp in tags.items():
            types_per_tag[tag] += 1
            old = folded[word.lower()].get(tag)
            if old is None or lp > old:
                  
                  folded[word.lower()][tag] = lp
    open_tags = {t: 0.0 for t, c in types_per_tag.items() if c >= min_distinct}
    if not open_tags: #tiny grammar: fall back to all tags
        open_tags = {t: 0.0 for t in types_per_tag}

    def lex(word):
        if word in grammar.lexicon:
            return grammar.lexicon[word]
        low = word.lower()
        if low in folded:
            return folded[low]
        return open_tags
    return lex

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: hw3_improved_parser.py <input_PCFG_file> "
                 "<test_sentence_filename> <output_parse_filename>")
    grammar = PCFG(sys.argv[1])
    n_ok, n = parse_file(grammar, make_lex_fn(grammar), sys.argv[2], sys.argv[3])
    sys.stderr.write("Improved: parsed %d / %d sentences\n" % (n_ok, n))

if __name__ == "__main__":
    main()