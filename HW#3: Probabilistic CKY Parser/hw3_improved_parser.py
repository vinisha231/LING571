#!/usr/bin/env python3
"""
hw3_improved_parser.py PCKY with better out-of-vocabulary (OOV) handling.
Usage: hw3_improved_parser.py <input_PCFG_file> <test_sentence_filename> <output_parse_filename>
The improvement is the coverage. The CKY itself is untouched and only the lexical lookup lex(word) -> {tag: logprob} changes.

SUBMITTED METHOD (OOV_MODE=threshold, the default):
1. exact match, as in the baseline;
2. otherwise a case-insensitive match ('Arriving' vs 'arriving');
3. otherwise the word is truly OOV: propose every open-class tag (>= 10 
distinct word types in the PCFG) with the same probability (1.0).

EXPERIMENTAL (not used by default): in modes 'hapax' / 'hapax_suffix' step 3
is replaced by a graded score, estimated from the PCFG file alone:
P(w | t)  ~  P_unk(t) * P(suffix | t)
estimated from the PCFG file alone (the test data is never consulted):

P_unk(t)=(n1(t) + DELTA) / Count(t) This is the "hapax" idea from lecture:
tags that rewrite to many once-seen words are the ones that take new words.  The PCFG file has probabilities, not counts, so I 
recover them:
the smallest lexical probability of t is 1/Count(t), and a rule whose probability equals that minimum was seen exactly once 
(a hapax). Tags whose estimated count is below COUNT_FLOOR (e.g. single-word tags with probability 1.0) get n1=0 and 
Count=COUNT_FLOOR.
P(suf|t) =(K(suf,t) + BETA) / (n_types(t) + BETA * V)
K=number of known word types with the same 3-letter suffix and tag t,
V=number of distinct suffixes (add-BETA smoothing).  This lets
'-ing' words prefer VBG, '-s' words plural nouns, and so on.

OOV_MODE environment variable (default 'threshold'=the submitted method):
threshold which is every tag with >= 10 word types is proposed, equal prob.
hapax experimental: P_unk(t) only.
hapax_suffix experimental: P_unk(t) * P(suffix | t).
The two experimental modes did not beat 'threshold' on held-out training data or on the test set (see readme), so they are not 
used by default.
"""
import math
import os
import sys
from collections import defaultdict
from hw3_pcky import PCFG, parse_file

MIN_DISTINCT=10 #'threshold' mode: word types needed to count as open-class
DELTA=0.5 #smoothing for the hapax count
BETA=0.5 #smoothing for the suffix model
COUNT_FLOOR=10 #a tag needs >= this many (estimated) tokens to trust its hapax count
SUFFIX_LEN=3


def make_lex_fn(grammar, mode="threshold"):
    lexicon=grammar.lexicon
    folded=defaultdict(dict) #lowercase word -> {tag: best logprob}
    probs=defaultdict(list) #tag -> lexical rule probabilities
    types_per_tag=defaultdict(int)
    for word, tags in lexicon.items():
        for tag, lp in tags.items():
            types_per_tag[tag] += 1
            probs[tag].append(math.exp(lp))
            old=folded[word.lower()].get(tag)
            if old is None or lp > old:
                folded[word.lower()][tag]=lp

    #'threshold' mode (first version)
    thr_tags={t: 0.0 for t, c in types_per_tag.items() if c >= MIN_DISTINCT}
    if not thr_tags:
        thr_tags={t: 0.0 for t in types_per_tag}

    #hapax estimates: log P_unk(t) 
    log_unk={}
    for tag, ps in probs.items():
        pmin=min(ps)
        count=1.0 / pmin #~ Count(tag)
        n1=sum(1 for p in ps if p < 1.5 * pmin) #words seen once
        if count < COUNT_FLOOR:
   #too little evidence (e.g. 'TO -> to [1.0]' makes count look like 1):
   #do not trust n1, and do not let such a tag look like a hapax-rich one
            count, n1=COUNT_FLOOR, 0
        log_unk[tag]=math.log(min(1.0, (n1 + DELTA) / count))

    #suffix statistics 
    suf_count=defaultdict(lambda: defaultdict(int)) #suffix -> tag -> K
    all_suffixes=set()
    for word, tags in lexicon.items():
        s=word.lower()[-SUFFIX_LEN:]
        all_suffixes.add(s)
        for tag in tags:
            suf_count[s][tag] += 1
    V=max(1, len(all_suffixes))

    cache={}

    def oov_scores(word):
        if mode == "threshold":
            return thr_tags
        scores=dict(log_unk)
        if mode == "hapax_suffix":
            s=word.lower()[-SUFFIX_LEN:]
            kmap=suf_count.get(s, {})
            for tag in scores:
                k=kmap.get(tag, 0)
                scores[tag] += math.log((k + BETA) / (types_per_tag[tag] + BETA * V))
        return scores

    def lex(word):
        if word in lexicon:
            return lexicon[word]
        low=word.lower()
        if low in folded:
            return folded[low]
        if word not in cache:
            cache[word]=oov_scores(word)
        return cache[word]
    return lex

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: hw3_improved_parser.py <input_PCFG_file> "
                 "<test_sentence_filename> <output_parse_filename>")
    mode=os.environ.get("OOV_MODE", "threshold")
    if mode not in ("threshold", "hapax", "hapax_suffix"):
        sys.exit("OOV_MODE must be threshold, hapax or hapax_suffix")
    grammar=PCFG(sys.argv[1])
    n_ok, n=parse_file(grammar, make_lex_fn(grammar, mode), sys.argv[2], sys.argv[3])
    sys.stderr.write("Improved (%s): parsed %d / %d sentences\n" % (mode, n_ok, n))

if __name__ == "__main__":
    main()