#!/usr/bin/env python3
"""
hw3_topcfg.py PCFG induction from a treebank.
 
This program reads one CNF parse per line, counts productions, and estimates P(A -> beta) = Count(A -> beta) / Count(A)
(maximum likelihood, no smoothing; see readme).  Output is in NLTK PCFG format, 
e.g. NP -> Det Noun [0.30000]/Noun -> 'flight' [0.05000]
The first rule's LHS is the start symbol (the NLTK convention).
"""
import re
import sys
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

def read_tree(line):
    """Parse a bracketed tree into (label, [children]); a child is a
    (label, [...]) tuple for a non-terminal or a str for a word."""
    toks = TOKEN_RE.findall(line)
    pos = 0

    def node():
        nonlocal pos
        if toks[pos] != "(":
            raise ValueError("expected '('")
        pos += 1
        if toks[pos] == "(": #unlabeled root, e.g. "( (S ...) )"
            label = "TOP"
        else:
            label = toks[pos]
            pos += 1
        children = []
        while toks[pos] != ")":
            if toks[pos] == "(":
                children.append(node())
            else:
                children.append(toks[pos])
                pos += 1
        pos += 1
        return (label, children)
    tree = node()
    if pos != len(toks):
        raise ValueError("trailing tokens after tree")
    return tree

def quote(word):
    """NLTK-style terminal quoting."""
    return '"%s"' % word if "'" in word else "'%s'" % word

def count_rules(tree, rule_counts, lhs_counts):
    label, children = tree
    rhs = []
    for ch in children:
        if isinstance(ch, str):
            rhs.append(quote(ch))
        else:
            rhs.append(ch[0])
            count_rules(ch, rule_counts, lhs_counts)
    rule_counts[(label, tuple(rhs))] += 1
    lhs_counts[label] += 1

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: hw3_topcfg.py <treebank_filename> <output_PCFG_file>")
    treebank, out_path = sys.argv[1], sys.argv[2]
    rule_counts, lhs_counts, roots = Counter(), Counter(), Counter()
    n_trees = 0
    with open(treebank, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tree = read_tree(line)
            except (ValueError, IndexError):
                sys.stderr.write("Skipping malformed tree on line %d\n" % lineno)
                continue
            roots[tree[0]] += 1
            count_rules(tree, rule_counts, lhs_counts)
            n_trees += 1

    start = roots.most_common(1)[0][0]
    by_lhs = defaultdict(list)
    for (lhs, rhs), c in rule_counts.items():
        by_lhs[lhs].append((rhs, c))

    # start symbol first (NLTK convention), then the rest alphabetically
    order = [start] + sorted(l for l in by_lhs if l != start)
    with open(out_path, "w", encoding="utf-8") as out:
        for lhs in order:
            total = lhs_counts[lhs]
            for rhs, c in sorted(by_lhs[lhs], key=lambda x: (-x[1], x[0])):
                out.write("%s -> %s [%.6f]\n" % (lhs, " ".join(rhs), c / total))
    sys.stderr.write("Induced %d rules over %d non-terminals from %d trees "
                     "(start symbol: %s)\n"
                     % (len(rule_counts), len(by_lhs), n_trees, start))

if __name__ == "__main__":
    main()