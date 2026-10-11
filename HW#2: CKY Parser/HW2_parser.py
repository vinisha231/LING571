#!/usr/bin/env python3 
#python3 cky_parser.py <grammar.cfg> <sentences.txt> <output.txt>

import sys
from collections import defaultdict
from nltk import CFG, Tree
import re

def read_grammar(fname): 
    f = open(fname)
    txt = f.read()
    f.close()
    return CFG.fromstring(txt)

class CKY:
    def __init__(self, g):
        self.start = g.start()
        self.lex_rules = defaultdict(set) #word -> set of LHS
        self.bin_rules = defaultdict(set) #(B, C) -> set of A
        for prod in g.productions():
            r = prod.rhs()
            if len(r) == 1 and isinstance(r[0], str):
                self.lex_rules[r[0]].add(prod.lhs())
            elif len(r) == 2:
                self.bin_rules[r].add(prod.lhs())
            else:
                #shouldnt happen since grammar is cnf
                raise ValueError("grammar not in CNF: " + str(prod))

    def parse(self, toks):
        n = len(toks)
        bp = defaultdict(lambda: defaultdict(list)) #(i,j) -> A -> list of backpointers
        cnt = defaultdict(lambda: defaultdict(int)) #(i,j) -> A -> how many parses
        #diagonal, fill in the words
        for i in range(n):
            for A in self.lex_rules.get(toks[i], ()):
                bp[(i, i + 1)][A].append(None)
                cnt[(i, i + 1)][A] = 1
        #now bigger and bigger spans
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for kk in range(j - 1, i, -1):
                    lcell = bp.get((i, kk))
                    rcell = bp.get((kk, j))
                    if not lcell or not rcell:
                        continue
                    for B in lcell:
                        for C in rcell:
                            for A in self.bin_rules.get((B, C), ()):
                                bp[(i, j)][A].append((kk, B, C))
                                cnt[(i, j)][A] += cnt[(i, kk)][B] * cnt[(kk, j)][C]
        #walk the backpointers to get the actual trees
        def get_trees(i, j, A): 
            for thing in bp[(i, j)][A]:
                if thing is None:
                    yield Tree(str(A), [toks[i]])
                else:
                    kk, B, C = thing
                    for lt in get_trees(i, kk, B):
                        for rt in get_trees(kk, j, C):
                            yield Tree(str(A), [lt, rt])
        ans = 0
        if n > 0:
            ans = cnt[(0, n)][self.start]
        if ans == 0:
            return 0, iter([])
        return ans, get_trees(0, n, self.start)

def main(gfile, sfile, ofile):
    parser = CKY(read_grammar(gfile))
    fin = open(sfile)
    fout = open(ofile, "w")
    first = True
    for line in fin:
        s = line.strip()
        if s == "":
            continue
        num, trees = parser.parse(re.findall(r"\w+|[^\w\s]", s))
        if not first:
            fout.write("\n")
        first = False
        fout.write(s + "\n")
        for t in trees:
            fout.write(str(t) + "\n")
        fout.write("Number of parses: " + str(num) + "\n")
    fin.close()
    fout.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: cky_parser.py <grammar> <sentences> <output>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])