"""
hw3_pcky.py shared library PCFG reader + probabilistic
CKY (Viterbi) parser.  Everything is done in log space to prevent underflow.

The chart is a (n+1) x (n+1) table whose cell [i][j] is a dict non-terminal A -> (best log-prob of A spanning i..j, backpointer),
i.e. the "third dimension" (V non-terminals).
Only the single best derivation for each (i, j, A) is stored.
"""
import math
import re
from collections import defaultdict

RULE_RE = re.compile(r"^\s*(\S+)\s*->\s*(.+?)\s*\[\s*([0-9eE+.\-]+)\s*\]\s*$")
SYM_RE = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")

class PCFG:
    """A PCFG in NLTK format.  Handles lexical rules (A -> 'w'), unaryrules (A -> B) and binary rules (A -> B C); anything else 
    is ignored."""
    def __init__(self, path):
        self.start = None
        self.lexicon = defaultdict(dict) #word -> {tag: logprob}
        self.unary = defaultdict(list) #B -> [(A, logprob)]
        self.binary = defaultdict(list) #(B, C) -> [(A, logprob)]
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = RULE_RE.match(line)
                if not m:
                    continue
                lhs, rhs_str, prob = m.group(1), m.group(2), float(m.group(3))
                if prob <= 0.0:
                    continue
                lp = math.log(prob)
                if self.start is None: #first rule's LHS = start symbol
                    self.start = lhs
                rhs = SYM_RE.findall(rhs_str)
                if len(rhs) == 1 and rhs[0][0] in "'\"":
                    self.lexicon[rhs[0][1:-1]][lhs] = lp
                elif len(rhs) == 1:
                    self.unary[rhs[0]].append((lhs, lp))
                elif len(rhs) == 2:
                    self.binary[(rhs[0], rhs[1])].append((lhs, lp))
 
    def _unary_closure(self, cell):
        """Apply unary rules (e.g. TOP -> S) until nothing improves."""
        queue = list(cell)
        while queue:
            b = queue.pop()
            lb = cell[b][0]
            for a, lp in self.unary.get(b, ()):
                s = lp + lb
                if a not in cell or s > cell[a][0] + 1e-12:
                    cell[a] = (s, ("un", b))
                    queue.append(a)

    def parse(self, words, lex_fn):
        """Return (bracketed best parse string, log-prob), or (None, None).
        lex_fn(word) -> {tag: logprob} supplies the lexical scores, which
        is the only place the baseline and improved parsers differ."""
        n = len(words)
        if n == 0:
            return None, None
        chart = [[None] * (n + 1) for _ in range(n + 1)]

        #length-1 spans: lexical rules (+ unary closure)  
        for i, w in enumerate(words):
            cell = {tag: (lp, ("lex", w)) for tag, lp in lex_fn(w).items()}
            if not cell:
                return None, None #a word with no tag => no parse
            self._unary_closure(cell)
            chart[i][i + 1] = cell

        #longer spans: binary rules over every split point k
        binary = self.binary
        for span in range(2, n + 1):
            for i in range(0, n - span + 1):
                j = i + span
                cell = {}
                for k in range(i + 1, j):
                    left, right = chart[i][k], chart[k][j]
                    if not left or not right:
                        continue
                    for b, (lb, _) in left.items():
                        for c, (lc, _) in right.items():
                            for a, lp in binary.get((b, c), ()):
                                s = lp + lb + lc
                                if a not in cell or s > cell[a][0]:
                                    cell[a] = (s, ("bin", k, b, c))
                if cell:
                    self._unary_closure(cell)
                chart[i][j] = cell

        top = chart[0][n]
        if not top or self.start not in top:
            return None, None
        return self._build(chart, 0, n, self.start), top[self.start][0]

    def _build(self, chart, i, j, a):
        _, bp = chart[i][j][a]
        if bp[0] == "lex":
            return "(%s %s)" % (a, bp[1])
        if bp[0] == "un":
            return "(%s %s)" % (a, self._build(chart, i, j, bp[1]))
        _, k, b, c = bp
        return "(%s %s %s)" % (a, self._build(chart, i, k, b),
                               self._build(chart, k, j, c))

def exact_lex_fn(grammar):
    """Baseline lexical lookup: the word must have been seen in training."""
    return lambda w: grammar.lexicon.get(w, {})

def parse_file(grammar, lex_fn, sent_path, out_path):
    """Parse every line of sent_path; write one parse (or a blank line) per input line to out_path."""
    n_parsed = n_total = 0
    with open(sent_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            words = line.split()
            n_total += 1
            tree, _ = grammar.parse(words, lex_fn)
            if tree is None:
                fout.write("\n")
            else:
                fout.write(tree + "\n")
                n_parsed += 1
    return n_parsed, n_total