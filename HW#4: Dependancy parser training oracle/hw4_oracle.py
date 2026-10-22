#!/usr/bin/env python3
""" 
Usage: 
This program reads gold-standard parses in the simplified CoNLL format
<index>\t<word>\t<dependency_relation>\t<head_index>
(ROOT is index 0 and has no row of its own and blank lines separate sentences),
runs the arc-standard oracle on each one, and writes
  * the sequence of transitions (one per line, with a blank line between sentences)
  * the arcs built by applying those transitions, in the input format, sorted by word index.
"""
import sys
from collections import namedtuple

SHIFT = "SHIFT"
LEFTARC= "LEFTARC"
RIGHTARC= "RIGHTARC"
ROOT = 0 #index of the artificial root node

Token = namedtuple("Token", "index word label head")
#An arc is (head_index, dependent_index, label).
Arc = namedtuple("Arc", "head dep label")

#Reading/writing
def read_parses(path):
    #Yield one sentence at a time as a list of Token objects.
    sentence = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                if sentence:
                    yield sentence
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) < 4: #be forgiving if tabs were turned into spaces :/
                fields = line.split()
            index, word, label, head = fields[0], fields[1], fields[2], fields[3]
            sentence.append(Token(int(index), word, label, int(head)))
    if sentence:
        yield sentence

def format_transition(transition):
    #SHIFT -> 'SHIFT';  (LEFTARC, 'det') -> '(LEFTARC,det)'.
    if transition == SHIFT:
        return SHIFT
    op, label = transition
    return "({},{})".format(op, label)

def format_arcs(sentence, arcs):
    #Arcs as 'index<TAB>word<TAB>label<TAB>head' lines, sorted by index.
    words = {tok.index: tok.word for tok in sentence}
    lines = []
    for arc in sorted(arcs, key=lambda a: a.dep):
        lines.append("{}\t{}\t{}\t{}".format(arc.dep, words[arc.dep], arc.label, arc.head))
    return lines

# Arc-standard transition system
class Configuration:
    """A parser configuration: (stack, buffer, arcs).
    The stack and buffer hold the word indices (ROOT == 0). The top of the stack is
    the end of the list, the front of the buffer is the start of the list.
    """
    def __init__(self, n_words):
        self.stack = [ROOT]
        self.buffer = list(range(1, n_words + 1))
        self.arcs = []

    def is_final(self):
        return not self.buffer and self.stack == [ROOT]

    def __repr__(self):
        return "({}, {}, {})".format(self.stack, self.buffer, self.arcs)

def apply_transition(config, transition):
    """Apply one transition to a configuration (in place) and return it.
    SHIFT: move the front of the buffer onto the stack.
    (LEFTARC, l): top of stack is the head of the second item; add arc, pop the second item.
    (RIGHTARC, l): second item is the head of the top of stack; add arc, pop the top item.
    Illegal transitions raise ValueError instead of corrupting state.
    """
    if transition == SHIFT:
        if not config.buffer:
            raise ValueError("SHIFT with an empty buffer")
        config.stack.append(config.buffer.pop(0))
        return config

    op, label = transition
    if len(config.stack) < 2:
        raise ValueError("{} needs two items on the stack".format(op))
    s1, s2 = config.stack[-1], config.stack[-2]

    if op == LEFTARC:
        if s2 == ROOT:
            raise ValueError("LEFTARC cannot make ROOT a dependent")
        config.arcs.append(Arc(head=s1, dep=s2, label=label))
        del config.stack[-2]
    elif op == RIGHTARC:
        config.arcs.append(Arc(head=s2, dep=s1, label=label))
        config.stack.pop()
    else:
        raise ValueError("unknown transition {!r}".format(transition))
    return config

# The training oracle
class Oracle:
    """Training oracle backed by a gold-standard parse.
    Policy, in order:
      1. LEFTARC(l) if the gold parse has  S1 --l--> S2  (and S2 is not ROOT)
      2. RIGHTARC(l) if the gold parse has  S2 --l--> S1 and every gold dependent of S1 has 
      already been attached in the config
      3. SHIFT otherwise

    Condition 2b is what stops us popping a word before it has collected all of its dependents.
    <maybe not necessary? remove this comment before submitting> We track it with a counter per head instead of rescanning the arc list.
    """
    def __init__(self, sentence):
        self.gold_head= {t.index: t.head for t in sentence}
        self.gold_label= {t.index: t.label for t in sentence}
        #how many dependents each head has in the gold parse
        self.n_gold_deps= {}
        for t in sentence:
            self.n_gold_deps[t.head] = self.n_gold_deps.get(t.head, 0) + 1
        #how many of those we have attached so far
        self.n_attached = {}

    def next_transition(self, config):
        #Return the correct transition for this configuration.
        if len(config.stack) >= 2:
            s1, s2 = config.stack[-1], config.stack[-2]
            if s2 != ROOT and self.gold_head[s2] == s1:
                return (LEFTARC, self.gold_label[s2])
            if (s1 != ROOT and self.gold_head[s1] == s2
                    and self.n_attached.get(s1, 0) == self.n_gold_deps.get(s1, 0)):
                return (RIGHTARC, self.gold_label[s1])
        return SHIFT

    def record(self, config, transition):
        #Update the attached-dependent counter after applying a transition.
        if transition != SHIFT:
            head = config.arcs[-1].head
            self.n_attached[head] = self.n_attached.get(head, 0) + 1

def run_oracle(sentence):
    """Run the oracle over one gold parse.
    Returns (transitions, arcs, ok) where ok is False if the oracle got stuck
    (e.g. a non-projective or malformed gold parse).
    """
    config = Configuration(len(sentence))
    oracle = Oracle(sentence)
    transitions = []
    while not config.is_final():
        transition = oracle.next_transition(config)
        if transition == SHIFT and not config.buffer:
            #Nothing left to shift and no arc is licensed: the gold tree
            #can't be built with arc-standard (non-projective / bad data).
            return transitions, config.arcs, False
        apply_transition(config, transition)
        oracle.record(config, transition)
        transitions.append(transition)
    return transitions, config.arcs, True

# Main
def main(argv):
    if len(argv) != 4:
        sys.stderr.write(
            "usage: {} <input_dependency_parse_filename> "
            "<output_dependency_filename> <output_sequence_filename>\n".format(argv[0]))
        return 2
    in_path, dep_out_path, seq_out_path = argv[1:4]

    n_sent = n_bad = 0
    with open(dep_out_path, "w", encoding="utf-8") as dep_out, \
            open(seq_out_path, "w", encoding="utf-8") as seq_out:
        for sentence in read_parses(in_path):
            n_sent += 1
            transitions, arcs, ok = run_oracle(sentence)
            if not ok or len(arcs) != len(sentence):
                n_bad += 1
                sys.stderr.write(
                    "warning: sentence {} could not be fully derived "
                    "(non-projective or malformed?)\n".format(n_sent))
            for t in transitions:
                seq_out.write(format_transition(t) + "\n")
            seq_out.write("\n")
            for line in format_arcs(sentence, arcs):
                dep_out.write(line + "\n")
            dep_out.write("\n")
    sys.stderr.write("processed {} sentences ({} problematic)\n".format(n_sent, n_bad))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))