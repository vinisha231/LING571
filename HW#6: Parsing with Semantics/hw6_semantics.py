#!/usr/bin/env python3
"""
This program loads a feature grammar with FOL semantic attachments (.fcfg), parses each sentence in the input file,
and writes the sentence followed by its first-order logic representation to the output file.
"""

import sys
from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureChartParser
from nltk.sem.util import root_semrep

def load_semantic_grammar(grammar_path):
    #This method reads a .fcfg file and returns a FeatureGrammar with SEM values parsed as logical expressions.
    with open(grammar_path, encoding="utf-8") as grammar_file:
        grammar_text = grammar_file.read()
    return FeatureGrammar.fromstring(grammar_text, logic_parser=None)

def read_sentences(sentences_path):
    #This method returns the non-empty lines of the sentences file, stripped.
    with open(sentences_path, encoding="utf-8") as sentences_file:
        return [line.strip() for line in sentences_file if line.strip()]


def collect_lexicon(grammar):
    #This method returns the set of terminal words the grammar knows about.
    lexicon = set()
    for production in grammar.productions():
        if production.is_lexical():
            lexicon.update(production.rhs())
    return lexicon


def tokenize(sentence, lexicon):
    """This method does whitespace tokenization and drops any sentence-final punctuation.
    The sentence-initial word is lowercased only when the grammar knows the lowercase form, so 'Every' -> 'every' 
    but a proper name like 'Alex' keeps its capital letter.
    """
    tokens = sentence.rstrip(".?!").split()
    if tokens and tokens[0] not in lexicon and tokens[0].lower() in lexicon:
        tokens[0] = tokens[0].lower()
    return tokens

def analyze_sentence(parser, lexicon, sentence):
    #This method returns the simplified semantic representation of the first parse, or None if the sentence cannot be parsed.
    try:
        for parse_tree in parser.parse(tokenize(sentence, lexicon)):
            #Only a single reading is needed for ambiguous sentences
            return str(root_semrep(parse_tree).simplify())
    except ValueError:
        #This is raised by NLTK when the grammar doesn't cover an input word
        pass
    return None

def main(grammar_path, sentences_path, output_path):
    grammar = load_semantic_grammar(grammar_path)
    parser = FeatureChartParser(grammar)
    lexicon = collect_lexicon(grammar)
    sentences = read_sentences(sentences_path)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for sentence in sentences:
            semantic_representation = analyze_sentence(parser, lexicon, sentence)
            if semantic_representation is None:
                semantic_representation = "NO PARSE"
            output_file.write(sentence + "\n")
            output_file.write(semantic_representation + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: hw6_semantics.sh <input_grammar_filename> "
                 "<input_sentences_filename> <output_filename>")
    main(sys.argv[1], sys.argv[2], sys.argv[3])