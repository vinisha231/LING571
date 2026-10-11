HW2 readme 
 Vinisha Bala Dhayanidhi  
Worked individually  


1. Overview 
This submission is a CKY parser for grammars that are already in the Chomsky Normal Form. It reads a .cfg grammar and a file of sentences, and for each sentence, it writes out the sentence, every parse the grammar licenses (as a bracketed tree), and the number of parses.


2. Design
Grammar loading: The .cfg file is read with an ordinary file I/O and then turned into an NLTK grammar with CFG.fromstring(), which is very similar to the program that I coded for HW1. NLTK is only used as a data structure for the grammar and I wrote the parsing algorithm. 


Rule indexes: At startup, I split the productions into two lookup tables:
* lexical rules: word -> set of LHS symbols (A -> 'word')
* binary rules: (B,C) -> set of LHS symbols (A -> B C)
I start indexing by the right-hand side which means that while filling the chart, I can
look up every A that can cover a pair of child categories directly instead of scanning the whole rule list. If a rule is neither lexical nor binary, the program raises an error, since the grammar is supposed to be in the CNF.


Chart and backpointers: The chart is a dictionary keyed by span (i, j). Each cell maps a nonterminal to a list of backpointers. A backpointer is either None (a lexical entry) or a tuple (k, B, C) recording the split point and the two child categories. I have appended a backpointer for every way a symbol can be built, instead of keeping only one, so that no derivation is lost. This makes the parser return all the parses.


Filling the chart: The diagonal is filled from the lexical rules. Then, the spans are processed by increasing length; for each span (i, j) every split point k is tried, and for every pair of nonterminals in the left and right cells the binary-rule index is consulted. The overall time is O(n^3 * |G|) for filling the chart.


Counting vs. generating: I keep a second table that counts the parses of each symbol in each cell. The count for A over (i, j) is the sum, over its backpointers, of count(B over i..k) * count(C over k..j). The number of parses for a sentence is the count of the start symbol over the whole sentence. This means the count is computed in polynomial time and does not require building
any trees. The trees themselves are built lazily by a recursive generator that follows the backpointers and yields NLTK Tree objects, and they are only built when the output file is being written. A sentence with a very large number of parses therefore does not slow down the counting step.


Failure cases: A sentence containing a word that is not in the grammar, or one that the grammar cannot derive, ends up with a count of 0 and is output with no trees and "Number of parses: 0". The program does not crash on these.


3. Problems and how I solved them
Every real sentence initially got 0 parses. My first run on grammar_cnf.cfg gave "Number of parses: 0" for every sentence. This was because I had split sentences with str.split(), which leaves punctuation attached to the final word ("system." instead of "system" and "."). The grammar has separate PUNC rules for '.' and '?', so these tokens could never be matched. I confirmed this by grepping the grammar for the punctuation terminals and for words like 'system' and 'Scientists'. I switched to the same regex tokenizer I used in HW1, re.findall(r"\w+|[^\w\s]", sentence), which splits punctuation into its own token. I did not lowercase anything, because the grammar contains capitalized terminals such as 'Scientists'. After this change, none of the 15 test sentences had 0 parses. I did not notice the problem on the toy data because the toy sentences contain no punctuation.


4. Results
* toy.cfg / toy_sentences.txt: output is identical to toy_output.txt.
* grammar_cnf.cfg / sentences.txt: all 15 sentences are parsed and everyone receives at least one parse. hw2_output.txt was produced through
* condor_submit hw2.cmd, and it is identical to the output of running hw2_parser.sh by hand on the same inputs.


5. What I learned
* Storing a list of backpointers per symbol per cell, instead of one, is what turns a recognizer into a parser that returns all analyses.
* Counting parses with dynamic programming over the chart is much cheaper than enumerating trees, and the two can be kept separate.
* Tokenization has to agree with the grammar. A parser can be perfectly correct and still produce nothing if the tokens do not match the grammar's terminals, and the symptom (all zeros) looks like a bug in the parsing algorithm. Also, checking the grammar directly is the quickest way to tell the two apart.
* comparing against the example output with diff, even on a tiny grammar, catches formatting and ordering issues early.


6. Limitations
* The grammar must already be in CNF; unary non-lexical rules and rules with more than two symbols on the right are not handled.
* Words missing from the grammar produce 0 parses rather than any kind of partial analysis.
* Writing out all trees for a highly ambiguous sentence could be slow and produce a very large file, even though counting stays fast.