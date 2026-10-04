import sys
import nltk
import re

grammar_file = sys.argv[1]
sentence_file = sys.argv[2]
output_file = sys.argv[3]

text = open(grammar_file).read()
g = nltk.CFG.fromstring(text)
parser = nltk.parse.EarleyChartParser(g) 

total = 0
count = 0

with open(sentence_file) as f, open(output_file, "w") as out:
        for line in f: 
            words = re.findall(r"\w+|[^\w\s]", line.strip())
            trees = list(parser.parse(words))
            out.write(line.strip() + "\n")
            for tree in trees:
                out.write(str(tree)+ "\n")
            out.write("Number of parses: %d\n\n" % len(trees)) 
            total += 1
            count += len(trees)  
 
        if total > 0:
            average = count / total
            print("Average parses per sentence: %.3f" % average)
            out.write("Average parses per sentence: %.3f" % average + "\n")