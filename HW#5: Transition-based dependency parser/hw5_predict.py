#!/usr/bin/env python3
""" 
Pipeline
  1. Oracle: es_train.conllu -> gold transition sequences (hw5_train.seq)
  2. Features: configuration -> feature dict (see extract_features)
  3. Train: MaxEnt (multinomial logistic regression, sklearn) on (features, gold transition) pairs
  4. Test:  walk through each test sentence using the GOLD transitions, 
            predict a transition at every configuration, write hw5_pred.seq
            and report micro-F1 against es_test.seq.gold

"""
import os
import sys
import random
from collections import Counter, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#Configuration constants
SHIFT= "SHIFT"
LEFT_ARC= "LEFTARC"
RIGHT_ARC= "RIGHTARC"
#Separator between the operator and the dependency label, e.g. LEFTARC_nsubj
#CHECK THIS BEFORE SUBMITTING!!!! (and parse_transition) if toy.seq uses a different convention.
LABEL_SEPARATOR= ", "

ROOT_ID= 0
ROOT_TOKEN_FIELDS= dict(form="ROOT", lemma="ROOT", upos="ROOT", feats="_")
MISSING= "NONE"

MAX_ITERATIONS= 200 #optimizer iterations for logistic regression
RANDOM_SEED= 571

#Feature ablations: set HW5_ABLATE to a comma-separated list of group names to drop them,
#e.g.  HW5_ABLATE=feats,lemmas ./hw5_predict.sh ... (if its not set then full feature template)
ABLATION_GROUPS = {
    "feats": lambda key: key.endswith(".feats"),
    "lemmas": lambda key: key.endswith(".lemma"),
    "forms": lambda key: key.endswith(".form") or key.startswith("s1.form"),
    "pairs": lambda key: "+" in key,
    "dep_labels": lambda key: "dep_rel" in key or key.endswith(".num_deps"),
    "lookahead": lambda key: key.startswith(("s3.", "b2.", "b3.")),
    "structure": lambda key: key in ("stack_depth", "buffer_empty", "s1_s2_distance"),
}
ABLATED_GROUPS = [g for g in os.environ.get("HW5_ABLATE", "").split(",") if g]
for _group in ABLATED_GROUPS:
    if _group not in ABLATION_GROUPS:
        sys.exit("Unknown ablation group %r; choose from %s" % (_group, sorted(ABLATION_GROUPS)))
Token = namedtuple("Token", "token_id form lemma upos feats head deprel")

#CoNLL-U reading
def read_conllu_sentences(path):
    #Return a list of sentences; each sentence is a list of Token (1-indexed ids).
    all_sentences= []
    current_tokens= []
    with open(path, encoding="utf-8") as conllu_file:
        for raw_line in conllu_file:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line.strip():
                if current_tokens:
                    all_sentences.append(current_tokens)
                    current_tokens = []
                continue
            columns = line.split("\t")
            token_id = columns[0]
            #Range ids (1-2) and empty nodes (1.1) are scrubbed, but be safe.
            if not token_id.isdigit():
                continue
            current_tokens.append(Token(
                token_id=int(token_id),
                form=columns[1],
                lemma=columns[2],
                upos=columns[3],
                feats=columns[5],
                head=int(columns[6]),
                deprel=columns[7],
            ))
    if current_tokens:
        all_sentences.append(current_tokens)
    return all_sentences

#Transition sequence reading/writing
def read_transition_sequences(path, sentences):
    #One transition per line, e.g. 'SHIFT' or '(LEFTARC, nsubj)' and blank lines are ignored
    #This method consumes transitions for each sentence until its parse reaches the final state
    with open(path, encoding="utf-8") as seq_file:
        all_transitions = [line.strip() for line in seq_file if line.strip()]
    sequences = []
    position = 0
    for sentence_tokens in sentences:
        state = ParserState(sentence_tokens)
        sentence_transitions = []
        while not state.is_final():
            operator, label = parse_transition(all_transitions[position])
            position += 1
            canonical_transition = make_transition(operator, label)
            sentence_transitions.append(canonical_transition)
            state.apply(canonical_transition)
        sequences.append(sentence_transitions)
    if position != len(all_transitions):
        print("WARNING: %d unused transitions in %s" % (
            len(all_transitions) - position, path), file=sys.stderr)
    return sequences

def write_transition_sequences(path, sequences):
    #Same layout as toy.seq: one transition per line with blank line between sentences
    with open(path, "w", encoding="utf-8") as seq_file:
        for sentence_index, transition_sequence in enumerate(sequences):
            if sentence_index > 0:
                seq_file.write("\n")
            for transition in transition_sequence:
                seq_file.write(transition + "\n")

# Transition helpers
def make_transition(operator, label=None):
    #SHIFT -> 'SHIFT';  ('LEFTARC', 'det') -> '(LEFTARC, det)'  (matches es_test.seq.gold)
    if operator == SHIFT:
        return SHIFT
    return "(%s%s%s)" % (operator, LABEL_SEPARATOR, label)

def parse_transition(transition):
    #'(LEFTARC, nsubj)' or '(LEFTARC,nsubj)' -> ('LEFTARC', 'nsubj'); 'SHIFT' -> ('SHIFT', None)
    transition = transition.strip()
    if transition == SHIFT:
        return SHIFT, None
    if transition.startswith("(") and transition.endswith(")"):
        operator, _, label = transition[1:-1].partition(",")
        operator, label = operator.strip(), label.strip()
        if operator in (LEFT_ARC, RIGHT_ARC):
            return operator, label
    raise ValueError("Unknown transition: %r" % transition)

# Parser configuration (stack, buffer, arcs)
class ParserState:
    def __init__(self, sentence_tokens):
        self.token_by_id = {t.token_id: t for t in sentence_tokens}
        self.stack = [ROOT_ID]
        self.buffer = [t.token_id for t in sentence_tokens]
        self.assigned_head = {} #dependent id -> head id
        self.assigned_label = {} #dependent id -> deprel
        self.left_dependents = {} #head id -> sorted dependent ids left of head
        self.right_dependents = {} #head id -> sorted dependent ids right of head

    def is_final(self):
        return self.stack == [ROOT_ID] and not self.buffer

    def apply(self, transition):
        operator, label = parse_transition(transition)
        if operator == SHIFT:
            self.stack.append(self.buffer.pop(0))
        elif operator == LEFT_ARC:
            head_id = self.stack[-1]
            dependent_id = self.stack.pop(-2)
            self._add_arc(head_id, dependent_id, label)
        else: #RIGHT_ARC
            dependent_id = self.stack.pop()
            head_id = self.stack[-1]
            self._add_arc(head_id, dependent_id, label)

    def _add_arc(self, head_id, dependent_id, label):
        self.assigned_head[dependent_id] = head_id
        self.assigned_label[dependent_id] = label
        side = self.left_dependents if dependent_id < head_id else self.right_dependents
        side.setdefault(head_id, []).append(dependent_id)
        side[head_id].sort()

    #accessors that return placeholders for missing positions 
    def stack_token_id(self, depth):
        #depth=1 is the top of the stack (s1), 2 is s2, ... 
        return self.stack[-depth] if len(self.stack) >= depth else None

    def buffer_token_id(self, position):
        #position=1 is the front of the buffer (b1).
        return self.buffer[position - 1] if len(self.buffer) >= position else None

    def property_of(self, token_id, property_name):
        if token_id is None:
            return MISSING
        if token_id == ROOT_ID:
            return ROOT_TOKEN_FIELDS[property_name]
        value = getattr(self.token_by_id[token_id], property_name)
        return value.lower() if property_name == "form" else value

#Oracle (arc-standard, labeled)
def generate_gold_transitions(sentence_tokens):
    #Run the training oracle; return the list of transitions, or None if the tree could not be derived (e.g. non-projective).
    gold_head = {t.token_id: t.head for t in sentence_tokens}
    gold_label = {t.token_id: t.deprel for t in sentence_tokens}
    unattached_dependents = Counter(gold_head.values())

    state = ParserState(sentence_tokens)
    transitions = []
    while not state.is_final():
        chosen = None
        if len(state.stack) >= 2:
            top_id, second_id = state.stack[-1], state.stack[-2]
            if second_id != ROOT_ID and gold_head[second_id] == top_id:
                chosen = make_transition(LEFT_ARC, gold_label[second_id])
                unattached_dependents[top_id] -= 1
            elif gold_head[top_id] == second_id and unattached_dependents[top_id] == 0:
                chosen = make_transition(RIGHT_ARC, gold_label[top_id])
                unattached_dependents[second_id] -= 1
        if chosen is None:
            if not state.buffer:
                return None #stuck and cannot derive this tree
            chosen = SHIFT
        transitions.append(chosen)
        state.apply(chosen)
    return transitions
 
# Feature extraction 
def extract_features(state):
    """Feature template:
      single positions: s1,s2,s3 {form,lemma,upos}; b1,b2,b3 {upos}; b1 {form,lemma}
      morphology: s1,s2,b1 feats (Spanish case/number/tense etc. in UD FEATS)
      pairs: s1.upos+s2.upos, s1.form+s2.form, s1.upos+b1.upos, s2.upos+s1.upos+b1.upos, s1.form+b1.upos
      arc context: deprel of leftmost/rightmost dependent of s1 and s2
      structure: stack depth, buffer emptiness, s1-s2 linear distance bucket
    """
    s1, s2, s3= (state.stack_token_id(depth) for depth in (1, 2, 3))
    b1, b2, b3= (state.buffer_token_id(pos) for pos in (1, 2, 3))
    prop= state.property_of

    features = {}
    for name, token_id in (("s1", s1), ("s2", s2), ("s3", s3)):
        features[name + ".form"] = prop(token_id, "form")
        features[name + ".lemma"] = prop(token_id, "lemma")
        features[name + ".upos"] = prop(token_id, "upos")
    for name, token_id in (("b1", b1), ("b2", b2), ("b3", b3)):
        features[name + ".upos"] = prop(token_id, "upos")
    features["b1.form"] = prop(b1, "form")
    features["b1.lemma"] = prop(b1, "lemma")

    for name, token_id in (("s1", s1), ("s2", s2), ("b1", b1)):
        features[name + ".feats"] = prop(token_id, "feats")

    features["s1.upos+s2.upos"] = prop(s1, "upos") + "|" + prop(s2, "upos")
    features["s1.form+s2.form"] = prop(s1, "form") + "|" + prop(s2, "form")
    features["s1.upos+b1.upos"] = prop(s1, "upos") + "|" + prop(b1, "upos")
    features["s2.upos+s1.upos+b1.upos"] = "|".join(
        (prop(s2, "upos"), prop(s1, "upos"), prop(b1, "upos")))
    features["s1.form+b1.upos"] = prop(s1, "form") + "|" + prop(b1, "upos")

    for name, token_id in (("s1", s1), ("s2", s2)):
        left = state.left_dependents.get(token_id, []) if token_id is not None else []
        right = state.right_dependents.get(token_id, []) if token_id is not None else []
        features[name + ".leftmost_dep_rel"] = (
            state.assigned_label[left[0]] if left else MISSING)
        features[name + ".rightmost_dep_rel"] = (
            state.assigned_label[right[-1]] if right else MISSING)
        features[name + ".num_deps"] = str(len(left) + len(right))

    features["stack_depth"] = str(min(len(state.stack), 5))
    features["buffer_empty"] = str(not state.buffer)
    if s1 is not None and s2 is not None:
        distance = s1 - s2
        features["s1_s2_distance"] = str(distance) if distance < 4 else "4+"
    else:
        features["s1_s2_distance"] = MISSING
    if ABLATED_GROUPS:
        features = {key: value for key, value in features.items()
                    if not any(ABLATION_GROUPS[group](key) for group in ABLATED_GROUPS)}
    return features

def sentence_to_training_instances(sentence_tokens, gold_transitions):
    """Replay gold transitions, yielding (feature_dict, transition) at each step."""
    state = ParserState(sentence_tokens)
    instances = []
    for transition in gold_transitions:
        instances.append((extract_features(state), transition))
        state.apply(transition)
    return instances
 
#Train/test 
class SklearnMaxentClassifier:
    #Multinomial logistic regression (= MaxEnt) with an NLTK-like classify().
    def __init__(self, training_instances):
        feature_dicts = [features for features, _ in training_instances]
        transition_labels = [label for _, label in training_instances]
        self.vectorizer = DictVectorizer()
        feature_matrix = self.vectorizer.fit_transform(feature_dicts)
        self.model = LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_SEED)
        self.model.fit(feature_matrix, transition_labels)

    def classify(self, features):
        return self.model.predict(self.vectorizer.transform([features]))[0]

def train_classifier(training_instances):
    random.seed(RANDOM_SEED)
    return SklearnMaxentClassifier(training_instances)

def predict_along_gold_path(classifier, sentence_tokens, gold_transitions):
    #Advance configurations with the gold transitions; record the classifier's prediction at every configuration.
    state = ParserState(sentence_tokens)
    predicted_transitions = []
    for gold_transition in gold_transitions:
        predicted_transitions.append(classifier.classify(extract_features(state)))
        state.apply(gold_transition)
    return predicted_transitions


def main():
    if len(sys.argv) != 6:
        sys.exit("Usage: %s <train_data> <test_data> <test_labels> <train_seq> <pred_seq>"
                 % sys.argv[0])
    train_path, test_path, test_labels_path, train_seq_path, pred_seq_path = sys.argv[1:]
    #Part 1: oracle
    train_sentences = read_conllu_sentences(train_path)
    train_transition_sequences = []
    training_instances = []
    skipped_sentence_count = 0
    for sentence_tokens in train_sentences:
        gold_transitions = generate_gold_transitions(sentence_tokens)
        if gold_transitions is None:
            skipped_sentence_count += 1
            continue
        train_transition_sequences.append(gold_transitions)
        #Part 2: features
        training_instances.extend(
            sentence_to_training_instances(sentence_tokens, gold_transitions))
    write_transition_sequences(train_seq_path, train_transition_sequences)
    print("Train sentences: %d (skipped %d), training instances: %d" % (
        len(train_transition_sequences), skipped_sentence_count, len(training_instances)),
        file=sys.stderr)

    #Part 3: train
    classifier = train_classifier(training_instances)

    #Part 3: test
    test_sentences = read_conllu_sentences(test_path)
    gold_test_sequences = read_transition_sequences(test_labels_path, test_sentences)

    predicted_sequences = []
    flat_gold, flat_pred = [], []
    for sentence_tokens, gold_transitions in zip(test_sentences, gold_test_sequences):
        predicted = predict_along_gold_path(classifier, sentence_tokens, gold_transitions)
        predicted_sequences.append(predicted)
        flat_gold.extend(gold_transitions)
        flat_pred.extend(predicted)
    write_transition_sequences(pred_seq_path, predicted_sequences)

    micro_f1 = f1_score(flat_gold, flat_pred, average="micro")
    macro_f1 = f1_score(flat_gold, flat_pred, average="macro")
    weighted_f1 = f1_score(flat_gold, flat_pred, average="weighted")
    print("Test transitions: %d" % len(flat_gold))
    print("Micro F1:    %.4f" % micro_f1)
    print("Macro F1:    %.4f" % macro_f1)
    print("Weighted F1: %.4f" % weighted_f1)

if __name__ == "__main__":
    main()