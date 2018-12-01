import os
from pathlib import Path

import spacy
import textacy
import textacy.keyterms
from spacy import displacy
from spacy.symbols import VERB, nsubj
from textacy.extract import subject_verb_object_triples

##
# SpaCy: https://spacy.io/usage/linguistic-features and https://spacy.io/usage/examples
# TextaCy: https://chartbeat-labs.github.io/textacy/
##


target_text = "./data/testparse.txt"

print("Loading SpaCy english lg model")

nlp = spacy.load('en_core_web_lg')

print("\nReading main text {}".format(target_text))
mainText = open(target_text).read()

print("\nSpaCy processing...")
doc = nlp(mainText)

print("\nTriples processing...")
svos = textacy.extract.subject_verb_object_triples(doc)

print("\nSVOs:")
print([svo for svo in svos])

print("\nKey terms: ")
print(textacy.keyterms.textrank(doc, normalize='lemma', n_keyterms=10))

print("\nNoun Chunks: ")

for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

print("\nParse Tree: ")

for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])


print("\nVerbs: ")

verbs = set()
for possible_subject in doc:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        verbs.add(possible_subject.head)
print(verbs)

print("\nRendering to sentence.svg: ")
svg = displacy.render(doc, style='dep', jupyter=False)
output_path = Path('./sentence.svg')
output_path.open('w', encoding='utf-8').write(svg)
