import numpy as np
import re
from nltk.tokenize.stanford import StanfordTokenizer


ptb_tokenizer = StanfordTokenizer('../../../../data/stanford-parser-full-2018-02-27/stanford-parser.jar')

def post_ptbtokenizer(text):
    return text.split(' ')


def ptbtokenizer(text, context=False):
    text = text.replace('’', "'")
    text = text.replace("..", ".")
    text = text.replace(". .", ".")
    text = text.replace("´", "'")
    text = text.replace("`", "'")
    doc = ptb_tokenizer.tokenize(text)

    output = []
    for token in doc:
        _token = re.split('([$.])', token)
        _token = [i for i in _token if len(i) > 0]
        output = output + _token

    if context:
        output.append('<eoc>')

    return output