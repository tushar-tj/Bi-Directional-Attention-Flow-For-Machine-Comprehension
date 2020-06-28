import torch
import argparse
import pickle
import time
import numpy as np
import datetime
from load import ptbtokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='BiDaF for Machine Comprehension & Cloze-Style Reading Comprehension - Prediction')

# Input data
parser.add_argument('-context', type=str, required=True, help='Context in text')
parser.add_argument('-query', type=str, required=True, help='Query in text')
parser.add_argument('-model_path', type=str, required=True, help='path to model')
parser.add_argument('-word_vocab', type=str, required=True, help='path to word vocab')
parser.add_argument('-char_vocab', type=str, required=True, help='path to char vocab')

args = parser.parse_args()

def load_vocab(path):
    inp = open(path, 'rb')
    vocab = pickle.load(inp)
    inp.close()

    return vocab

def predict_data(context_tokenized, question_tokenized, word_vocab, char_vocab):
    longest_context_word = max([len(w) for w in context_tokenized])
    longest_question_word = max([len(w) for w in question_tokenized])

    context_words = (torch.tensor([[word_vocab[word.lower()] for word in context_tokenized]]),
                     torch.tensor([len(context_tokenized)]))

    question_words = (torch.tensor([[word_vocab[word.lower()] for word in question_tokenized]]),
                      torch.tensor([len(question_tokenized)]))

    context_char = []
    for word in context_tokenized:
        _context_word = []
        for c_index in range(longest_context_word):
            if c_index < len(word):
                _context_word.append(char_vocab[word[c_index]])
            else:
                _context_word.append(char_vocab['<pad>'])
        context_char.append(_context_word)
    context_char = torch.tensor([context_char])

    question_char = []
    for word in question_tokenized:
        _question_word = []
        for c_index in range(longest_question_word):
            if c_index < len(word):
                _question_word.append(char_vocab[word[c_index]])
            else:
                _question_word.append(char_vocab['<pad>'])
        question_char.append(_question_word)
    question_char = torch.tensor([question_char])

    predict_data.context_words = context_words
    predict_data.question_words = question_words
    predict_data.context_char = context_char
    predict_data.question_char = question_char
    predict_data.batch_size = 1


def get_prediction(context, question, model, word_vocab, char_vocab):
    model.eval()
    context_tokenized = ptbtokenizer(context, context=True)
    question_tokenized = ptbtokenizer(question)

    predict_data(context_tokenized, question_tokenized, word_vocab, char_vocab)
    p1, p2 = model(predict_data)

    answer = " ".join(context_tokenized[p1.argmax(1): p2.argmax(1)])

    return answer, p1.argmax(1), p2.argmax(1)

start_time = time.time()

WORD = load_vocab(args.word_vocab)
CHAR = load_vocab(args.char_vocab)
MODEL = torch.load(args.model_path)

answer, abs_start_index, ans_end_index = get_prediction(args.context, args.query, MODEL, WORD, CHAR)
time_delta = datetime.timedelta(seconds=np.round(time.time() - start_time, 0))

print(f'Time: {time_delta} , ANSWER: {answer}')