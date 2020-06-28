from load import LoadData
from preprocess import PreprocessData
from model import BidaF
from evaluate import evaluate as eval
from load import ptbtokenizer
import torch.optim as optim
import torch.nn as nn

import torch
import pickle
import time
import sys
import argparse
import copy
import numpy as np
import datetime
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_checkpoint_paths(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


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


def train(model, optimizer, criterion, path, epochs, epochs_log, model_name):
    start_time = time.time()
    for epoch in range(epochs):
        # SETTING MODEL IN TRAINING MODE
        model.train()

        epoch_loss = 0
        batch_num = 0.0
        exmaples_count = 0.0
        best_dev_acc_exact = -1.0
        best_dev_acc_f1 = -1.0

        for train_data in iter(data_preprocessor.train_iter):
            batch_num += 1.0
            exmaples_count += train_data.batch_size
            p1, p2 = model(train_data)
            optimizer.zero_grad()
            try:
                batch_loss = criterion(p1, train_data.start_idx) + criterion(p2, train_data.end_idx)
            except Exception as e:
                print(e)
                return (p1, p2, train_data)

            epoch_loss += batch_loss.item()
            batch_loss.backward()

            optimizer.step()

            time_delta = datetime.timedelta(seconds=np.round(time.time() - start_time, 0))
            sys.stdout.write(f'\rEpoch:{epoch} | Batch:{batch_num} | Time Running: {time_delta}')
            break

        if epoch % epochs_log == 0:

            train_loss = epoch_loss/(exmaples_count)
            dev_accuracy, dev_loss = eval(data_preprocessor.dev_iter,
                                              model,
                                              criterion,
                                              data_preprocessor.WORDS.vocab,
                                              calculate_loss=True,
                                              calculate_accuracy=True)
            dev_accuracy_exact = dev_accuracy.groupby('id')['Exact'].max().mean()
            dev_accuracy_f1 = dev_accuracy.groupby('id')['F1'].max().mean()

            train_accuracy, _ = eval(data_preprocessor.train_iter,
                                         model,
                                         criterion,
                                         data_preprocessor.WORDS.vocab,
                                         calculate_loss=False,
                                         calculate_accuracy=True)

            train_accuracy_exact = train_accuracy.groupby('id')['Exact'].max().mean()
            train_accuracy_f1 = train_accuracy.groupby('id')['F1'].max().mean()

            print(
                f'\nTrain Loss:{train_loss:.4f} Train Acc Exact:{train_accuracy_exact:.4f} Train Acc F1:{train_accuracy_f1:.4f}')
            print(
                f'Validation Loss :{dev_loss:.4f} Dev Acc Exact:{dev_accuracy_exact:.4f} Dev Acc F1:{dev_accuracy_f1:.4f}')

            print('Test Prediction Results')
            predict_context = "He was speaking after figures showed that the country's economy shrank by 20.4% in April - " \
                              "the largest monthly contraction on record - as the country spent its first full month in lockdown."
            predict_ques = "By how much did the country's economy shrank"

            print(get_prediction(predict_context,
                                 predict_ques,
                                 model,
                                 data_preprocessor.WORDS.vocab,
                                 data_preprocessor.CHAR.vocab))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, path + '/' + model_name + '.torch')

            if dev_accuracy_f1 > best_dev_acc_f1:
                best_dev_acc_f1 = dev_accuracy_f1
                best_dev_acc_exact = dev_accuracy_exact
                torch.save(model, path + '/' + 'best_' + model_name + '.torch')

    print (f'Best Validation Results '
           f'Dev Acc Exact:{best_dev_acc_exact:.4f} '
           f'Dev Acc F1:{best_dev_acc_f1:.4f}')


parser = argparse.ArgumentParser(description='BiDaF for Machine Comprehension & Cloze-Style Reading Comprehension - Training')

# Input data
parser.add_argument('-data', type=str, default='../../../../data/squad v1.1/', help='Path to input data')

# Checkpoints paths
parser.add_argument('-data_checkpoint', type=str, default='./data_checkpoints', help='Path to store preprocessed data checkpoints')
parser.add_argument('-model_checkpoint', type=str, default='./model_checkpoints', help='Path to store modelled data checkpoints')
parser.add_argument('-model_name', type=str, default=None, required=True, help='provide a name to the model for storing at chekpoints')
parser.add_argument('-dataset_name', type=str, default=None, required=True, help='Name of the Dataset')

parser.add_argument('-load_data', type=str2bool, nargs='?', default=False, help='To Load data of use preprocessed data')

# Modelling parameters
parser.add_argument('-epochs', type=int, default=20, help='No. of Epoch to run')
parser.add_argument('-batch_size', type=int, default=60, help='Number of examples in each batch')
parser.add_argument('-glove_size', type=int, default=100, help='Size of Glove  vector to use')
parser.add_argument('-char_embedding_size', type=int, default=100, help='Size of Character embeddings to be used')
parser.add_argument('-kernel_size', type=int, default=5, help='Kernel Size')
parser.add_argument('-channels_count', type=int, default=100 ,help='Count of channels for character embeddings')
parser.add_argument('-learning_rate', type=float, default=0.5, help='Learning Rate')
parser.add_argument('-epoch_log', type=int, default=2, help='Print logs after xx epochs')

args = parser.parse_args()

if args.load_data:
    _ = LoadData(data_path=args.data,
                 checkpoint_path=args.data_checkpoint,
                 train_file=args.dataset_name + '_' + 'train.csv',
                 dev_file=args.dataset_name + '_' + 'dev.csv')

# Create Checkpoint folders for in between sessions storage
create_checkpoint_paths(args.data_checkpoint)
create_checkpoint_paths(args.model_checkpoint)

data_preprocessor = PreprocessData(data_path=args.data_checkpoint,
                                   glove_size=args.glove_size,
                                   batch_size=args.batch_size,
                                   train_file='train1.csv',
                                   dev_file='dev1.csv')
                                   # train_file=args.dataset_name + '_' + 'train.csv',
                                   # dev_file=args.dataset_name + '_' + 'dev.csv')

save_vocab(data_preprocessor.WORDS.vocab, args.model_checkpoint + '/' + args.dataset_name + '_' + 'WORDS.vocab')
save_vocab(data_preprocessor.CHAR.vocab, args.model_checkpoint + '/' + args.dataset_name + '_' + 'CHAR.vocab')

# Initializing Bidaf Model
model = BidaF(data_preprocessor.WORDS,
              data_preprocessor.CHAR,
              char_embedding_size=args.char_embedding_size,
              char_conv_kernel_size=args.kernel_size,
              char_conv_channels_count=args.channels_count).to(device)

# Intialize Optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.999)
criterion = nn.CrossEntropyLoss()

## This is a piece of code for retraining the model from where left
# if os.path.isfile(args.model_checkpoint + '/model.torch'):
#     checkpoint = torch.load(args.model_checkpoint + '/model.torch')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print('Loaded Trained Model')
#     print(f'Trained for {epoch} Epochs, Achieved {loss} Training Loss')

_error = train(model=model,
               optimizer=optimizer,
               criterion=criterion,
               path=args.model_checkpoint,
               epochs=args.epochs,
               epochs_log=args.epoch_log,
               model_name=args.dataset_name + '_' + args.model_name)