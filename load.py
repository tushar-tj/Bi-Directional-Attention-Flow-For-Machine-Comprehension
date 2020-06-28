import pandas as pd
import json
import sys
import time
import torch
import datetime
import numpy as np
from tokenizer import ptbtokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LoadData():
    def __init__(self,
                 data_path,
                 checkpoint_path,
                 train_file='train.csv',
                 dev_file='dev.csv'):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path

        self.train_data = self.load_train_data()
        self.dev_data = self.load_dev_data()
        self.train_file = train_file
        self.dev_file = dev_file

    def load_train_data(self):
        with open(self.data_path + 'train-v1.1.json', 'r') as f:
            train_data = json.load(f)
        print(f'\nFlattening SQUAD {train_data["version"]} - Train')
        train_data_flat, errors = self.load_squad_data(train_data)
        print(f'\nErronous Train Datapoints: {errors}')
        pd.DataFrame(train_data_flat).to_csv(self.checkpoint_path + self.train_file, encoding='utf-8')

    def load_dev_data(self):
        with open(self.data_path + 'dev-v1.1.json', 'r') as f:
            dev_data = json.load(f)
        print(f'\nFlattening SQUAD Dev')
        dev_data_flat, errors = self.load_squad_data(dev_data)
        print(f'\nErronous Dev Datapoints: {errors}')
        pd.DataFrame(dev_data_flat).to_csv(self.checkpoint_path + self.dev_file, encoding='utf-8')

    def convert_charidx_to_wordidx(self, context_tok, ans_tok):
        length = len(ans_tok)
        _start, _end = None, None

        for i in range(len(context_tok)):
            context_text = " ".join(context_tok[i:i + length]).strip('.')
            ans_text = " ".join(ans_tok).strip('.')
            _ans_text = "".join(ans_tok).strip('.')
            __ans_text = " ".join(ans_tok).strip(',')

            if ans_text in context_text or _ans_text in context_text or __ans_text in context_text:
                _start = i
                _end = i + length
                break

        if _start == None:
            for i in range(len(context_tok)):
                if ans_tok[0] == context_tok[i]:
                    _start = i
                    _end = i + length

        if _start == None:
            print(ans_tok)
            print(context_tok)

        if _end != None and _end >= len(context_tok):
            _end = len(context_tok) - 1

        return _start, _end

    def load_squad_data(self, data):
        progress = 0
        errors = 0
        start_time = time.time()
        flatened_data = []
        for topics in data['data']:
            title = topics['title']
            for paragraphs in topics['paragraphs']:
                context = paragraphs['context']
                context_tok = ptbtokenizer(context, context=True)
                for qas in paragraphs['qas']:
                    id = qas['id']
                    question = qas['question']
                    for answer in qas['answers']:
                        progress += 1

                        time_delta = datetime.timedelta(seconds=np.round(time.time() - start_time, 0))
                        sys.stdout.write(f'\rCompleted: {progress} | Time: {time_delta}')

                        answer_start = answer['answer_start']
                        answer_end = answer['answer_start'] + len(answer['text'])

                        ans_tok = ptbtokenizer(context[answer_start:answer_end])
                        question_tok = ptbtokenizer(question)

                        _start, _end = self.convert_charidx_to_wordidx(context_tok, ans_tok)
                        if _start == None:
                            errors += 1
                            continue
                        flatened_data.append({'id': id,
                                              'context': context,
                                              'context_ptb_tok': ' '.join(context_tok),
                                              'question': question,
                                              'question_ptb_tok': ' '.join(question_tok),
                                              'answer_ptb_tok': ' '.join(ans_tok),
                                              'start_idx': _start,
                                              'end_idx': _end})

        return flatened_data, errors