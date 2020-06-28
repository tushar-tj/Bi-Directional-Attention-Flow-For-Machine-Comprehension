import torch
from torchtext.vocab import GloVe
from torchtext import data
from tokenizer import post_ptbtokenizer


class PreprocessData:
    def __init__(self,
                 data_path,
                 glove_size,
                 batch_size,
                 train_file='train.csv',
                 dev_file='dev.csv'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Defining the Fields
        self.RAW = data.RawField(is_target=False)
        self.WORDS = data.Field(batch_first=True,
                                tokenize=post_ptbtokenizer,
                                lower=True,
                                include_lengths=True)
        self.CHAR = data.NestedField(data.Field(batch_first=True,
                                                tokenize=list,
                                                lower=True),
                                     tokenize=post_ptbtokenizer)

        self.INDEX = data.Field(sequential=False,
                                unk_token=None,
                                use_vocab=False)

        fields = {
            'id': ('id', self.RAW),
            'context_ptb_tok': [('context_words', self.WORDS), ('context_char', self.CHAR)],
            'question_ptb_tok': [('question_words', self.WORDS), ('question_char', self.CHAR)],
            'answer_ptb_tok': [('answer_words', self.WORDS), ('answer_char', self.CHAR)],
            'start_idx': ('start_idx', self.INDEX),
            'end_idx': ('end_idx', self.INDEX)
        }

        print('Loading CSV Data Into Torch Tabular Dataset')
        self.train, self.dev = data.TabularDataset.splits(
            path=data_path,
            train=train_file,
            validation=dev_file,
            format='csv',
            fields=fields)

        print('Building Vocabulary')
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORDS.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=glove_size))

        print('Creating Iterators')
        self.train_iter = PreprocessData.create_train_iterator(self.train, device, batch_size)
        self.dev_iter = PreprocessData.create_dev_iterator(self.dev, device, batch_size)

    @staticmethod
    def create_train_iterator(train, device, batch_size):
        train_iter = data.BucketIterator(
            train,
            batch_size=batch_size,
            device=device,
            repeat=False,
            shuffle=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.context_words))

        return train_iter

    @staticmethod
    def create_dev_iterator(dev, device, batch_size):
        dev_iter = data.BucketIterator(
            dev,
            batch_size=batch_size,
            device=device,
            repeat=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.context_words))

        return dev_iter