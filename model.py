import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BidaF(nn.Module):
    def __init__(self,
                 WORDS,
                 CHAR,
                 char_embedding_size,  # each character is represented by
                 char_conv_kernel_size,  # width of convolution layer kernel
                 char_conv_channels_count,  # no. of convolution layer channels
                 dropout=0.2,
                 ):  #
        super(BidaF, self).__init__()
        self.embedding_size = WORDS.vocab.vectors.shape[1] + char_conv_channels_count

        # 1. Word Embeddings
        self.word_embedding_layer = nn.Embedding.from_pretrained(WORDS.vocab.vectors, freeze=True)

        # 2. Char Embeddings
        self.char_embedding_layer = nn.Embedding(len(CHAR.vocab), char_embedding_size)
        self.convolution_layer = nn.Conv2d(1, char_conv_channels_count, (char_conv_kernel_size, char_embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer.weight)
        self.dropout = nn.Dropout(p=dropout)

        # 3. Highway Network
        self.highway_linear1 = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size), nn.ReLU())
        self.highway_gate1 = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size), nn.Sigmoid())

        self.highway_linear2 = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size), nn.ReLU())
        self.highway_gate2 = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size), nn.Sigmoid())

        # 4. Embedding LSTM Layer
        self.embed_LSTM = nn.LSTM(input_size=self.embedding_size,
                                  hidden_size=self.embedding_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)
        nn.init.xavier_normal_(self.embed_LSTM.weight_hh_l0)
        nn.init.xavier_normal_(self.embed_LSTM.weight_ih_l0)

        # 5. Attention Flow Layer
        self.w = torch.empty((6 * self.embedding_size, 1), device=device)
        nn.init.xavier_normal_(self.w)

        # 6. Modelling Layer
        self.modelling_LSTM1 = nn.LSTM(input_size=8 * self.embedding_size,
                                       hidden_size=self.embedding_size,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=True)
        nn.init.xavier_normal_(self.modelling_LSTM1.weight_hh_l0)
        nn.init.xavier_normal_(self.modelling_LSTM1.weight_ih_l0)

        self.modelling_LSTM2 = nn.LSTM(input_size=2 * self.embedding_size,
                                       hidden_size=self.embedding_size,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=True)
        nn.init.xavier_normal_(self.modelling_LSTM2.weight_hh_l0)
        nn.init.xavier_normal_(self.modelling_LSTM2.weight_ih_l0)

        # 7. Output Layer
        self.p1_w = nn.Linear(10 * self.embedding_size, 1)
        nn.init.xavier_normal_(self.p1_w.weight)

        self.p2_w = nn.Linear(10 * self.embedding_size, 1)
        nn.init.xavier_normal_(self.p2_w.weight)

    def word_embeddings(self, x):
        x = self.word_embedding_layer(x)
        return x

    def character_embeddings(self, x):
        x = self.char_embedding_layer(x)  # batch X words X char X char_emb
        _s = x.shape

        # Reshaping vector as we wish to run the character level embedding over each word in the batch
        x = x.view(_s[0] * _s[1], 1, _s[2], _s[3])
        x = self.convolution_layer(x)
        x = nn.functional.relu(x).squeeze(3)
        x = nn.functional.max_pool1d(x, x.shape[2]).squeeze(2)
        x = self.dropout(x)

        # Shaping back to original batch_size, max_count_of_word_in_batch, char_conv_channels_count
        x = x.view(_s[0], _s[1], -1)

        return x

    def highway_network(self, x1, x2):
        x = torch.cat([x1, x2], -1)

        x = self.highway_linear1(x)
        t1 = self.highway_gate1(x)

        x = t1 * x + (1 - t1) * x

        x = self.highway_linear2(x)
        t2 = self.highway_gate2(x)

        x = t2 * x + (1 - t2) * x

        return x

    def lstm_embedding(self, x, x_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.embed_LSTM(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        return x

    def attention_flow_layer(self, c, q, c_len, q_len, batch_size):
        # Calculate the similarity matrix
        S = torch.empty((batch_size, c_len, q_len), device=device)
        for t in range(c_len):
            for j in range(q_len):
                _x = torch.cat([c[:, t, :], q[:, j, :], c[:, t, :] * q[:, j, :]], dim=-1)
                S[:, t, j] = torch.mm(_x, self.w).squeeze(1)

        # Calculating Context to Query Attention Matrix
        c2q_attn = torch.empty((batch_size, c_len, 2 * self.embedding_size), device=device)
        A = F.softmax(S, dim=1)
        for t in range(c_len):
            c2q_attn[:, t, :] = (S[:, t, :].unsqueeze(2) * q).sum(1)

        # Calculating Query to Context Attention Matrix
        z, _ = torch.max(S, dim=2)
        b = F.softmax(z, dim=1).unsqueeze(2)
        _h = (b * c2q_attn).sum(1).unsqueeze(1)
        q2c_attn = _h.repeat(1, c_len, 1)

        # Merging all Information
        G = torch.cat([c, c2q_attn, c * c2q_attn, c * q2c_attn], dim=-1)

        return G

    def modelling_layer(self, x):
        m1, _ = self.modelling_LSTM1(x)
        m1 = self.dropout(m1)

        m2, _ = self.modelling_LSTM2(m1)
        m2 = self.dropout(m2)

        return m1, m2

    def output_layer(self, G, m1, m2):

        p1 = self.p1_w(torch.cat([G, m1], dim=-1)).squeeze(2)
        p2 = self.p2_w(torch.cat([G, m2], dim=-1)).squeeze(2)
        return p1, p2

    def mask_output(self, p1, p2, data):
        # This function replaced the pads with zeros to remove them from loss calculations
        max_size = p1.shape[1]
        mask = torch.tensor([[1.0] * lengths.item() + [0.0] * (max_size - lengths.item())
                             for lengths in data.context_words[1]])

        return mask * p1, mask * p2

    def forward(self, data):
        context_word_emb = self.word_embeddings(data.context_words[0])
        ques_word_emb = self.word_embeddings(data.question_words[0])

        context_char_emb = self.character_embeddings(data.context_char)
        ques_char_emb = self.character_embeddings(data.question_char)

        context_emb = self.highway_network(context_word_emb, context_char_emb)
        ques_emb = self.highway_network(ques_word_emb, ques_char_emb)

        context_emb = self.lstm_embedding(context_emb, data.context_words[1])
        ques_emb = self.lstm_embedding(ques_emb, data.question_words[1])

        G = self.attention_flow_layer(context_emb, ques_emb,
                                      data.context_words[1].max().item(), data.question_words[1].max().item(),
                                      data.batch_size)

        m1, m2 = self.modelling_layer(G)
        p1, p2 = self.output_layer(G, m1, m2)

        # masking output before softmax to remove pads from output calculations
        # p1, p2 = self.mask_output(p1, p2, data)

        p1 = F.softmax(p1, dim=-1)
        p2 = F.softmax(p2, dim=-1)

        return p1, p2