import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.bilstm = nn.LSTM(
                input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)
        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        """
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        """
        orderd_lens, index = lens.sort(descending=True)
        ordered_x = x[index]

        packed_x = nn.utils.rnn.pack_padded_sequence(
            ordered_x, orderd_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]
        return recover_output


class ESIM(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_size, hidden_size, dropout_rate=0.1, layer_num=1, \
                pretrained_embed=None, freeze=False):
        super(ESIM, self).__init__()
        self.pretrained_embed = pretrained_embed
        if pretrained_embed is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.bilstm1 = BiLSTM(embed_size, hidden_size, dropout_rate, layer_num)
        self.bilstm2 = BiLSTM(hidden_size, hidden_size, dropout_rate, layer_num)
        self.fc1 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)


        self.init_weights()

    def init_weights(self):
        if self.pretrained_embed is None:
            nn.init.normal_(self.embed.weight)
            self.embed.weight.data.mul_(0.01)
        nn.init.normal_(self.fc1.weight)
        self.fc1.weight.data.mul_(0.01)
        nn.init.normal_(self.fc2.weight)
        self.fc2.weight.data.mul_(0.01)
        nn.init.normal_(self.fc3.weight)
        self.fc3.weight.data.mul_(0.01)
        
    def soft_align_attention(self, x1, x1_lens, x2, x2_lens):
