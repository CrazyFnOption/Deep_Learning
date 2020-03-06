import torch as t
from torch import nn
from torch.nn import functional as F


class PoetModel(nn.Module):
    def __init__(self, vocab_size, word_vec, hidden_dim):
        super(PoetModel, self).__init__()
        # 之前一直不太理解这里的隐藏纬度是什么意思，后面就知道了用于描述状态的节点数
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, word_vec)
        self.lstm = nn.LSTM(word_vec, self.hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0,c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden
