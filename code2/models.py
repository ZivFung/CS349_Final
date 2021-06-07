import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class sentimentAnalyze(nn.Module):
    def __init__(self):
        super(sentimentAnalyze, self).__init__()
        self.hidden1 = nn.Linear(2048, 2048)
        self.hidden2 = nn.Linear(2048, 1024)
        self.hidden3 = nn.Linear(1024, 512)
        self.hidden4 = nn.Linear(512, 256)
        self.hidden5 = nn.Linear(256, 128)
        self.BN1     = nn.BatchNorm2d(1024)
        self.out     = nn.Linear(128, 2)
        
    def forward(self, inputs):
        hidden1_out = F.relu(self.hidden1(inputs) + inputs)
        # hidden1_out = self.BN1(hidden1_out)
        hidden1_out = F.dropout(hidden1_out, p = 0.2)
        hidden2_out = F.relu(self.hidden2(hidden1_out))
        hidden2_out = F.dropout(hidden2_out, p = 0.2)
        hidden3_out = F.relu(self.hidden3(hidden2_out))
        hidden3_out = F.dropout(hidden3_out, p = 0.2)
        hidden4_out = F.relu(self.hidden4(hidden3_out))
        hidden4_out = F.dropout(hidden4_out, p = 0.2)
        hidden5_out = F.relu(self.hidden5(hidden4_out))
        output = F.relu(self.out(hidden5_out))
        return output


class sentimentAnalyze1(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 word_vectors,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(sentimentAnalyze1, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)
        self.att_proj=nn.Linear(hidden_size*2,1)
        self.hidden1 = nn.Linear(hidden_size*2, int(hidden_size))
        self.hidden2 = nn.Linear(int(hidden_size), int(hidden_size/2))
        self.out     = nn.Linear(int(hidden_size/2), 2)

    def forward(self, x):
        mask=torch.zeros_like(x)!=x #B * N
        lengths = mask.sum(-1)
        x = self.embed(x)
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training) # B * N * 2H

        #attention aggregation
        att=self.att_proj(x).squeeze() #B * N
        att=self.mask_softmax(att,mask).unsqueeze(-1) #B * N * 1
        x=torch.sum(att*x,dim=-2) #B * 2H

        # print(x.shape)
        hidden1_out = F.relu(self.hidden1(x))
        hidden1_out = F.dropout(hidden1_out, self.drop_prob, self.training)
        hidden2_out = F.relu(self.hidden2(hidden1_out))
        hidden2_out = F.dropout(hidden2_out, self.drop_prob, self.training)
        output = self.out(hidden2_out)
        return F.log_softmax(output,dim=-1)

    def mask_softmax(self,att,mask):
        mask = mask.type(torch.float32)
        masked_logits = mask * att + (1 - mask) * -1e30
        probs = F.softmax(masked_logits, dim=-1)
        return probs
