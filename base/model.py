import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
class textclass(nn.Module):
    def __init__(self, embedding, x_feature, h_state, n_layers, out_h):
        super(textclass, self).__init__()
        self.embedding = embedding
        self.rnn = nn.RNN(x_feature, h_state, n_layers, batch_first = True)
        self.outlayer = nn.Linear(h_state, out_h)

    def forward(self, inputs, lengths, h_state):
        input_embed = self.embedding(inputs)
        input_packed = nn.utils.rnn.pack_padded_sequence(input_embed, lengths, batch_first=True)
        out, h = self.rnn(input_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        
        xlices = torch.arange(0, inputs.size(0)).long()
        cols = lengths - 1
        out = out[xlices, cols, :]
        out = self.outlayer(out)
        Logs = nn.LogSoftmax(dim=1)
        out = Logs(out)
        return out,h


 
        
        
