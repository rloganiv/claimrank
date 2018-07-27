'''
Created on Jul 25, 2018

@author: ddua
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 ntokens,
                 dropout):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ntokens = ntokens
        self.embedding = nn.Embedding(ntokens, input_dim)
        self.hidden_to_vocab = nn.Linear(hidden_dim, ntokens)
        self.decoder = nn.LSTM(input_size=input_dim+hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=1,
                               batch_first=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.hidden_to_vocab.weight.data.uniform_(-initrange, initrange)
        self.hidden_to_vocab.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = torch.zeros(1, bsz, self.hidden_dim)
        zeros2 = torch.zeros(1, bsz, self.hidden_dim)
        if next(self.parameters()).is_cuda:
            zeros1 = zeros1.cuda()
            zeros2 = zeros2.cuda()
        return (zeros1, zeros2)

    def forward(self, hidden, target=None):
        if target is not None:
            batch_size, seq_len = target.size()

            state = self.init_hidden(batch_size)
            all_hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

            embeddings = self.embedding(target)
            augmented_embeddings = torch.cat((embeddings, all_hidden), -1)

            lengths = [int(len(t.gt(0))) for t in target]
            lengths, sorted_indices = torch.sort(torch.tensor(lengths, device=hidden.device),
                                                 descending=True)
            _, unsorted_indices = torch.sort(sorted_indices)
            lengths = lengths.tolist()
            augmented_embeddings = augmented_embeddings[sorted_indices]

            packed = pack_padded_sequence(augmented_embeddings, lengths, batch_first=True)
            rnn_out, _ = self.decoder(packed, state)
            unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True)

            logits = self.hidden_to_vocab(unpacked)
            logp = F.log_softmax(logits, dim=-1)

            logp = logp[unsorted_indices]

        else:
            batch_size = hidden.shape[0]
            hidden = hidden.unsqueeze(1)
            state = self.init_hidden(batch_size)
            logp = torch.zeros(batch_size, 10, self.ntokens,
                               device=hidden.device)
            x = torch.ones(batch_size, 1, dtype=torch.long,
                           device=hidden.device)
            for i in range(10):
                embeddings = self.embedding(x)
                augmented_embeddings = torch.cat((embeddings, hidden), -1)
                packed = pack_padded_sequence(augmented_embeddings, [1], batch_first=True)
                rnn_out, state = self.decoder(packed, state)
                unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True)
                logits = self.hidden_to_vocab(unpacked).squeeze(1)
                logp[:,i,:] = F.log_softmax(logits, dim=-1)
                x = x.detach()

        return logp

