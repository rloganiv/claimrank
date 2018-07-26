'''
Created on Jul 25, 2018

@author: ddua
'''
import torch.nn as nn
import torch
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
        self.decoder = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        
    def init_hidden(self, bsz):
        zeros1 = torch.zeros(1, bsz, self.hidden_dim)
        zeros2 = torch.zeros(1, bsz, self.hidden_dim)
        return  (zeros1, zeros2)
        
    def forward(self, hidden, target):
        bsz, maxlen = target.size()
        
        state = self.init_hidden(bsz)
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)
        
        embeddings = self.embedding(target)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        
        lengths = [len(t.get(0)) for t in target]
        lengths, sorted_indices = torch.sort(torch.Tensor(lengths), 0, descending=True)
        embeddings = torch.index_select(embeddings, 0, sorted_indices)
        lengths = lengths.tolist()
        
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)
   
        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        decoded = self.hidden_to_vocab(output.contiguous().view(-1, self.hidden_dim))
        decoded = decoded.view(bsz, maxlen, self.ntokens)
        
        decoded_sorted = []
        for idx in sorted_indices.data.tolist():
            decoded_sorted.append(decoded[idx].unsqueeze(0))
        decoded_sorted = torch.cat(decoded_sorted,0)

        return decoded_sorted
        
        