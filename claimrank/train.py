'''
Created on Jul 24, 2018

@author: ddua
'''

import argparse
import sys
import torch.utils.data

sys.path.insert(0, './dataset')
from load import PMDataset, make_vocab

parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--maxlen', type=str, default=20,
                    help='location of the data corpus')
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32,
                    help='location of the data corpus')

args = parser.parse_args()

vocab = make_vocab(args.data_path, args.vocab_size)
# create corpus
corpus_train = PMDataset(vocab=vocab, maxlen=args.maxlen, path=args.data_path+"/train")
corpus_test = PMDataset(vocab=vocab, maxlen=args.maxlen, path=args.data_path+"/valid")

def collate_pm(batch):
    positive_claims = []
    negative_claims = []
    post_modifier = []
    sentences = []
    # Pad batches to maximum sequence length in batch
    for sample in batch:
        sentences.append(sample[0])
        post_modifier.append(sample[1])
        positive_claims.append(sample[2])
        negative_claims.append(sample[3])
    sentences = [[1]+sentence+[2]+[0]*corpus_train.maxlen_sent for sentence in sentences]
    sentences = [sentence[:corpus_train.maxlen_sent] for sentence in sentences]
    sentences = torch.Tensor(sentences)
    positive_claims = zip(*[[[1]+c_name+[2]+[0]*corpus_train.maxlen_claim for (c_name,c_value,c_score) in pc] for pc in positive_claims])
    positive_claims = (torch.cat(0,positive_claims[0]), torch.cat(0,positive_claims[1]), torch.cat(0,positive_claims[2]))
    negative_claims = zip(*[[1]+c_name+[2]+[0]*corpus_train.maxlen_claim for (c_name,c_value,c_score) in negative_claims])
    negative_claims = (torch.cat(0,negative_claims[0]), torch.cat(0,negative_claims[1]), torch.cat(0,negative_claims[2]))
    post_modifier = [[1]+pm+[2]+[0]*corpus_train.maxlen_claim for pm in post_modifier]
    
    return (sentences, post_modifier, positive_claims, negative_claims)

train_data = torch.utils.data.DataLoader(corpus_train,  batch_size = args.batch_size,collate_fn=collate_pm, shuffle=True)
train_iter = iter(train_data)
test_data = torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_pm, shuffle=False)
test_iter = iter(test_data)
for batch in train_data:
    print(batch)

