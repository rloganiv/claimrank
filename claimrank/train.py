'''
Created on Jul 24, 2018

@author: ddua
'''
import argparse
import sys
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import json
import torch.optim as optim

from claimrank.dataset import PMDataset, make_vocab
from claimrank.models import AttentivePoolingNetwork, Decoder

parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--maxlen', type=str, default=50,
                    help='location of the data corpus')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='lmax vocabulary size')
parser.add_argument('--batch_size', type=int, default=16,
                    help='size of mini-batch')
parser.add_argument('--cuda', action='store_true',
                    help='use gpu')
parser.add_argument('--epochs', type=int, default=50,
                    help='total number of epochs')

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
    positive_fields = torch.Tensor([[[1]+c_name+[2]+[0]*(corpus_train.maxlen_claim-len(c_name)) for (c_name,_,_) in pc] for pc in positive_claims])
    positive_scores = torch.Tensor([[c_score for (_,_,c_score) in pc] for pc in positive_claims])
    positive_scores.fill_(1)

#     print([[len([1]+c_name+[2]+[0]*(corpus_train.maxlen_claim-len(c_name))) for (c_name,_,_) in pc] for pc in negative_claims])
    negative_fields = torch.Tensor([[[1]+c_name+[2]+[0]*(corpus_train.maxlen_claim-len(c_name)) for (c_name,_,_) in pc] for pc in negative_claims])

#     negative_scores = torch.Tensor([[c_score for (_,_,c_score) in pc] for pc in negative_claims])
    negative_scores = positive_scores.clone().fill_(-1)
    
    post_modifier = [[1]+pm+[2]+[0]*corpus_train.maxlen_claim for pm in post_modifier]
    post_modifier = [pm[:corpus_train.maxlen_claim] for pm in post_modifier]
    post_modifier = torch.Tensor(post_modifier)
    
    return (sentences, post_modifier, (positive_fields,positive_scores), (negative_fields,negative_scores))

train_data = torch.utils.data.DataLoader(corpus_train,  batch_size = args.batch_size,collate_fn=collate_pm, shuffle=True)
train_iter = iter(train_data)
test_data = torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_pm, shuffle=False)
test_iter = iter(test_data)
# criterion = nn.MarginRankingLoss()
criterion = nn.NLLLoss(ignore_index=0, size_average=False)

encoder = AttentivePoolingNetwork(len(vocab.word2idx),1024,1024)
decoder = Decoder(1024,1024,len(vocab.word2idx),0.0)
# wgts = torch.load('pretrained.pt')
# wgts[0].fill_(0)
# encoder._embedding.from_pretrained(wgts)

encoder_optimizer = optim.Adam(encoder.parameters(),lr=3e-6,betas=(0.9, 0.999))
decoder_optimizer = optim.Adam(decoder.parameters(),lr=3e-6,betas=(0.9, 0.999))

total_loss = 0
cnt = 0
for ep in range(0,args.epochs):
    encoder.train()
    for batch in train_data:
        sentences, post_modifier, (positive_fields,positive_scores), (negative_fields,negative_scores) = batch
        claims = torch.cat([positive_fields,negative_fields],1)
        sentences = sentences.long()
        claims = claims.long()
        sentences_mask = sentences.gt(0).float()
        claims_mask = claims.gt(0).float()
        post_modifier = post_modifier.long()
        target = torch.Tensor(sentences.size(0),5).fill_(1)
        if args.cuda:
            sentences = sentences.cuda()
            sentences_mask = sentences_mask.cuda()
            claims = claims.cuda()
            claims_mask = claims_mask.cuda()
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            post_modifier = post_modifier.cuda()
            target = target.cuda()

        scores, hidden = encoder(sentences, sentences_mask, claims, claims_mask)
        scores_pos = scores[:,0].unsqueeze(1).repeat(1,5)
        scores_neg = scores[:,1:]

        decoded = decoder(hidden, post_modifier[:,:-1])

        if cnt % 2:
            loss = F.margin_ranking_loss(scores_pos, scores_neg, target)
        else:
            loss = criterion(decoded.view(-1, decoder.ntokens),
                             post_modifier[:,1:].contiguous().view(-1))
            total_loss += loss.data[0]
        cnt+=1
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if cnt%100==0:
            print("[{0}/{1}] Average running Loss: {2}".format(cnt,ep,2*total_loss/float(cnt)))

    print("==Train decodings==")
    for pred in decoded:
        _, word_ids = pred.max(dim=-1)
        print(' '.join([vocab.idx2word[idx] for idx in word_ids.cpu().numpy()]))
    encoder.eval()
    test_loss = 0
    for batch in test_data:
        sentences, post_modifier, (positive_fields, positive_scores), (negative_fields, negative_scores) = batch
        claims = torch.cat([positive_fields,negative_fields],1)
        sentences = sentences.long()
        claims = claims.long()
        sentences_mask = sentences.gt(0).float()
        claims_mask = claims.gt(0).float()
        post_modifier = post_modifier.long()
        if args.cuda:
            sentences = sentences.cuda()
            sentences_mask = sentences_mask.cuda()
            claims = claims.cuda()
            claims_mask = claims_mask.cuda()
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            post_modifier = post_modifier.cuda()

        scores, hidden = encoder(sentences, sentences_mask, claims, claims_mask)
        scores_pos = scores[:,0].unsqueeze(1).repeat(1,5)
        scores_neg = scores[:,1:]

        decoded = decoder(hidden, post_modifier[:,:-1])

        # loss = criterion(scores_pos,scores_neg,target)
        loss = criterion(decoded.view(-1, decoder.ntokens),
                         post_modifier[:,1:].contiguous().view(-1))
        test_loss += loss.data[0]

    print("==Test decodings==")
    for pred in decoded:
        _, word_ids = pred.max(dim=-1)
        print(' '.join([vocab.idx2word[idx] for idx in word_ids.cpu().numpy()]))
    print('Predicted scores')
    print(scores)
    print("Test loss {0}".format(test_loss))
    torch.save(encoder.state_dict(), open("./trained_models/encoder"+str(ep)+".pt", 'wb'))
    torch.save(decoder.state_dict(), open("./trained_models/decoder"+str(ep)+".pt", 'wb'))
    with open('./vocab.json','w',encoding='utf-8') as f:
        json.dump(vocab.word2idx, f)

