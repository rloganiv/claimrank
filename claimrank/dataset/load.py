"""Load serialized data into Python."""

import csv
import json
from collections import namedtuple
import torch
import os
import random
from nltk.corpus import stopwords
import numpy as np
import torch.utils.data as data
import spacy

# TODO: Filter out trailing tabs from .pm file.
PMInstance = namedtuple('PMInstance', ['input_sentence', 'entity_name',
                                       'post_modifier', 'gold_sentence',
                                       'wiki_id', 'previous_sentence',
                                       'blank'])
WikiInstance = namedtuple('WikiInstance', ['wiki_id', 'entity_name', 'aliases',
                                           'description', 'claims'])
Claim = namedtuple('Claim', ['field_name', 'value', 'qualifiers'])

def load_pm(filename):
    """Reads a .pm file into a list of PMInstances.

    Parameters
    ----------
    filename : str
        Path to the .pm file.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        out = [PMInstance(*row) for row in reader]
    return out

def load_wiki(filename):
    """Reads a .wiki file into a dictionary whose keys are ``wiki_id``s and
    whose value are ``WikiInstance``s.

    Parameters
    ----------
    filename : str
        Path to the .wiki file.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        out = dict()
        for line in reader:
            wiki_id = line[0]
            entity_name = line[1]
            aliases = line[2]
            description = line[3]
            unprocessed_claims = json.loads(line[4])
            processed_claims = []
            for unprocessed_claim in unprocessed_claims:
                field_name, value = unprocessed_claim['property']
                qualifiers = unprocessed_claim['qualifiers']
                processed_claim = Claim(field_name, value, qualifiers)
                processed_claims.append(processed_claim)
            out[wiki_id] = WikiInstance(wiki_id, entity_name, aliases,
                                        description, processed_claims)
    return out

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        # prune by most frequently seen words
        vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

class PMDataset(data.Dataset):
    def __init__(self, path, maxlen, vocab):
        self.dictionary = vocab
        self.maxlen = maxlen
        self.maxlen_sent = 0
        self.maxlen_claim = 0
        self.path = path
        self.stopwords = stopwords.words('english')
        self.wiki = load_wiki(path + '.wiki')
        self.pm = load_pm(path + '.pm')
        self.data = self.align_data(self.pm, self.wiki)
        self.maxlen_sent = min(self.maxlen_sent, maxlen)
        
    def get_indices(self, words):
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']
        return [vocab[w] if w in vocab else unk_idx for w in words]
        
    def align_data(self, pm_data, wiki_data):
        data = []
        cnt_no_pos = 0
        neg_pool_claims = set()
        for instance in pm_data:
            sentence = self.get_indices(instance.input_sentence.strip().lower().split())
            post_modifier = self.get_indices(instance.post_modifier.strip().lower().split())
            post_modifier_bag = set(instance.post_modifier.strip().lower().split())
            post_modifier_bag = post_modifier_bag.difference(self.stopwords)
            claim_names = []
            claim_values = []
            overlap_scores = []
            
            for inst in wiki_data[instance.wiki_id].claims:
                claim_names.append(self.get_indices(inst.field_name.strip().lower().split()))
                claim_values.append(self.get_indices(inst.value.strip().lower().split()))
                claim_bag = inst.field_name.strip().lower().split()+inst.value.strip().lower().split()
                for qualifier in inst.qualifiers:
                    for q in qualifier:
                        if 'time' in q:
                            continue
                        else:
                            claim_bag += q.lower().split()
                
                claim_bag = set(claim_bag)
                overlap_scores.append(len(post_modifier_bag.intersection(claim_bag))/float(len(post_modifier_bag)))
                self.maxlen_claim = max(self.maxlen_claim, len(claim_names[-1]))
                neg_pool_claims.add((inst.field_name.strip().lower()+"\t"+inst.value.strip().lower()))
                
            overlap_scores = np.array(overlap_scores)
            sample_claims = sorted(list(zip(claim_names, claim_values, overlap_scores)),key=lambda x:x[2], reverse=True)
            num_pos_claims = min(len(overlap_scores[overlap_scores>0]),1)
#             num_neg_claims = min(5, len(overlap_scores[overlap_scores==0]))
            num_neg_claims = 5
            if (num_pos_claims==0):
                #print("No positive claim found for {0}".format(instance.post_modifier))
                cnt_no_pos+=1
                continue
            positive_claims = sample_claims[:num_pos_claims]
            candidate_indices = list(range(num_pos_claims,len(overlap_scores)))
            if len(candidate_indices)==0:
                negative_claims = []
                for i in range(0,num_neg_claims):
                    ind = random.choice(list(neg_pool_claims))
                    prop, val = ind.split("\t")
                    negative_claims.append((self.get_indices(prop.split()),self.get_indices(val.split()),0))
            else:
                if len(candidate_indices)<num_neg_claims:
                    negative_claims = sample_claims[num_pos_claims:]  
                    for i in range(0,num_neg_claims-len(candidate_indices)):
                        ind = random.choice(list(neg_pool_claims))
                        prop, val = ind.split("\t")
                        negative_claims.append((self.get_indices(prop.split()),self.get_indices(val.split()),0))  
                                                           
                else:
                    negative_claims_ind = np.random.choice(candidate_indices, num_neg_claims)
                    negative_claims = [sample_claims[ind] for ind in negative_claims_ind]
            
            data.append((sentence, post_modifier, positive_claims, negative_claims))
            
            self.maxlen_sent = max(self.maxlen_sent, len(sentence))
        print("Total #instance with no positive claims {0}".format(cnt_no_pos))
        print("Max length for sentences {0}".format(self.maxlen_sent))
        print("Max length for claim {0}".format(self.maxlen_claim))
        return data
    
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)
        
    
    
def make_vocab(path, vocab_size):
    dictionary = Dictionary()
    file_path = os.path.join(path, 'train.pm')
    assert os.path.exists(file_path)
    # Add words to the dictionary
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip().split('\t')
            sentence = line[0]
            pm = line[2]
            words = sentence.split(" ") + pm.split(" ")
                
            for word in words:
                dictionary.add_word(word)
        
        
    file_path = os.path.join(path, 'train.wiki')
    assert os.path.exists(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip().split('\t')
            claims = json.loads(line[-1])
            for claim in claims:
                instance = claim['property']
                words = instance[0].split(" ") + instance[1].split(" ") 
                for word in words:
                    if (not "00:00:00" in word) and (not "+" in word):
                        dictionary.add_word(word) 
        
        # prune the vocabulary
        dictionary.prune_vocab(k=vocab_size)
#  json.dump(open(path+"/vocab.json", 'w'))
        
    return dictionary
        
def batchify(data, bsz, maxlen_sent, maxlen_claim, shuffle=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []
    
    for i in range(nbatch):
        positive_claims = []
        negative_claims = []
        post_modifier = []
        sentences = []
        # Pad batches to maximum sequence length in batch
        for sample in data[i*bsz:(i+1)*bsz]:
            sentences.append(sample[0])
            post_modifier.append(sample[1])
            positive_claims.append(sample[2])
            negative_claims.append(sample[3])
        sentences = [[1]+sentence+[2]+[0]*maxlen_sent for sentence in sentences]
        sentences = [sentence[:maxlen_sent] for sentence in sentences]
        sentences = torch.cat(0,sentences)
        positive_claims = zip(*[[1]+c_name+[2]+[0]*maxlen_claim for (c_name,c_value,c_score) in positive_claims])
        positive_claims = (torch.cat(0,positive_claims[0]), torch.cat(0,positive_claims[1]), torch.cat(0,positive_claims[2]))
        negative_claims = zip(*[[1]+c_name+[2]+[0]*maxlen_claim for (c_name,c_value,c_score) in negative_claims])
        negative_claims = (torch.cat(0,negative_claims[0]), torch.cat(0,negative_claims[1]), torch.cat(0,negative_claims[2]))
        post_modifier = [[1]+pm+[2]+[0]*maxlen_claim for pm in post_modifier]
        batches.append((sentence, post_modifier, positive_claims, negative_claims))

    return batches
