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
Instance = namedtuple('Instance', ['input_sentence', 'entity_name',
                                   'post_modifier', 'gold_sentence',
                                   'wiki_id', 'previous_sentence',
                                   'blank', 'claims', 'overlap_scores'])


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


def merge(pm, wiki):
    """Merges entries in a .pm and .wiki file into single ``Instance`` tuples.

    Parameters
    ----------
    pm : List[PMInstance]
    wiki : List[WikiInstance]

    Returns
    -------
    A list of ``Instances``.
    """
    out = []
    stopword_set = set(stopwords.words('english'))
    for pm_instance in pm:
        post_modifier = pm_instance.post_modifier
        wiki_instance = wiki[pm_instance.wiki_id]
        claims = wiki_instance.claims
        overlap_scores = [overlap_score(post_modifier, claim, stopword_set) for claim in claims]
        instance = Instance(*pm_instance, claims, overlap_scores)
        out.append(instance)
    return out


# TODO: Fix name conflict for stopwords (affects code below)
def overlap_score(post_modifier, claim, stopword_set):
    """Computes the overlap between a post modifier and a claim.

    Overlap score is computed as:
        intersection(post_modifier, claim) / len(post_modifier)

    Parameters
    ----------
    post_modifier : str
    claim : Claim
    stopword_set : set
        Stopwords to filter from ``post_modifier``computing overlap.

    Returns
    -------
    overlap_score : float
    """
    def _preprocess(string):
        tokens = string.strip().lower().split()
        token_set = set(tokens)
        filtered_token_set = token_set.difference(stopword_set)
        return filtered_token_set
    # Extract set of relevant tokens in post modifier
    post_modifier_set = _preprocess(post_modifier)

    # Extract set of relevant tokens from claim
    claim_set = set()
    claim_set.update(_preprocess(claim.field_name))
    claim_set.update(_preprocess(claim.value))
    for qualifier in claim.qualifiers:
        qualifier_string = ' '.join(qualifier)
        claim_set.update(_preprocess(qualifier_string))

    # Compute and return overlap score
    overlap_score = len(post_modifier_set.intersection(claim_set))/len(post_modifier_set)
    return overlap_score


def pad(input, pad_value=0):
    if isinstance(input[0][0], list):
        outer_max_len = max(len(x) for x in input)
        inner_max_len = max(len(y) for x in input for y in x)
        padded = []
        for outer_seq in input:
            if len(outer_seq) < outer_max_len:
                for _ in range(outer_max_len - len(outer_seq)):
                    outer_seq.append([])
            entry = []
            for inner_seq in outer_seq:
                entry.append(inner_seq + [pad_value]*(inner_max_len - len(inner_seq)))
            padded.append(entry)
    else:
        max_len = max(len(x) for x in input)
        padded = [x + [pad_value]*(max_len - len(x)) for x in input]
    return padded


def batch_collate_fn(batch):
    inputs, claims, scores, pms = zip(*batch)
    inputs = torch.tensor(pad(inputs))
    claims = torch.tensor(pad(claims))
    scores = torch.tensor(pad(scores))
    pms = torch.tensor(pad(pms))
    return inputs, claims, scores, pms


class SimplePMDataset(data.Dataset):
    def __init__(self, prefix, vocab, maxlen=300):
        """A simplified post modifier data loader.

        Parameters
        ----------
        prefix : str
            Path to .pm and .wiki file
        vocab : Dictionary
            Vocabulary for mapping tokens to ids.
        maxlen : int, optional
            Maximum number of tokens.
        """
        self._prefix = prefix
        self._vocab = vocab
        self._maxlen = maxlen

        pm = load_pm(prefix + '.pm')
        wiki = load_wiki(prefix + '.wiki')
        self._data = merge(pm, wiki)

    def __getitem__(self, i):
        instance = self._data[i]
        sentence = self._vocab.map_sentence(instance.input_sentence)
        if len(sentence) > self._maxlen:
            sentence = sentence[:self._maxlen]
        claims = [claim.field_name + ' ' + claim.value for claim in instance.claims]
        claims = [self._vocab.map_sentence(claim) for claim in claims]
        scores = [1 if x > 0 else 0 for x in instance.overlap_scores]
        post_modifier = self._vocab.map_sentence(instance.post_modifier)
        if len(post_modifier) > self._maxlen:
            post_modifier = post_modifier[:self._maxlen]
        return sentence, claims, scores, post_modifier

    def __len__(self):
        return len(self._data)


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

    def map_sentence(self, sentence):
        tokens = sentence.strip().lower().split()
        unk_idx = self.word2idx['<oov>']
        ids = [self.word2idx[x] if x in self.word2idx else unk_idx for x in tokens]
        ids = [self.word2idx['<sos>']] + ids + [self.word2idx['<eos>']]
        return ids


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
    

    return dictionary


