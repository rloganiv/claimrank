import argparse
import json
import logging
import sys
import torch


logger = logging.getLogger(__name__)


def weights_from_glove(vocab, embedding_file, embedding_size):
    logger.debug('Initializing weights')
    weights = torch.FloatTensor(len(vocab), embedding_size).uniform_(-1,1)
    embedding_lookup = dict()
    with open(embedding_file, 'r') as f:
        for i, line in enumerate(f):
            if not i % 1000:
                logger.debug('Reading line: %i', i)
            word, *embedding = line.split()
            embedding = [float(x) for x in embedding]
            embedding_lookup[word] = embedding
    logger.debug('Reading embeddings into tensor')
    for word, idx in vocab.items():
        if word in embedding_lookup:
            weights[idx, :] = torch.tensor(embedding_lookup[word])
    return weights


def main(_):
    with open(FLAGS.vocab, 'r') as f:
        vocab = json.load(f)
    weights = weights_from_glove(vocab,
                                 FLAGS.embedding_file,
                                 FLAGS.embedding_size)
    print(weights)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab', type=str)
    parser.add_argument('embedding_file', type=str)
    parser.add_argument('embedding_size', type=int)
    FLAGS, _ = parser.parse_known_args()

    main(_)

