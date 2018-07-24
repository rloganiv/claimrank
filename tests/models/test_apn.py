import torch

from claimrank.models import AttentivePoolingNetwork


def test_forward():
    # Initialize network
    vocab_size = 32
    embedding_dim = 2
    num_filters = 6
    apn = AttentivePoolingNetwork(vocab_size,
                                  embedding_dim,
                                  num_filters)

    # Define test data
    batch_size = 7
    sentence_length = 12
    num_claims = 11
    claim_length = 5

    sentences = torch.randint(vocab_size, (batch_size, sentence_length),
                              dtype=torch.long)
    claims = torch.randint(vocab_size, (batch_size, num_claims, claim_length),
                           dtype=torch.long)

    # Compute scores
    scores = apn(sentences, claims)

    # Test is passed if no errors were raised :)

