import torch
import torch.nn.functional as F


class AttentivePoolingNetwork(torch.nn.Module):
    """Attentive pooling network. As described in:

        https://arxiv.org/abs/1602.03609

    Parameters
    ----------
    vocab_size : int
    embedding_dim : int
    num_sentence_filters : int
    num_claim_filters : int
    num_layers : int
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 num_filters):
        super(AttentivePoolingNetwork, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters

        self._embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self._sentence_convolution = torch.nn.Conv2d(in_channels=1,
                                                     out_channels=num_filters, 
                                                     kernel_size=(1, embedding_dim))
        self._claim_convolution = torch.nn.Conv2d(in_channels=1,
                                                  out_channels=num_filters,
                                                  kernel_size=(1, embedding_dim))
        self._bilinear_fc = torch.nn.Linear(num_filters,
                                            num_filters)

    # TODO: Mask
    def forward(self, sentences, claims):
        """Computes the forward pass of the baseline model.

        Parameters
        ----------
        sentences : torch.Tensor(shape=(batch_size, sentence_length))
            The sentence to add a post modifier to.
        claims : torch.Tensor(shape=(batch_size, num_claims, claim_length))
            The claims to score.

        Returns
        -------
        scores : torch.Tensor(shape=(batch_size, num_claims))
            The claim scores.
        """

        # Get relevant dimensions
        batch_size, sentence_length = sentences.shape
        _batch_size, num_claims, claim_length = claims.shape
        assert batch_size == _batch_size, "Batch sizes do not match"

        # Embed sentence and repeat to match claims
        embedded_sentences = self._embedding(sentences)
        embedded_sentences.unsqueeze_(1)
        embedded_sentences = embedded_sentences.repeat(1, num_claims, 1, 1)
        embedded_sentences = embedded_sentences.view(-1, sentence_length,
                                                     self._embedding_dim)
        embedded_sentences.unsqueeze_(1) # Dummy 'channels' dim

        # Embed claims
        claims = claims.view(-1, claim_length)
        embedded_claims = self._embedding(claims)
        embedded_claims.unsqueeze_(1) # Dummy 'channels' dim

        # Apply convolutions
        convolved_sentences = self._sentence_convolution(embedded_sentences)
        convolved_sentences.squeeze_(-1)
        convolved_claims = self._claim_convolution(embedded_claims)
        convolved_claims.squeeze_(-1)

        # Apply bilinear transformation
        lhs = self._bilinear_fc(convolved_claims.transpose(1,2))
        transformed = torch.bmm(lhs, convolved_sentences)

        # Pool to get attention over words in sentence
        pooled1, _ = transformed.max(dim=1)
        sentence_attention = F.softmax(pooled1, dim=-1)

        # Pool to get attention over claims 
        pooled2, _ = transformed.max(dim=2)
        claim_attention = F.softmax(pooled2, dim=-1)

        # Apply attention
        encoded_sentences = sentence_attention.unsqueeze(1) * convolved_sentences
        encoded_sentences = encoded_sentences.sum(-1)
        encoded_claims = claim_attention.unsqueeze(1) * convolved_claims
        encoded_claims = encoded_claims.sum(-1)

        # Compute scores
        scores = torch.einsum('bi,bi->b', (encoded_sentences, encoded_claims))
        scores = scores.view(batch_size, num_claims)

        return scores

