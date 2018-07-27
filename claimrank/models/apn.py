import torch
import torch.nn.functional as F

from claimrank.utils import masked_softmax


class AttentivePoolingNetwork(torch.nn.Module):
    """Attentive pooling network. As described in:

        https://arxiv.org/abs/1602.03609

    Parameters
    ----------
    vocab_size : int
    embedding_dim : int
    num_filters : int
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
                                                     kernel_size=(3, embedding_dim))
        self._claim_convolution = torch.nn.Conv2d(in_channels=1,
                                                  out_channels=num_filters,
                                                  kernel_size=(3, embedding_dim))
        self._bilinear_fc = torch.nn.Linear(num_filters,
                                            num_filters)
        self._ultimate_fc = torch.nn.Linear(num_filters, embedding_dim)

    def forward(self, sentences, sentence_masks, claims, claim_masks):
        """Computes the forward pass of the baseline model.

        Parameters
        ----------
        sentences : torch.Tensor(shape=(batch_size, sentence_length))
            The sentence to add a post modifier to.
        sentence_masks : torch.Tensor(shape=(batch_size, sentence_length))
            Masks pad values in the ``sentences`` tensor.
        claims : torch.Tensor(shape=(batch_size, num_claims, claim_length))
            The claims to score.
        claim_masks :  torch.Tensor(shape=(batch_size, num_claims, claim_length))
            Masks pad values in the ``claims`` tensor.

        Returns
        -------
        scores : torch.Tensor(shape=(batch_size * num_claims))
            The claim scores.
        encoded : torch.Tensor(shape=(batch_size * num_claims, 2 * embedding_dimembedding_dim))
            The encoded sentence/claim pairs.
        """
        # Get relevant dimensions
        batch_size, sentence_length = sentences.shape
        _batch_size, num_claims, claim_length = claims.shape
        assert batch_size == _batch_size, "Batch sizes do not match"

        # Get outer mask
        outer_mask = claim_masks.sum(-1).ne(0).float()

        # Embed and flatten sentences / sentence_masks, tile to match claims
        embedded_sentences = self._embedding(sentences)
        embedded_sentences.unsqueeze_(1)
        embedded_sentences = embedded_sentences.repeat(1, num_claims, 1, 1)
        embedded_sentences = embedded_sentences.view(-1, sentence_length,
                                                     self._embedding_dim)
        embedded_sentences.unsqueeze_(1) # Dummy 'channels' dim

        sentence_masks.unsqueeze_(1)
        sentence_masks = sentence_masks.repeat(1, num_claims, 1)
        sentence_masks = sentence_masks.view(-1, sentence_length)

        # Embed and flatten claims / claim_masks
        claims = claims.view(-1, claim_length)
        embedded_claims = self._embedding(claims)
        embedded_claims.unsqueeze_(1) # Dummy 'channels' dim

        claim_masks = claim_masks.view(-1, claim_length)

        # Apply convolutions
        convolved_sentences = self._sentence_convolution(embedded_sentences)
        convolved_sentences.squeeze_(-1)
        convolved_claims = self._claim_convolution(embedded_claims)
        convolved_claims.squeeze_(-1)

        # Apply bilinear transformation
        lhs = self._bilinear_fc(convolved_claims.transpose(1,2))
        transformed = torch.bmm(lhs, convolved_sentences)
        transformed = F.tanh(transformed)

        # Pool to get attention over words in sentence
        pooled1, _ = transformed.max(dim=1)
        sentence_attention = masked_softmax(pooled1, sentence_masks[:,:-2])

        # Pool to get attention over claims
        pooled2, _ = transformed.max(dim=2)
        claim_attention = masked_softmax(pooled2, claim_masks[:,:-2])

        # Apply attention
        encoded_sentences = sentence_attention.unsqueeze(1) * convolved_sentences
        encoded_sentences = encoded_sentences.sum(-1)
        encoded_claims = claim_attention.unsqueeze(1) * convolved_claims
        encoded_claims = encoded_claims.sum(-1)

        # Compute scores / outer attention
        scores = F.cosine_similarity(encoded_sentences, encoded_claims)
        scores = scores.view(batch_size, num_claims)
        outer_attention = masked_softmax(scores, outer_mask)

        # Apply attention to claims
        encoded_claims = encoded_claims.view(batch_size, num_claims, -1)
        encoded = outer_attention.unsqueeze(2) * encoded_claims
        encoded = encoded.sum(1)
        encoded = self._ultimate_fc(encoded)

        return scores, encoded

