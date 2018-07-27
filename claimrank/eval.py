import argparse
import json
import sys
import torch
from torch.utils.data import DataLoader

from claimrank.dataset import SimplePMDataset, batch_collate_fn, make_vocab
from claimrank.models import AttentivePoolingNetwork, Decoder
from claimrank.dataset.load import Dictionary


args = None


def main(_):
    vocab = make_vocab(args.data_path, 20000)

    encoder = AttentivePoolingNetwork(len(vocab.word2idx),1024,1024)
    encoder_ckpt = torch.load(args.encoder_ckpt)
    encoder.load_state_dict(encoder_ckpt)
    encoder.eval()

    decoder = Decoder(1024,1024,len(vocab.word2idx),0.0)
    decoder_ckpt = torch.load(args.decoder_ckpt)
    decoder.load_state_dict(decoder_ckpt)
    decoder.eval()

    test_dataset = SimplePMDataset(args.data_path + 'test', vocab, 50)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             collate_fn=batch_collate_fn,
                             shuffle=False)

    for sentences, claims, scores, pms, meta in test_loader:
        claim_masks = claims.gt(0).float()
        sentence_masks = sentences.gt(0).float()
        if args.cuda:
            encoder.cuda()
            decoder.cuda()
            claims = claims.cuda()
            claim_masks = claim_masks.cuda()
            sentences = sentences.cuda()
            sentence_masks = sentence_masks.cuda()
            scores = scores.cuda()
            pms = pms.cuda()

        predicted_scores, hidden = encoder(sentences, sentence_masks, claims, claim_masks)

        # DO SOME KIND OF SCORE EVALUATION HERE

        decoded = decoder(hidden, pms)

        for pred in decoded:
            _, word_ids = pred.max(dim=-1)
            print(' '.join([vocab.idx2word[idx] for idx in word_ids.cpu().numpy()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab',type=str)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--encoder_ckpt',type=str)
    parser.add_argument('--decoder_ckpt',type=str)
    parser.add_argument('--cuda', action='store_true')
    args, _ = parser.parse_known_args()

    main(_)

