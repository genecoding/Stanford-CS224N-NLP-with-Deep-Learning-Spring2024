# Implementation of MultipleNegativesRankingLoss, MultipleNegativesSymmetricRankingLoss, 
# MultipleNegativesRankingLoss with Hard Negatives (including symmetric), CoSENTLoss and AnglELoss.
# Inspired by
#     https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/MultipleNegativesRankingLoss.py
#     https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/MultipleNegativesSymmetricRankingLoss.py
#     https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/CoSENTLoss.py
#     https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/util/similarity.py#L197-L232


import torch
from torch import nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale=20.0, use_symmetric=False):
        """
        scale: inverse of temperature
        use_symmetric: whether use symmetric MNRL
        """
        super().__init__()
        self.scale = scale
        self.use_symmetric = use_symmetric
    
    def forward(self, embeds_a, embeds_b):
        """
        MNRL only takes positive pairs (label = 1). 
        MNRL works by taking a batch of N positive pairs (ai, pi) and treating 
        all N-1 other positives in the same batch as in-batch negatives for each anchor.
        embeds_a: (batch_size, hidden_dim) first sentences in positive pairs
        embeds_b: (batch_size, hidden_dim) second sentences in positive pairs
        """
        # compute cosine similarity between all pairs in the batch
        a_norm = F.normalize(embeds_a, p=2, dim=1)
        b_norm = F.normalize(embeds_b, p=2, dim=1)
        scores = (a_norm @ b_norm.T) * self.scale  # (batch, dim) @ (dim, batch) -> (batch, batch)
        
        # only the diagonal elements [i, i] are the true positive pairs
        labels = torch.arange(len(embeds_a), device=embeds_a.device)
        
        if self.use_symmetric:
            # symmetric MNRL: not only compute losses of anchors to positives, but also positives to anchors
            return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
        else:
            # use cross entropy to maximize the diagonal and minimize the rest
            return F.cross_entropy(scores, labels)


class MNRL_HardNeg(nn.Module):
    """MultipleNegativesRankingLoss with Hard Negatives"""
    def __init__(self, scale=20.0, use_symmetric=False):
        """
        scale: inverse of temperature
        use_symmetric: whether use symmetric MNRL with hard negatives
        """
        super().__init__()
        self.scale = scale
        self.use_symmetric = use_symmetric
        
    def forward(self, emb_anchor, emb_pos, emb_hard_neg):
        """
        MNRL_HardNeg works on triplets (ai, pi, ni).
        emb_anchor: (batch_size, hidden_dim)
        emb_pos: (batch_size, hidden_dim)
        emb_hard_neg: (batch_size, hidden_dim)
        scale: inverse of temperature
        """
        a_norm = F.normalize(emb_anchor, p=2, dim=1)
        p_norm = F.normalize(emb_pos, p=2, dim=1)
        n_norm = F.normalize(emb_hard_neg, p=2, dim=1)
        
        labels = torch.arange(len(emb_anchor), device=emb_anchor.device)
        
        # archor to [positive, hard_negative], (batch, 2*batch)
        scores_a2p = (a_norm @ torch.cat([p_norm, n_norm], dim=0).T) * self.scale
        
        if self.use_symmetric:
            # positive to [anchor, hard_negative], (batch, 2*batch)
            scores_p2a = (p_norm @ torch.cat([a_norm, n_norm], dim=0).T) * self.scale
            return (F.cross_entropy(scores_a2p, labels) + F.cross_entropy(scores_p2a, labels)) / 2
        else:
            return F.cross_entropy(scores_a2p, labels)


class CoSENTLoss(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
        
    def get_similarity(self, embeds_1, embeds_2):
        return F.cosine_similarity(embeds_1, embeds_2)
        
    def forward(self, embeds_1, embeds_2, labels):
        # get score for each pair in the batch
        scores = self.get_similarity(embeds_1, embeds_2)  # (batch)
        # scores[i, j] = scores[j] - scores[i]
        scores = scores[None, :] - scores[:, None]  # (batch, batch)
        
        # we only care about labels[i] > labels[j] where we want scores[i] > scores[j]
        mask = labels[:, None] > labels[None, :]
        scores = scores[mask] * self.scale  # (# of labels[i] > labels[j])
        
        # e^0 = 1
        return torch.logsumexp(torch.cat((torch.zeros(1).to(scores.device), scores)), dim=0)


class AnglELoss(CoSENTLoss):
    def get_similarity(self, embeds_1, embeds_2):
        assert embeds_1.size(1) % 2 == 0 and embeds_2.size(1) % 2 == 0
        
        # split into real and imaginary components
        a, b = torch.chunk(embeds_1, 2, dim=1)
        c, d = torch.chunk(embeds_2, 2, dim=1)
        
        # L2 norm of a complex vector: sqrt(sum(a^2 + b^2))
        # clamp for avoiding division by zero and gradient explosion
        z_norm = torch.sqrt((a**2 + b**2).sum(dim=1).clamp(min=1e-8))  # (batch)
        w_norm = torch.sqrt((c**2 + d**2).sum(dim=1).clamp(min=1e-8))  # (batch)
        
        # inner product of complex vectors: (a + bi)(c - di) = (ac + bd) + (bc - ad)i
        real = (a * c + b * d).sum(dim=1) / (z_norm * w_norm)  # (batch)
        imag = (b * c - a * d).sum(dim=1) / (z_norm * w_norm)  # (batch)
        
        # ensure positive angle difference
        return torch.abs(real + imag)  # (batch)

