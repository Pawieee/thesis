import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Online Batch Hard Triplet Loss (Squared Euclidean, L2-normalized embeddings).

    Given a flat batch of embeddings and their writer-ID labels, mines the
    hardest valid triplet for every anchor in the batch:

        Hardest positive: the genuine sample of the SAME writer that is
                          FURTHEST from the anchor in embedding space.
        Hardest negative: the sample of a DIFFERENT writer that is
                          CLOSEST to the anchor in embedding space.

    Only triplets where anchor and positive belong to the same class AND
    at least one other class exists in the batch are considered valid.

    This is the "batch hard" strategy from Hermans et al.,
    "In Defense of the Triplet Loss for Person Re-Identification" (2017).

    Operating on the unit hypersphere (L2-normalized embeddings):
        SED range: [0, 4]
        Recommended margin: 0.5 – 1.0

    Args:
        margin (float): Hinge loss margin. Default 1.0.

    Inputs:
        embeddings (Tensor): [B, D] — L2-normalized embedding vectors.
        labels     (Tensor): [B]    — integer writer ID per sample.

    Returns:
        Tensor: Scalar mean loss over all valid anchors.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.last_fraction_active = 0.0

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        device = embeddings.device

        # ── Pairwise Squared Euclidean Distance matrix ────────────────────────
        # ||a - b||² = ||a||² + ||b||² - 2·aᵀb
        # On the unit sphere: ||a||² = ||b||² = 1, so SED = 2 - 2·aᵀb
        # torch.cdist is equivalent but the dot-product form is faster with AMP.
        dot   = torch.mm(embeddings, embeddings.t())          # [B, B]
        sq    = dot.diagonal().unsqueeze(1)                    # [B, 1]
        dist  = (sq + sq.t() - 2.0 * dot).clamp(min=0.0)     # [B, B], SED

        # ── Boolean masks ─────────────────────────────────────────────────────
        labels_row = labels.unsqueeze(1)   # [B, 1]
        labels_col = labels.unsqueeze(0)   # [1, B]

        same_class = labels_row == labels_col          # [B, B]  True = same writer
        diff_class = ~same_class                       # [B, B]  True = different writer
        eye        = torch.eye(B, dtype=torch.bool, device=device)

        # Valid positives: same class, not self
        pos_mask = same_class & ~eye                   # [B, B]
        # Valid negatives: different class
        neg_mask = diff_class                          # [B, B]

        # ── Per-anchor validity check ─────────────────────────────────────────
        # An anchor is only valid if it has at least one positive AND one
        # negative in the batch. Writers with only one sample in the batch
        # have no valid positive and must be skipped.
        has_pos = pos_mask.any(dim=1)   # [B]
        has_neg = neg_mask.any(dim=1)   # [B]
        valid   = has_pos & has_neg     # [B]

        if valid.sum() == 0:
            # Edge case: no valid anchors in this batch (e.g. batch too small)
            self.last_fraction_active = 0.0
            return embeddings.sum() * 0.0  # zero loss, keeps graph connected

        # ── Batch hard mining ─────────────────────────────────────────────────
        # For invalid positions, fill with sentinel values so max/min
        # do not accidentally select them.
        NEG_INF = torch.finfo(dist.dtype).min
        POS_INF = torch.finfo(dist.dtype).max

        # Hardest positive: max distance among same-class pairs
        dist_pos_all = dist.masked_fill(~pos_mask, NEG_INF)   # [B, B]
        hardest_pos  = dist_pos_all.max(dim=1).values          # [B]

        # Hardest negative: min distance among different-class pairs
        dist_neg_all = dist.masked_fill(~neg_mask, POS_INF)   # [B, B]
        hardest_neg  = dist_neg_all.min(dim=1).values          # [B]

        # ── Triplet hinge loss ────────────────────────────────────────────────
        losses = F.relu(hardest_pos - hardest_neg + self.margin)  # [B]

        # Restrict to valid anchors only
        losses = losses[valid]   # [n_valid]

        # ── Active triplet tracking ───────────────────────────────────────────
        active_mask = losses > 0
        self.last_fraction_active = active_mask.float().mean().item()

        if active_mask.sum() > 0:
            return losses[active_mask].mean()
        else:
            return losses.mean()