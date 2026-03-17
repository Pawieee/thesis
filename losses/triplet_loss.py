import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Offline Element-Wise Triplet Loss (Squared Euclidean Distance).

    Computes the standard hinge-based triplet loss on pre-formed
    (Anchor, Positive, Negative) triplets provided by SplitTripletDataset's
    offline hard negative miner. Each triplet is evaluated element-wise
    — no cross-batch comparison is performed, completely preventing
    class collisions between unrelated pairs.

    Distance metric:
        Euclidean mode (default): Squared Euclidean Distance (SED).
            SED(a, b) = ||a - b||²
            On L2-normalized embeddings (unit hypersphere): SED ∈ [0, 4].
            SED = 2 - 2·cos(a, b), making it monotonically equivalent
            to cosine distance on the unit sphere.

        Cosine mode: Angular distance derived from cosine similarity.
            d(a, b) = 1 - cos(a, b), range [0, 2].

    Loss formulation:
        L = max(0, d(a, p) - d(a, n) + margin)

        where d(a, p) is the anchor-positive distance and d(a, n) is
        the anchor-negative distance. Only active triplets (loss > 0)
        contribute to the gradient — satisfied triplets (loss = 0) are
        excluded from the mean to prevent gradient dilution.

    Margin:
        Controls the minimum required separation between positive and
        negative distances. On L2-normalized embeddings with SED,
        the recommended margin is 1.0, which corresponds to a meaningful
        angular separation on the unit hypersphere.

    Active triplet tracking:
        last_fraction_active stores the fraction of triplets in the
        most recent forward pass that were active (loss > 0). This
        is used during training to monitor learning progress:
            ~100%  → model has not yet learned to separate embeddings
            ~30-60% → healthy learning signal
            ~0%    → embedding space has collapsed or margin is too small

    Args:
        margin (float): Hinge loss margin. Default 1.0.
                        Recommended range for SED on unit sphere: [0.5, 2.0].
        mode   (str)  : Distance metric. 'euclidean' (default) or 'cosine'.

    Inputs:
        anchor   (Tensor): [B, D] — L2-normalized anchor embeddings.
        positive (Tensor): [B, D] — L2-normalized positive embeddings
                           (same writer as anchor).
        negative (Tensor): [B, D] — L2-normalized negative embeddings
                           (different writer from anchor).

    Returns:
        Tensor: Scalar mean loss over active triplets. If no active
                triplets exist, returns the mean over all triplets
                to maintain gradient flow.
    """
    def __init__(self, margin=1.0, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        # 1. Anchor-Positive distance
        if self.mode == 'euclidean':
            # Squared Euclidean Distance: ||a - p||² ∈ [0, 4] on unit sphere
            dist_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        else:
            # Angular distance: 1 - cos(a, p) ∈ [0, 2]
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Anchor-Negative distance
        if self.mode == 'euclidean':
            # Squared Euclidean Distance: ||a - n||² ∈ [0, 4] on unit sphere
            dist_neg = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        else:
            # Angular distance: 1 - cos(a, n) ∈ [0, 2]
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative)

        # 3. Hinge loss: penalises triplets where the positive is not
        # sufficiently closer to the anchor than the negative by margin
        losses = F.relu(dist_pos - dist_neg + self.margin)

        # 4. Active triplet filtering
        # Only triplets with loss > 0 (violated margin) contribute to the
        # gradient. Averaging over satisfied triplets (loss = 0) would
        # dilute the gradient signal and slow convergence.
        active_mask     = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            # All triplets satisfied — return mean over all to preserve
            # gradient flow and avoid zero-loss instability
            return losses.mean()