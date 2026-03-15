import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Online Batch-Hard Triplet Loss.
    SQUARED EUCLIDEAN

    Instead of relying on the dataset to randomly guess a hard negative,
    this loss function computes the distance between every Anchor in the batch
    and every Negative in the batch. For each Anchor, it mathematically selects
    the absolute closest Negative (the Hardest Negative) to compute the loss.
    """
    def __init__(self, margin=0.25, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor   (Tensor): [B, D]
            positive (Tensor): [B, D]
            negative (Tensor): [B, D]
        """
        B = anchor.size(0)

        # 1. Distance between Anchor and its paired Positive
        if self.mode == 'euclidean':
            dist_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1) # Shape: [B]
        else:
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Pairwise Distance between ALL Anchors and ALL Negatives in the batch
        # We compute a [B, B] matrix where element (i, j) is the distance 
        # between Anchor i and Negative j.
        if self.mode == 'euclidean':
            # Efficient pairwise squared euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            dot_product = torch.mm(anchor, negative.t())             # [B, B]
            anchor_norm = torch.sum(anchor ** 2, dim=1, keepdim=True)  # [B, 1]
            negative_norm = torch.sum(negative ** 2, dim=1).unsqueeze(0) # [1, B]
            
            # Clamp to prevent tiny negative numbers due to floating point precision
            dist_matrix = torch.clamp(anchor_norm + negative_norm - 2.0 * dot_product, min=1e-16)
        else:
            # Pairwise Cosine Distance
            anchor_normed = F.normalize(anchor, p=2, dim=1)
            negative_normed = F.normalize(negative, p=2, dim=1)
            cosine_sim_matrix = torch.mm(anchor_normed, negative_normed.t())
            dist_matrix = 1.0 - cosine_sim_matrix

        # 3. HARD NEGATIVE MINING
        # For each Anchor (each row), find the minimum distance to ANY Negative in the batch
        # This is the absolute hardest negative available.
        hardest_dist_neg, _ = torch.min(dist_matrix, dim=1) # Shape: [B]

        # 4. Compute standard Hinge Loss using the hardest negatives
        losses = F.relu(dist_pos - hardest_dist_neg + self.margin)

        # --- Active triplet filtering ---
        active_mask = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            return losses.mean()