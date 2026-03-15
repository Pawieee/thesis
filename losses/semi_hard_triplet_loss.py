import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Online Batch Semi-Hard Triplet Loss.
    SQUARED EUCLIDEAN

    Computes pairwise distances. For each Anchor, it selects a Negative that is
    further away than the Positive, but still within the margin:
    d(a, p) < d(a, n) < d(a, p) + margin.
    If no such negative exists in the batch, it defaults to the hardest negative.
    """
    def __init__(self, margin=0.25, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        B = anchor.size(0)

        # 1. Distance between Anchor and its paired Positive
        if self.mode == 'euclidean':
            dist_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1) # Shape: [B]
        else:
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Pairwise Distance between ALL Anchors and ALL Negatives
        if self.mode == 'euclidean':
            dot_product = torch.mm(anchor, negative.t())
            anchor_norm = torch.sum(anchor ** 2, dim=1, keepdim=True)
            negative_norm = torch.sum(negative ** 2, dim=1).unsqueeze(0)
            dist_matrix = torch.clamp(anchor_norm + negative_norm - 2.0 * dot_product, min=1e-16)
        else:
            anchor_normed = F.normalize(anchor, p=2, dim=1)
            negative_normed = F.normalize(negative, p=2, dim=1)
            cosine_sim_matrix = torch.mm(anchor_normed, negative_normed.t())
            dist_matrix = 1.0 - cosine_sim_matrix

        # 3. SEMI-HARD NEGATIVE MINING
        # We want negatives where: dist_pos < dist_neg < dist_pos + margin
        
        # Create masks
        dist_pos_expanded = dist_pos.unsqueeze(1).expand_as(dist_matrix) # [B, B]
        
        # Mask 1: Negative must be further than the Positive
        is_harder_than_pos = dist_matrix > dist_pos_expanded
        
        # Mask 2: Negative must be within the margin
        is_violating_margin = dist_matrix < (dist_pos_expanded + self.margin)
        
        # Combine masks to find valid Semi-Hard negatives
        semi_hard_mask = is_harder_than_pos & is_violating_margin
        
        # Select the negative
        selected_dist_neg = torch.zeros(B, device=anchor.device)
        
        for i in range(B):
            valid_indices = torch.nonzero(semi_hard_mask[i]).squeeze(1)
            if valid_indices.numel() > 0:
                # If semi-hard negatives exist, pick the hardest one among them
                valid_dists = dist_matrix[i, valid_indices]
                selected_dist_neg[i] = torch.min(valid_dists)
            else:
                # Fallback: If no semi-hard exists, pick the absolute hardest
                selected_dist_neg[i] = torch.min(dist_matrix[i])

        # 4. Compute standard Hinge Loss using the selected negatives
        losses = F.relu(dist_pos - selected_dist_neg + self.margin)

        # --- Active triplet filtering ---
        active_mask = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            return losses.mean()