import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Standard Element-wise Triplet Loss.
    
    Computes the loss strictly between the corresponding (anchor, positive, negative)
    pairs provided by the DataLoader. It does not perform batch-wide mining, 
    preventing class collision when explicit triplets are passed.
    """
    def __init__(self, margin=0.25, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()
        self.last_fraction_active = 0.0

    def forward(self, anchor, positive, negative):
        # 1. Distance between Anchor and Positive (Element-wise)
        if self.mode == 'euclidean':
            dist_pos_sq = torch.sum(torch.pow(anchor - positive, 2), dim=1) 
            dist_pos = torch.sqrt(torch.clamp(dist_pos_sq, min=1e-16)) 
        else:
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive)

        # 2. Distance between Anchor and its explicit Negative (Element-wise)
        if self.mode == 'euclidean':
            dist_neg_sq = torch.sum(torch.pow(anchor - negative, 2), dim=1)
            dist_neg = torch.sqrt(torch.clamp(dist_neg_sq, min=1e-16))
        else:
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative)

        # 3. Compute standard Hinge Loss
        losses = F.relu(dist_pos - dist_neg + self.margin)

        active_mask = losses > 0
        active_triplets = losses[active_mask]

        self.last_fraction_active = active_mask.float().mean().item()

        if active_triplets.numel() > 0:
            return active_triplets.mean()
        else:
            return losses.mean()