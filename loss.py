from torch import nn
from utils import L2_dist
import torch
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, anc, pos, neg):
        pos_dist = L2_dist(anc, pos)
        neg_dist = L2_dist(anc, neg)
        return torch.mean(max(pos_dist - neg_dist + self.alpha, 0))

    def get_hard(self, anc, pos, neg):
        pos_dist = L2_dist(anc, pos)
        neg_dist = L2_dist(anc, neg)
        
        hard_mask = (pos_dist - neg_dist + self.alpha > 0).cpu().numpy().flatten()
        hard_indexes = np.where(hard_mask == True)
        return hard_indexes