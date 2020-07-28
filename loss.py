import torch.nn as nn
import torch
import numpy as np


class Myloss(nn.Module):
    def __init__(self):
        super().__init__()

    """
        Cross Entropy Error Function
    """

    def forward(self, y_head, y, K):
        self.K = K
        sigma = 0
        for i in range(np.math.factorial(K)):
            sigma += self.P(y_head) * torch.log(self.P(y))
        return -sigma

    """
        calculate probabily of each/all permutation(s)
    """

    def P(self, fw):
        p = 1
        for t in range(self.K):
            upper = torch.exp(fw[t])
            downer = 0
            for l in range(fw.shape[0]):
                downer += torch.exp(fw[l])
            p *= upper / downer
        return p.float()