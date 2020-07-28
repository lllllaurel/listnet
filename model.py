import torch
import torch.nn as nn
import numpy as np
import logging


class ListNet(nn.Module):
    def __init__(self, h1_input, h1_output, h2_input, h2_output):
        super().__init__()
        self.K = 1
        self.model = nn.Sequential(nn.Linear(h1_input, h1_output), nn.ReLU(),
                                   nn.Linear(h2_input, h2_output), nn.ReLU())

    def train(self, data, label, iters, learning_rate, optmz):
        x = torch.tensor(data).float()
        y = torch.tensor(label).float()
        if optmz.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=learning_rate)

        for t in range(iters):
            self.optimizer.zero_grad()
            y_head = self.model(x)
            loss = self.forward(y_head, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.probability = self.model(torch.tensor(x).float())
        return self.probability

    """
        return a permutation with highest probability
    """
    def permutation(self, id):
        perm = np.hstack((id, self.probability.detach().numpy()))
        return sorted(perm, key=lambda x: x[1], reverse=True)

    """
        Cross Entropy Error Function
    """
    def forward(self, y_head, y):
        sigma = 0
        for i in range(np.math.factorial(self.K)):
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