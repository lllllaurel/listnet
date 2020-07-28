import torch
import torch.nn as nn
import numpy as np
from loss import Myloss
import logging


class ListNet():
    def __init__(self, h1_input, h1_output, h2_input, h2_output):
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
            loss = Myloss().forward(y_head, y, self.K)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.probability = self.model(torch.tensor(x).float())
        return self.probability

    def permutation(self, id):
        perm = np.hstack((id, self.probability.detach().numpy()))
        return sorted(perm, key=lambda x: x[1], reverse=True)