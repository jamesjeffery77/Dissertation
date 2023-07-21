import torch
from torch import nn

#Perceptron used for evaluating VAE
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 10) #assumes 10 classes like my VAE class

    def forward(self, x):
        return self.fc(x)
