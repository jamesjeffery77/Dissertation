import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, n_classes=10):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # latent space
        self.fc_mu = nn.Linear(64, 10)
        self.fc_logvar = nn.Linear(64, 10)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(10, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar) #sigma
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def decode(self, z):
        return self.decoder(z)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, 784))
        return self.decode(z), self.classify(z), mu, logvar

