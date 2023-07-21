import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, encoder_dims, latent_dim, decoder_dims, n_classes=10):
        super(VAE, self).__init__()

        # set the latent_dim as an instance variable
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        for i in range(len(encoder_dims)-1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        for i in range(len(decoder_dims)-1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_dims[-1], encoder_dims[0]))  # Assumes input and output dims are same
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, 784))
        return self.decode(z), self.classify(z), mu, logvar

