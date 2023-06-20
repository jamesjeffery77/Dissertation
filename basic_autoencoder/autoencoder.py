import torch
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Parameters
learning_rate = 1e-3
batch_size = 64
epochs = 20

# Datasets
train_data = datasets.MNIST('path_to_data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('path_to_data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Model, Loss and Optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    for data, _ in train_loader:
        data = data.view(data.size(0), -1)  # Reshape the data
        output = model(data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Testing
model.eval()  # Switch the model to evaluation mode

encoded_images = []
targets = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(data.size(0), -1)  # Reshape the data
        encoded_img = model.encode(data).cpu().numpy()  # convert to numpy array
        encoded_images.extend(encoded_img)
        targets.extend(target.numpy().tolist())  # convert tensor to list

# Convert list to numpy array
encoded_images = np.array(encoded_images)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(encoded_images)
print("completed t-SNE")
