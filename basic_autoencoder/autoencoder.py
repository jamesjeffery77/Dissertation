import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

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
model = AE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training process...")
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

print("Training completed. Starting encoding and t-SNE process...")

# Training...

# Testing and Visualization
model.eval()  # Switch the model to evaluation mode

# Visualizing the reconstructed images
with torch.no_grad():
    # Take one batch from your test loader
    data, _ = next(iter(test_loader)) 
    data = data.view(data.size(0), -1)  # Reshape the data
    encoded_imgs = model.encode(data)
    decoded_imgs = model.decoder(encoded_imgs)  # Get reconstructed images through the decoder only


    # Get the first 5 images from the batch, reshape them to 28*28 and detach the gradient
    original_imgs = data[:5].view(-1, 28, 28).detach()
    reconstructed_imgs = decoded_imgs[:5].view(-1, 28, 28).detach()

    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12,6))

    # Input images on top row, reconstructions on bottom
    for images, row in zip([original_imgs, reconstructed_imgs], axes):
        for img, ax in zip(images, row):
            ax.imshow(img, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

# Continue with the t-SNE visualization
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
print("Starting t-SNE...")
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(encoded_images)
print("Completed t-SNE.")

# Plotting
print("Starting plotting...")
scatter = plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=targets, cmap='viridis')
plt.colorbar(scatter)
plt.show()
print("Completed Plotting.")
