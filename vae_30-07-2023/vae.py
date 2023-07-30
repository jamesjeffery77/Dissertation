import torch
from torch import nn
import csv
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time


#CONSTANTS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
DATA_PATH = 'path_to_data'

# Create a new folder with current date and time
folder_path = 'C:\\Users\\james\\autoencoder\\vae_use\\vae'
results_folder_path = os.path.join(folder_path, 'results')
os.makedirs(folder_path, exist_ok=True)
os.makedirs(results_folder_path, exist_ok=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(), 
        )
        
        # latent space
        self.fc_mu = nn.Linear(128, 10)
        self.fc_logvar = nn.Linear(128, 10)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
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

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, 784))
        return self.decode(z), mu, logvar

#Perceptron used for evaluating VAE
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 10) #assumes 10 classes like my VAE class

    def forward(self, x):
        return self.fc(x)


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD

def write_metrics_to_csv(metrics, filename):
    """
    Function to write metrics to a csv file. 
    """
    # If file doesn't exist, write the headers
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()

    # Write the metrics
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writerow(metrics)

def load_data(data_path, batch_size):
    # Load the full training data
    full_train_data = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    # Determine sizes for training and validation sets
    num_train = int(len(full_train_data) * 0.8)  # 80% for training
    num_val = len(full_train_data) - num_train   # 20% for validation

    # Split the data
    train_data, val_data = random_split(full_train_data, [num_train, num_val])

    # Create data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, EPOCHS, optimizer):
    # Initialize empty list to store losses for each epoch
    ae_losses = []
    val_ae_losses = []  

    best_val_loss = float('inf')  # initialize best validation loss as infinity
    patience_counter = 0  # counter to track number of epochs without improvement
    patience = 5

    # Training
    for epoch in range(EPOCHS):
        epoch_ae_loss = 0.0
        num_batches = 0

        val_epoch_ae_loss = 0.0
        num_val_batches = 0

        model.train()
        for data, targets in train_loader:
            data = data.view(data.size(0), -1)  # Reshape the data
            recon_data, mu, logvar = model(data)  # Get the reconstructed data, mu and logvar
            BCE, KLD = loss_function(recon_data, data, mu, logvar)  # Using the loss_function instead of criterion
            loss = BCE + KLD
            # zero the gradients
            optimizer.zero_grad()
            # backward propagation for autoencoder
            loss.backward()
            optimizer.step()
            # Add the losses for each batch to the total
            epoch_ae_loss += loss.item()
            num_batches += 1

        model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.view(data.size(0), -1)
                recon_data, mu, logvar = model(data)
                BCE, KLD = loss_function(recon_data, data, mu, logvar)
                loss = BCE + KLD
                val_epoch_ae_loss += loss.item()
                num_val_batches += 1

        # Calculate average losses
        epoch_ae_loss /= num_batches
        val_epoch_ae_loss /= num_val_batches

        # Add the early stopping check here
        if val_epoch_ae_loss < best_val_loss:
            best_val_loss = val_epoch_ae_loss
            patience_counter = 0  # reset counter
        else:
            patience_counter += 1  # increment counter
            if patience_counter >= patience:  # check if patience has been exceeded
                print(f'Early stopping at epoch {epoch+1}')
                break

        metrics, *_= evaluate_model(model, val_loader, optimizer, epoch)

        #latent space interpolation
        interpolate_in_latent_space(model, train_loader, 5, epoch, results_folder_path, 3, 9)   

        metrics = {
            "Epoch": epoch+1,
            "AE Loss": epoch_ae_loss,
            "Val AE Loss": val_epoch_ae_loss,
            "BCE": BCE.item(),
            "KLD": KLD.item(),
        }
        
        write_metrics_to_csv(metrics, 'training_metrics.csv')
    
        print(f'Epoch: {epoch+1}, AE Loss: {epoch_ae_loss}, Val AE Loss: {val_epoch_ae_loss}')

    return ae_losses, val_ae_losses

def evaluate_model(model, test_loader, optimizer, epoch):
    model.eval()  # Switch the model to evaluation mode

    ae_losses = []
    original_imgs = []
    reconstructed_imgs = []
    encoded_images = []
    all_targets = []

    with torch.no_grad():  # We don't need gradients for evaluation
        for data, targets in test_loader:
            data = data.view(data.size(0), -1)  # Reshape the data
            recon_data, mu, logvar = model(data)  # Get the reconstructed data, mu, and logvar
            # Calculate autoencoder loss
            BCE, KLD = loss_function(recon_data, data, mu, logvar)
            loss = BCE + KLD
            # Append losses, original and reconstructed images
            ae_losses.append(loss.item())
            original_imgs.extend(data.view(-1, 28*28).cpu().numpy())  # Flattened here
            reconstructed_imgs.extend(recon_data.view(-1, 28*28).detach().cpu().numpy())  # And here
            all_targets.extend(targets.cpu().numpy())  # Store the targets

            # For t-SNE visualization
            z, _, _ = model.encode(data)
            encoded_img = z.cpu().numpy()
            encoded_images.extend(encoded_img)
            
    # Convert lists to numpy arrays
    original_imgs = np.stack(original_imgs)
    reconstructed_imgs = np.stack(reconstructed_imgs)
    encoded_images = np.array(encoded_images)

        # Convert lists to numpy arrays
    original_imgs = np.stack(original_imgs)
    reconstructed_imgs = np.stack(reconstructed_imgs)
    encoded_images = np.array(encoded_images)
    k = 10  # Choose the desired number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(encoded_images)

    # NEW: Calculate confusion matrix and Transmitted Information for the VAE
    vae_conf_mat = confusion_matrix(all_targets, cluster_labels)
    vae_trans_info = compute_transmitted_info(vae_conf_mat)
    print("Transmitted Information VAE: ", vae_trans_info)

    # At the end of the function, after the K-means clustering:
    rand_index = adjusted_rand_score(all_targets, cluster_labels)

    # Calculate Silhouette Score
    sil_score = silhouette_score(encoded_images, cluster_labels)
    print("Silhouette Score: ", sil_score)
    
    # Calculate Calinski-Harabasz Index
    cal_har_score = calinski_harabasz_score(encoded_images, cluster_labels)
    print("Calinski-Harabasz Index: ", cal_har_score)
    
    # Calculate Normalized Mutual Information
    nmi_score = normalized_mutual_info_score(all_targets, cluster_labels)
    print("Normalized Mutual Information: ", nmi_score)

    # Calculate confusion matrix and F1 Score
    #conf_mat = confusion_matrix(all_targets, all_preds)
    #f1 = f1_score(all_targets, all_preds, average='weighted')  # Calculate weighted F1 score

    # Calculate Transmitted Information
    #trans_info = compute_transmitted_info(conf_mat)
    #print("Transmitted Information: ", trans_info)

    metrics = {
        "Epoch": epoch,
        "Silhouette Score": sil_score,
        "Calinski-Harabasz Index": cal_har_score,
        "Normalized Mutual Information": nmi_score,
        #"Transmitted Information": trans_info,
        #"Weighted F1 Score": f1,
        "Rand Index": rand_index,
        "Transmitted Information VAE": vae_trans_info
    }

    #return metrics
    write_metrics_to_csv(metrics, 'evaluation_metrics.csv')
    
    return metrics, ae_losses, original_imgs, reconstructed_imgs, encoded_images, all_targets, cluster_labels, rand_index, vae_conf_mat

def visualize_results(ae_losses, val_ae_losses, original_imgs, reconstructed_imgs, encoded_images, targets,  cluster_labels, vae_conf_mat):
    # After training, plot the losses
    fig, axs = plt.subplots(figsize=(10,10))

    axs.plot(ae_losses, label="AE Loss")
    axs.plot(val_ae_losses, label="Val AE Loss")  # Add validation losses to the plot
    axs.set_title("Autoencoder Loss During Training")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    axs.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder_path, 'loss_plot.png'))  # Save plot instead of showing it
    plt.close(fig)

    # Visualizing the reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12,6))

    # Input images on top row, reconstructions on bottom
    for images, row in zip([original_imgs[:5], reconstructed_imgs[:5]], axes):
        for img, ax in zip(images, row):
            img_2d = img.reshape((28, 28))  # Reshape 1D image to 2D
            ax.imshow(img_2d.squeeze(), cmap='gray')  # Use the reshaped image here
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(results_folder_path, 'reconstructed_images.png'))  # Save plot instead of showing it
    plt.close(fig)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj = tsne.fit_transform(encoded_images)

    # Plotting t-SNE results
    scatter = plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=targets, cmap='viridis')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(results_folder_path, 'tsne_plot.png'))  # Save plot instead of showing it
    plt.close(fig)

    # Apply t-SNE with cluster labels
    tsne_cluster = TSNE(n_components=2, random_state=0)
    tsne_obj_cluster = tsne_cluster.fit_transform(encoded_images)

    # Plotting t-SNE results with cluster labels
    scatter_cluster = plt.scatter(tsne_obj_cluster[:, 0], tsne_obj_cluster[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter_cluster)
    plt.savefig(os.path.join(results_folder_path, 'tsne_cluster_plot.png'))  # Save plot instead of showing it
    plt.close(fig)

    # CONFUSION MATRIX Kmeans
    plt.figure(figsize=(10, 10))
    sns.heatmap(vae_conf_mat, annot=True, fmt=".0f", square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(results_folder_path, 'Kmeans_confusion_matrix.png'))  # Save plot instead of showing it
    plt.close(fig)

def compute_transmitted_info(conf_mat):
    K = conf_mat.shape[0] #number of classes or clusters
    n = np.sum(conf_mat)  # total number of responses
    nij = conf_mat  # number of responses to stimulus i clustered into stimulus j

    # create an empty array to store the results of each element in the confusion matrix
    h_elements = np.zeros_like(nij, dtype=float)

    # iterate over each element in the confusion matrix
    for i in range(K):
        for j in range(K):
            # skip if there are no responses for this combination
            if nij[i, j] == 0:
                continue
            
            nkj = np.sum(conf_mat[:, j])  # total number of responses assigned to cluster j
            nik = np.sum(conf_mat[i, :])  # total number of responses to stimulus i

            h_elements[i, j] = nij[i, j] * (np.log(nij[i, j]) - np.log(nkj) - np.log(nik) + np.log(n))

    h = (1 / n) * np.sum(h_elements)
    h_normalized = h / np.log(K)  # normalize transmitted information
    return h_normalized

def train_perceptron(vae_model, perceptron, data_loader, optimizer, criterion):
    vae_model.eval()
    perceptron.train()
    for data, targets in data_loader:
        data = data.view(data.size(0), -1)  # Reshape the data
        _, mu, _ = vae_model.encode(data)  # We only care about mu
        # Train perceptron on mean vectors
        optimizer.zero_grad()
        output = perceptron(mu)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

def evaluate_perceptron(vae_model, perceptron, data_loader, criterion, folder_path):
    vae_model.eval()
    perceptron.eval()
    all_targets = []
    all_preds = []
    all_latent = []
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.view(data.size(0), -1)  # Reshape the data
            _, mu, _ = vae_model.encode(data)  # We only care about mu
            # Forward pass
            output = perceptron(mu)
            # Calculate loss
            _, predicted = torch.max(output.data, 1)
            all_targets.extend(targets)
            all_preds.extend(predicted)
            all_latent.extend(mu.numpy())  # save all the mu vectors
    all_latent = np.array(all_latent)

    # Now we cluster the latent space representation using KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(all_latent)
    # KMeans cluster labels
    kmeans_labels = kmeans.labels_

    # Get confusion matrix with true labels and KMeans cluster labels
    kmeans_conf_mat = confusion_matrix(all_targets, kmeans_labels)
    
    # KMEANS CONFUSION MATRIX
    fig = plt.figure(figsize=(10, 10))  # create a new figure
    sns.heatmap(kmeans_conf_mat, annot=True, fmt=".0f", square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Cluster label')
    plt.title('KMeans Confusion matrix')
    plt.savefig(os.path.join(folder_path, 'perceptron_kmeans_confusion_matrix.png'))  # Save the figure
    plt.close(fig)  # close the figure

    # Calculate Transmitted Information
    perceptron_trans_info = compute_transmitted_info(kmeans_conf_mat)
    print("Transmitted Information Perceptron: ", perceptron_trans_info)
    print('Accuracy of the network on the perceptron test mu vectors: %d %%' % (100 * sum([a == b for a, b in zip(all_targets, all_preds)]) / len(all_targets)))
    conf_mat = confusion_matrix(all_targets, all_preds)
    
    # CONFUSION MATRIX
    fig = plt.figure(figsize=(10, 10))  # create a new figure
    sns.heatmap(conf_mat, annot=True, fmt=".0f", square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Perceptron Confusion matrix')
    plt.savefig(os.path.join(results_folder_path, 'perceptron_confusion_matrix.png'))  # Save the figure
    plt.close(fig)  # close the figure

    return perceptron_trans_info

def get_first_samples_of_classes(dataset, class1, class2):
    """Get the first samples of the specified classes from a dataset."""
    idx1, idx2 = None, None
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == class1 and idx1 is None:
            idx1 = i
        if label == class2 and idx2 is None:
            idx2 = i
        if idx1 is not None and idx2 is not None:
            break
    # Return the samples
    sample1, _ = dataset[idx1]
    sample2, _ = dataset[idx2]
    return sample1, sample2


def get_samples_of_class(dataset, class_label):
    """Return the indices of all samples of a given class in a dataset."""
    indices = []
    for i in range(len(dataset.dataset)):   # Corrected here
        _, label = dataset.dataset[i]       # Corrected here
        if label == class_label:
            indices.append(i)
    return indices

def interpolate_in_latent_space(model, dataset, num_steps, epoch, results_folder_path, class1, class2):
    model.eval()
    
    # Get the indices of the first samples of the specified classes
    idx1 = get_samples_of_class(dataset, class1)[0]
    idx2 = get_samples_of_class(dataset, class2)[0]

    # Get the actual samples
    data_point_1 = dataset.dataset[idx1][0].view(-1)  # Corrected here
    data_point_2 = dataset.dataset[idx2][0].view(-1)  # Corrected here
    
    # Encode the data points to get latent vectors
    latent_vector_1, _, _ = model.encode(data_point_1)
    latent_vector_2, _, _ = model.encode(data_point_2)

    # Interpolate between the latent vectors
    interpolated_vectors = [latent_vector_1 + (latent_vector_2 - latent_vector_1) * i / num_steps for i in range(num_steps+1)]

    # Decode the interpolated vectors
    interpolated_data_points = [model.decode(vector).view(28, 28).detach().cpu().numpy() for vector in interpolated_vectors]
    
    # Save the interpolated images for every 2nd epoch
    if epoch % 2 == 0:
        fig, axs = plt.subplots(1, num_steps+1, figsize=(2*(num_steps+1), 2))
        for i, img in enumerate(interpolated_data_points):
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
        plt.savefig(os.path.join(results_folder_path, f'interpolation_{epoch}.png'))
        plt.close(fig)


#from finalvae import VAE
#from perceptron import Perceptron
#from utils import loss_function, load_data, train_model, evaluate_model, train_perceptron, evaluate_perceptron, visualize_results, LEARNING_RATE, BATCH_SIZE, DATA_PATH, EPOCHS, results_folder_path

def main():
    # Prepare data
    train_loader, val_loader, test_loader = load_data(DATA_PATH, BATCH_SIZE)
    
    #for run in range(5):

            # Define the results directory for this run
        #results_folder_path = str(run + 1)
        # Create the directory if it doesn't exist
        #if not os.path.exists(results_folder_path):
            #os.makedirs(results_folder_path)

    # Define model and optimizer
    model = VAE()
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()

    # Train model
    ae_losses_test, val_ae_losses = train_model(model, train_loader, val_loader, EPOCHS, optimizer)   
    
    # Evaluate model
    metrics, ae_losses, original_imgs, reconstructed_imgs, encoded_images, targets, cluster_labels, rand_index, vae_conf_mat = evaluate_model(model, test_loader, optimizer, "Final")   
    
    
    # Visualize results
    visualize_results(ae_losses_test, val_ae_losses, original_imgs, reconstructed_imgs, encoded_images, targets, cluster_labels, vae_conf_mat)
    #print(f'Weighted F1 Score: {f1}')  # Print the F1 Score
    print(f'Rand Index: {rand_index}')  # Print the Rand Index

    # Write metrics to a csv file
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
    #metrics_df.to_csv('metrics.csv', index=False)  # Write DataFrame to csv

    #Define perceptron model and optimizer
    perceptron_model = Perceptron(input_dim=10)
    perceptron_optimizer = torch.optim.Adam(perceptron_model.parameters(), lr=LEARNING_RATE)

    #Train and evaluate the perceptron model
    train_perceptron(model, perceptron_model, train_loader, perceptron_optimizer, classifier_criterion)
    perceptron_trans_info = evaluate_perceptron(model, perceptron_model, test_loader, classifier_criterion, results_folder_path)

    # Update the metrics dictionary with the perceptron transmitted information
    metrics["Perceptron Transmitted Information"] = perceptron_trans_info

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total time taken

    # Write metrics to a csv file
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
    metrics_df.to_csv('vae_metrics.csv', index=False)

    metrics["Total Time"] = total_time  # Add total time to metrics dictionary
    
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
    metrics_df.to_csv('vae_metrics.csv', index=False)  # Write DataFrame to csv
    
# Entry point
if __name__ == "__main__":
    main()
