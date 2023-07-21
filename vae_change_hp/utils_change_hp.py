import torch
from torch import nn
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
import pandas as pd

#CONSTANTS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5
DATA_PATH = 'path_to_data'
ROOT_FOLDER = 'C:\\Users\\james\\autoencoder\\vae_use\\vae_change_hp'
VISUALIZATION_DIRS = ['loss_plots', 'reconstructed_images', 'tsne_plots', 'confusion_matrices']


# Create a new folder with current date and time
folder_path = 'C:\\Users\\james\\autoencoder\\vae_use'
#results_folder_path = os.path.join(folder_path, 'results')
os.makedirs(folder_path, exist_ok=True)
#os.makedirs(results_folder_path, exist_ok=True)

def create_directories():
    dir_paths = {}
    for dir_name in VISUALIZATION_DIRS:
        dir_path = os.path.join(ROOT_FOLDER, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        dir_paths[dir_name] = dir_path
    
    return dir_paths

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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


def train_model(model, train_loader, val_loader, EPOCHS, classifier_criterion, optimizer):
    # Initialize empty lists to store losses for each epoch
    ae_losses = []
    classifier_losses = []
    #initialize empty list to store validation losses for each epoch. 
    val_ae_losses = []  
    val_classifier_losses = []  

    #print("Starting training process...")
    # Training
    for epoch in range(EPOCHS):
        epoch_ae_loss = 0.0
        epoch_classifier_loss = 0.0
        num_batches = 0

        val_epoch_ae_loss = 0.0
        val_epoch_classifier_loss = 0.0
        num_val_batches = 0

        model.train()
        for data, targets in train_loader:
            data = data.view(data.size(0), -1)  # Reshape the data
            recon_data, pred_class, mu, logvar = model(data)  # Get the reconstructed data, predicted class, mu and logvar
            loss = loss_function(recon_data, data, mu, logvar)  # Using the loss_function instead of criterion
            # classifier loss
            targets = targets.long()  # Convert targets to type Long
            classifier_loss = classifier_criterion(pred_class, targets)
            # total loss
            total_loss = loss + classifier_loss
            # zero the gradients
            optimizer.zero_grad()
            # backward propagation for autoencoder and classifier
            total_loss.backward()
            optimizer.step()
            # Add the losses for each batch to the total
            epoch_ae_loss += loss.item()
            epoch_classifier_loss += classifier_loss.item()
            num_batches += 1

        model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.view(data.size(0), -1)
                recon_data, pred_class, mu, logvar = model(data)
                loss = loss_function(recon_data, data, mu, logvar)
                targets = targets.long()
                classifier_loss = classifier_criterion(pred_class, targets)
                val_epoch_ae_loss += loss.item()
                val_epoch_classifier_loss += classifier_loss.item()
                num_val_batches += 1
        # Calculate average losses
        epoch_ae_loss /= num_batches
        epoch_classifier_loss /= num_batches
        val_epoch_ae_loss /= num_val_batches
        val_epoch_classifier_loss /= num_val_batches
        # Append losses to the lists
        ae_losses.append(epoch_ae_loss)
        classifier_losses.append(epoch_classifier_loss)
        val_ae_losses.append(val_epoch_ae_loss)
        val_classifier_losses.append(val_epoch_classifier_loss)

        print(f'Epoch: {epoch+1}, AE Loss: {epoch_ae_loss}, Classifier Loss: {epoch_classifier_loss}, Val AE Loss: {val_epoch_ae_loss}, Val Classifier Loss: {val_epoch_classifier_loss}')

    return ae_losses, classifier_losses, val_ae_losses, val_classifier_losses

def evaluate_model(model, test_loader, classifier_criterion, optimizer):
    model.eval()  # Switch the model to evaluation mode

    ae_losses = []
    classifier_losses = []
    original_imgs = []
    reconstructed_imgs = []
    all_targets = []
    all_preds = []
    encoded_images = []

    with torch.no_grad():  # We don't need gradients for evaluation
        for data, targets in test_loader:
            data = data.view(data.size(0), -1)  # Reshape the data
            recon_data, pred_class, mu, logvar = model(data)  # Get the reconstructed data, predicted class, mu, and logvar
            # Calculate autoencoder and classifier losses
            loss = loss_function(recon_data, data, mu, logvar)
            classifier_loss = classifier_criterion(pred_class, targets)
            _, preds = torch.max(pred_class.data, 1)
            # Append losses, original and reconstructed images, targets and predictions
            ae_losses.append(loss.item())
            classifier_losses.append(classifier_loss.item())
            original_imgs.extend(data.view(-1, 28*28).cpu().numpy())  # Flattened here
            reconstructed_imgs.extend(recon_data.view(-1, 28*28).detach().cpu().numpy())  # And here
            all_targets.extend(targets)
            all_preds.extend(preds)
            # For t-SNE visualization
            z, _, _ = model.encode(data)
            encoded_img = z.cpu().numpy()
            encoded_images.extend(encoded_img)
            
    # Calculate confusion matrix and F1 Score
    conf_mat = confusion_matrix(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')  # Calculate weighted F1 score

    # Convert lists to numpy arrays
    original_imgs = np.stack(original_imgs)
    reconstructed_imgs = np.stack(reconstructed_imgs)
    encoded_images = np.array(encoded_images)
    k = 10  # Choose the desired number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(encoded_images)

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
    conf_mat = confusion_matrix(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')  # Calculate weighted F1 score

    # Calculate Transmitted Information
    trans_info = compute_transmitted_info(conf_mat)
    print("Transmitted Information: ", trans_info)

    metrics = {
        "Silhouette Score": sil_score,
        "Calinski-Harabasz Index": cal_har_score,
        "Normalized Mutual Information": nmi_score,
        "Transmitted Information": trans_info,
        "Weighted F1 Score": f1,
        "Rand Index": rand_index
    }
    print(metrics)
    #metrics_df = pd.DataFrame([metrics])
    return metrics, ae_losses, classifier_losses, original_imgs, reconstructed_imgs, encoded_images, all_targets, conf_mat, f1, cluster_labels, rand_index

def visualize_results(ae_losses, classifier_losses, val_ae_losses, val_classifier_losses, original_imgs, reconstructed_imgs, encoded_images, targets, conf_mat, cluster_labels, dir_paths, combo_idx):
    # After training, plot the losses
    fig, axs = plt.subplots(2, figsize=(10,10))

    axs[0].plot(ae_losses, label="AE Loss")
    axs[0].plot(val_ae_losses, label="Val AE Loss")  # Add validation losses to the plot
    axs[0].set_title("Autoencoder Loss During Training")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].plot(classifier_losses, label="Classifier Loss")
    axs[1].plot(val_classifier_losses, label="Val Classifier Loss")  # Add validation losses to the plot
    axs[1].set_title("Classifier Loss During Training")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_paths['loss_plots'], f'loss_plot_combination_{combo_idx}.png'))  # Save plot
    plt.close(fig)
    plt.clf()  # clear the figure

    # Visualizing the reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12,6))

    # Input images on top row, reconstructions on bottom
    for images, row in zip([original_imgs[:5], reconstructed_imgs[:5]], axes):
        for img, ax in zip(images, row):
            img_2d = img.reshape((28, 28))  # Reshape 1D image to 2D
            ax.imshow(img_2d.squeeze(), cmap='gray')  # Use the reshaped image here
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(dir_paths['reconstructed_images'], f'reconstructed_images_combination_{combo_idx}.png'))  # Save plot instead of showing it
    plt.close(fig)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj = tsne.fit_transform(encoded_images)

    # Plotting t-SNE results
    scatter = plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=targets, cmap='viridis')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(dir_paths['tsne_plots'], f'tsne_plot_combination_{combo_idx}.png'))  # Save plot instead of showing it
    plt.close(fig)
    plt.clf()

    # Apply t-SNE with cluster labels
    tsne_cluster = TSNE(n_components=2, random_state=0)
    tsne_obj_cluster = tsne_cluster.fit_transform(encoded_images)

    # Plotting t-SNE results with cluster labels
    scatter_cluster = plt.scatter(tsne_obj_cluster[:, 0], tsne_obj_cluster[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter_cluster)
    plt.savefig(os.path.join(dir_paths['tsne_plots'], f'tsne_plot_cluster_combination_{combo_idx}.png'))  # Save plot instead of showing it
    plt.close(fig)
    plt.clf()

    # CONFUSION MATRIX
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt=".0f", square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(dir_paths['confusion_matrices'], f'confusion_matrix_combination_{combo_idx}.png'))  # Save plot instead of showing it
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

def evaluate_perceptron(vae_model, perceptron, data_loader, criterion, dir_paths, combo_idx):
    vae_model.eval()
    perceptron.eval()
    all_targets = []
    all_preds = []
    perceptron_accuracies = {}
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
    accuracy = 100 * sum([a == b for a, b in zip(all_targets, all_preds)]) / len(all_targets)
    print('Accuracy of the network on the test mu vectors: %d %%' % accuracy)
    # Save accuracy to the dictionary
    perceptron_accuracies[str(combo_idx)] = accuracy
    conf_mat = confusion_matrix(all_targets, all_preds)
    
    # CONFUSION MATRIX
    fig = plt.figure(figsize=(10, 10))  # create a new figure
    sns.heatmap(conf_mat, annot=True, fmt=".0f", square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Perceptron Confusion matrix')
    plt.savefig(os.path.join(dir_paths['confusion_matrices'], f'perceptron_confusion_matrix_combination_{combo_idx}.png'))  # Save plot instead of showing it
    plt.close(fig)  # close the figure

    return perceptron_accuracies


