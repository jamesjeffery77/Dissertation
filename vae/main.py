from finalvae import VAE
from perceptron import Perceptron
from utils import loss_function, load_data, train_model, evaluate_model, train_perceptron, evaluate_perceptron, visualize_results, LEARNING_RATE, BATCH_SIZE, DATA_PATH, EPOCHS, results_folder_path
import torch.optim as optim
import torch.nn as nn
import torch
import os

def main():
    # Prepare data
    train_loader, val_loader, test_loader = load_data(DATA_PATH, BATCH_SIZE)
    
    # Define model and optimizer
    model = VAE()
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    ae_losses_test, classifier_losses_test, val_ae_losses, val_classifier_losses = train_model(model, train_loader, val_loader, EPOCHS, classifier_criterion, optimizer)   
    
    # Evaluate model
    metrics, ae_losses, classifier_losses, original_imgs, reconstructed_imgs, encoded_images, targets, conf_mat, f1, cluster_labels, rand_index = evaluate_model(model, test_loader, classifier_criterion, optimizer)   
    
    # Visualize results
    visualize_results(ae_losses_test, classifier_losses_test, val_ae_losses, val_classifier_losses, original_imgs, reconstructed_imgs, encoded_images, targets, conf_mat, cluster_labels)
    print(f'Weighted F1 Score: {f1}')  # Print the F1 Score
    print(f'Rand Index: {rand_index}')  # Print the Rand Index

    # Write metrics to a csv file
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
    metrics_df.to_csv('metrics.csv', index=False)  # Write DataFrame to csv

    #Define perceptron model and optimizer
    perceptron_model = Perceptron(10)
    perceptron_optimizer = torch.optim.Adam(perceptron_model.parameters(), lr=LEARNING_RATE)

    #Train and evaluate the perceptron model
    train_perceptron(model, perceptron_model, train_loader, perceptron_optimizer, classifier_criterion)
    perceptron_trans_info = evaluate_perceptron(model, perceptron_model, test_loader, classifier_criterion, results_folder_path)

    # Update the metrics dictionary with the perceptron transmitted information
    metrics["Perceptron Transmitted Information"] = perceptron_trans_info

    # Write metrics to a csv file
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
    metrics_df.to_csv('vae_metrics.csv', index=False)
    
# Entry point
if __name__ == "__main__":
    main()
