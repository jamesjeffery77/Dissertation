from finalvae_change_hp import VAE
from perceptron_change_hp import Perceptron
from utils_change_hp import loss_function, create_directories, load_data, train_model, evaluate_model, train_perceptron, evaluate_perceptron, visualize_results, LEARNING_RATE, BATCH_SIZE, DATA_PATH, EPOCHS, folder_path
import torch.optim as optim
import torch.nn as nn
import torch
import os
import pandas as pd

def main_hyperparameter_tuning():
    # Define the combinations
    combinations = [
        ([784, 128, 64, 36, 18, 9], 9, [9, 18, 36, 64, 128, 784]), #initial one
        ([784, 512], 256, [256, 512, 784]), #really shallow and wide
        ([784, 512, 256, 128, 64, 32], 32, [32, 64, 128, 256, 512, 784]), # kind of shallow and wide
        ([784, 400, 200, 100, 50], 20, [20, 50, 100, 200, 400, 784]), #deep and narrow
        ([784, 256, 128, 64, 32, 16, 8, 4], 4, [4, 8, 16, 32, 64, 128, 256, 784]), #really deep and narrow
        ([784, 128, 64, 32], 16, [16, 32, 64, 128, 784]), #asymmetric
        ([784, 128, 64], 10, [10, 64, 128, 784]) #bottleneck
    ]

    dir_paths = create_directories()  # Create directories once
    metrics_df = pd.DataFrame()
    metrics_list = []

    # Loop through the combinations
    for idx, combo in enumerate(combinations):
        print(f"Running combination {idx + 1} of {len(combinations)}: {combo}")
        model = VAE(combo[0], combo[1], combo[2])
        #metrics_df = main(model, dir_paths, idx + 1, metrics_df, combo)  # Pass the combination index to main
        metrics = main(model, dir_paths, idx + 1, metrics_df, combo)  
        metrics_list.append(metrics)

    # Save DataFrame to csv after all combinations have run
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('change_hp_metrics.csv', index=False)

def main(model, dir_paths, combo_idx, metrics_df, hyperparams):
    # Prepare data
    train_loader, val_loader, test_loader = load_data(DATA_PATH, BATCH_SIZE)
    
    # Define model and optimizer
    #model = VAE()
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    ae_losses_test, classifier_losses_test, val_ae_losses, val_classifier_losses = train_model(model, train_loader, val_loader, EPOCHS, classifier_criterion, optimizer)   
    
    # Evaluate model
    metrics, ae_losses, classifier_losses, original_imgs, reconstructed_imgs, encoded_images, targets, conf_mat, f1, cluster_labels, rand_index = evaluate_model(model, test_loader, classifier_criterion, optimizer)   
    
    # Visualize results
    visualize_results(ae_losses_test, classifier_losses_test, val_ae_losses, val_classifier_losses, original_imgs, reconstructed_imgs, encoded_images, targets, conf_mat, cluster_labels, dir_paths, combo_idx)
    print(f'Weighted F1 Score: {f1}')  # Print the F1 Score
    print(f'Rand Index: {rand_index}')  # Print the Rand Index

    #Define perceptron model and optimizer
    latent_dim = model.latent_dim  # Assuming 'latent_dim' attribute exists in your VAE model which corresponds to the dimension of 'mu'
    print("latent dim: " + str(latent_dim))
    perceptron_model = Perceptron(latent_dim)
    perceptron_optimizer = torch.optim.Adam(perceptron_model.parameters(), lr=LEARNING_RATE)

    #Train and evaluate the perceptron model
    train_perceptron(model, perceptron_model, train_loader, perceptron_optimizer, classifier_criterion)
    perceptron_accuracies = evaluate_perceptron(model, perceptron_model, test_loader, classifier_criterion, dir_paths, combo_idx)

    # Add hyperparameters to metrics
    metrics["Hyperparameters"] = str(hyperparams)
    metrics["Combination"] = combo_idx
    metrics["Perceptron Accuracies"] = str(perceptron_accuracies)

    #return metrics_df
    return metrics
# Entry point
if __name__ == "__main__":
    main_hyperparameter_tuning()
