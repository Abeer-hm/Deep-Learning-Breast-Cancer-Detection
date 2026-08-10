"""
Train and validate an ImageNet-pretrained ResNet50 model on CMMD mammograms.

This script represents the 224 × 224 training experiment. ResNet50 is
trained using full fine-tuning for binary classification of benign and
malignant mammograms. Training uses class-weighted CrossEntropyLoss, the
Adam optimiser, and 20 epochs.

The experiment is repeated using different random seeds. Validation
performance is calculated after each epoch, checkpoints are saved, and
epoch runtime is recorded.

The same training pipeline was applied to the other input resolutions by
changing the input resolution and corresponding run settings.
"""
# Import the necessary libraries
import time
import pandas as pd
import matplotlib.pyplot as plt
import os 
import random
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

import torch

# Import neural network tools.
import torch.nn as nn

# Import DataLoader to create image batches.
from torch.utils.data import DataLoader

# Import pretrained models from torchvision.
from torchvision import models

# Import my custom CMMD Dataset class.
from cmmd_resnet50_dataset import CMMDResNetDataset


# Set the random seed for the current training run
# The experimens were repeated using seeds 42, 123 and 2026
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use deterministic cuDNN behaviour to improve reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Training random seed:", SEED)

# Step 1: selecte the computation device

# Use GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Step 2: Set training settings

# Number of training epochs
num_epochs =20

# Number of images in each training batch
batch_size = 16

# Learning rate used by Adam optimizer
learning_rate = 0.0001

run_name = f"resnet50_224_seed{SEED}"

# Create folders for saving results.
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# Step 3: Create training and validation datasets

# Create the training dataset at the representative 224 × 224 resolution
train_dataset = CMMDResNetDataset(
    "outputs/train_manifest.csv",
    target_size=224
)

# Create the validation dataset using the same input resolution
val_dataset = CMMDResNetDataset(
    "outputs/validation_manifest.csv",
    target_size=224
)

# Step 4: Create DataLoaders
# Create a seeded generator for reproducible training-data shuffling
train_generator = torch.Generator()
train_generator.manual_seed(SEED)

# Training DataLoader:
# shuffle=True because training images should be mixed each epoch
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    generator=train_generator
)

# Validation DataLoader:
# shuffle=False because validation should be kept in fixed order
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Step 5: Load pretrained ResNet50 with ImageNet-pretrained weights
# No layers are frozen, so all model parameters remain trainable
model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

# Get the number of features entering the original final layer
num_features = model.fc.in_features

# Replace the original ImageNet classification layer with a
# two-class output layer: benign = 0 and malignant = 1.
model.fc = nn.Linear(
    num_features,
    2
)

# Move the model to GPU or CPU.
model = model.to(device)

# Step 6: Define loss function and optimizer

# Class weights calculated from the training set only:
# 0 = benign, 1 = malignant
class_weights = torch.tensor(
    [1.6790, 0.7120],
    dtype=torch.float32
).to(device)

# Cross-entropy loss is used to account for class imbalance

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# Adam optimizer updates the training weights 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Step 7: Training and validation loop

# Create lists to store the training history 

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_aucs = []
val_sensitivities = []
val_specificities = []
val_f1_scores = []

# Store the total training time
epoch_times_seconds=[]


for epoch in range(num_epochs):

    # Record the start time of the complete training run
    epoch_start_time = time.time()

    print("\nEpoch", epoch + 1, "of", num_epochs)

    # Training phase
    
    # Put the model in training mode.
    model.train()

    # Counters for training loss and accuracy.
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    # Loop through training batches.
    for images, labels in train_loader:

        # Move images and labels to GPU or CPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass: get model predictions.
        outputs = model(images)

        # Calculate training loss.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Update model weights.
        optimizer.step()

        # Add this batch loss to the total training loss.
        train_running_loss += loss.item()

        # Get predicted class labels.
        _, predicted = torch.max(outputs, 1)

        # Count total images and correct predictions.
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Calculate average training loss and accuracy for this epoch
    train_loss = train_running_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    
    # Validation phase

    # Put the model in evaluation mode.
    model.eval()

    # Counters for validation loss and accuracy.
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_val_labels = []
    all_val_predictions = []
    all_val_probabilities = []
    # No gradients are needed during validation.
    with torch.no_grad():

        # Loop through validation batches.
        for images, labels in val_loader:

            # Move images and labels to GPU or CPU.
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(outputs, labels)

            # Add this batch loss to the total validation loss.
            val_running_loss += loss.item()

            # Get predicted class labels.
            _, predicted = torch.max(outputs, 1)

            # Convert model outputs to probabilities.
            probabilities = torch.softmax(outputs, dim=1)

            # Store true labels, predicted labels and malignant probabilities.
            all_val_labels.extend(labels.cpu().numpy())
            all_val_predictions.extend(predicted.cpu().numpy())
            all_val_probabilities.extend(probabilities[:, 1].cpu().numpy())  
            
           # Count total images and correct predictions.
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy for this epoch.
    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    #Calculate validation metrics.
    val_auc = roc_auc_score(
    all_val_labels,
    all_val_probabilities
    )
    # Save the model if this epoch has the best validation AUC so far.
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            f"outputs/{run_name}_best_auc.pth"
        )

        print("New best validation AUC model saved.")
        print("Best epoch:", best_epoch)
        print("Best validation AUC:", best_val_auc)
    cm = confusion_matrix(
    all_val_labels,
    all_val_predictions,
    labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    val_f1 = f1_score(
        all_val_labels,
        all_val_predictions,
        pos_label=1
    )

    # Print results for this epoch.
    print("Training loss:", train_loss)
    print("Training accuracy:", train_accuracy)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_accuracy)
    print("Validation AUC:", val_auc)
    print("Validation sensitivity:", sensitivity)
    print("Validation specificity:", specificity)
    print("Validation F1-score:", val_f1)
    print("Validation confusion matrix:")
    print(cm)

    print("Best validation AUC:", best_val_auc)
    print("Best epoch:", best_epoch)
    print(f"Best model saved to outputs/{run_name}_best_auc.pth")

    epoch_time_seconds = time.time() - epoch_start_time
    epoch_times_seconds.append(epoch_time_seconds)

    print("Epoch time seconds:", epoch_time_seconds)   
   
 # Save this epoch's results in the history lists.
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    val_aucs.append(val_auc)
    val_sensitivities.append(sensitivity)
    val_specificities.append(specificity)
    val_f1_scores.append(val_f1)

# Save a checkpoint for the current epoch	
    checkpoint_path =( f"outputs/checkpoints/{run_name}_epoch_{epoch + 1:02d}.pth"
    )
    torch.save(
      {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "val_sensitivity": sensitivity,
        "val_specificity": specificity,
        "val_f1_score": val_f1,
      },
     checkpoint_path
    )

    print("Checkpoint saved to:", checkpoint_path)

# Step 8: Save training metrics

# Create a table containing the results from all epochs.
metrics_df = pd.DataFrame({
    "epoch": list(range(1, num_epochs + 1)),
    "seed": [SEED] * num_epochs,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accuracies,
    "val_accuracy": val_accuracies,
    "val_auc": val_aucs,
    "val_sensitivity": val_sensitivities,
    "val_specificity": val_specificities,
    "val_f1_score": val_f1_scores,
    "epoch_time_seconds": epoch_times_seconds 
})

# Save the metrics table as a CSV file.
metrics_df.to_csv(
   f"outputs/metrics/{run_name}_20epochs_metrics.csv",
    index=False
)

print(f"\nMetrics saved to outputs/metrics/{run_name}_20epochs_metrics.csv")


# Print the time of training
training_end_time = time.time()
total_training_time_seconds = training_end_time - training_start_time

print("Total training time seconds:", total_training_time_seconds)


# Step 9: Save loss curve figure


plt.figure(figsize=(7, 5))
plt.plot(metrics_df["epoch"], metrics_df["train_loss"], marker="o", label="Training loss")
plt.plot(metrics_df["epoch"], metrics_df["val_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 224 x 224 Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"outputs/figures/{run_name}_20epochs_loss_curve.png",
    dpi=300
)
plt.close()


# Step 9: Save accuracy curve figure

plt.figure(figsize=(7, 5))
plt.plot(metrics_df["epoch"], metrics_df["train_accuracy"], marker="o", label="Training accuracy")
plt.plot(metrics_df["epoch"], metrics_df["val_accuracy"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 224 x 224 Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"outputs/figures/{run_name}_20epochs_accuracy_curve.png",
    dpi=300
)
plt.close()

print("Training figures saved successfully.")

# Step 10: Save the trained model

# Save the model after the 20-epoch training.
torch.save(
    model.state_dict(),
    f"outputs/{run_name}_20epochs.pth"
)

print("\nFull 224 x 224 training completed successfully.")
print("Model saved to outputs/resnet50_224_42seed_20epochs.pth")


# Step 11  Save validation metrics curve figure

plt.figure(figsize=(7, 5))

plt.plot(
    metrics_df["epoch"],
    metrics_df["val_auc"],
    marker="o",
    label="Validation AUC"
)

plt.plot(
    metrics_df["epoch"],
    metrics_df["val_sensitivity"],
    marker="o",
    label="Validation sensitivity"
)

plt.plot(
    metrics_df["epoch"],
    metrics_df["val_specificity"],
    marker="o",
    label="Validation specificity"
)

plt.plot(
    metrics_df["epoch"],
    metrics_df["val_f1_score"],
    marker="o",
    label="Validation F1-score"
)

plt.xlabel("Epoch")
plt.ylabel("Metric value")
plt.title("ResNet50 224 x 224 Validation Metrics")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"outputs/figures/{run_name}_20epochs_validation_metrics.png",
    dpi=300
)

plt.close()
