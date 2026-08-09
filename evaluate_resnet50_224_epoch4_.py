"""
Evaluate a trained ResNet50 model on the independent CMMD test set

This script loads the selected ResNet50 checkpoint and evaluates it on the independent test set.
Performance is assessed using accuracy, AUC, sensitivity, specificity , precision, F1-score and a confusion matrix
The script also saves the test metrics and confusion matrix for the
evaluated model.
"""
# Import the necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
)

from cmmd_resnet50_dataset import CMMDResNetDataset

# Step 1: Select the computation device
# The dissertation experiments will be conducted using a CUDA-enabled GPU
# CPU is retained as a fallback if CUDA is unavailable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 2: Create output folder
os.makedirs("outputs/evaluation", exist_ok=True)

# Step 3: Create test dataset and DataLoader

test_dataset = CMMDResNetDataset(
    "outputs/test_manifest.csv",
    target_size=224
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
    pin_memory=True
)

# Step 4: Load ResNet50 model architecture

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)

# Step 5: Load the selected model checkpoint

checkpoint_path = "outputs/checkpoints/resnet50_224_epoch_04.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Loaded checkpoint:", checkpoint_path)


# Step 6: Evaluate on the test set

model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())


# Step 7: Calculate test metrics

test_accuracy = accuracy_score(all_labels, all_predictions)

test_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn + fp)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=1
)

test_precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=1
)

# Step 8: Print results

print("\nTest results for ResNet50 224 x 224 epoch04")
print("Test accuracy:", test_accuracy)
print("Test AUC:", test_auc)
print("Test sensitivity:", test_sensitivity)
print("Test specificity:", test_specificity)
print("Test precision:", test_precision)
print("Test F1-score:", test_f1)
print("Confusion matrix:")
print(cm)


# Step 9: Save results

results_df = pd.DataFrame({
    "resolution": ["224 x 224"],
    "selected_epoch":[4],
    "test_accuracy": [test_accuracy],
    "test_auc": [test_auc],
    "test_sensitivity": [test_sensitivity],
    "test_specificity": [test_specificity],
    "test_precision": [test_precision],
    "test_f1_score": [test_f1],
    "true_negative": [tn],
    "false_positive": [fp],
    "false_negative": [fn],
    "true_positive": [tp]
})

