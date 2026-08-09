"""
Calculate class weights for the CMMD training set.

This script calculates balanced class weights for the benign and malignant
classes using the training manifest. The resulting weights are converted
to a PyTorch tensor for use with weighted CrossEntropyLoss during model
training.
"""
import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Load the training manifest only.
train_df = pd.read_csv("outputs/train_manifest.csv")

# Create numerical labels from the classification column.
train_df["label"] = train_df["classification"].str.lower().map({
    "benign": 0,
    "malignant": 1
})

# Extract training labels.
labels = train_df["label"].values

# Calculate class weights from the training labels.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels
)

# Convert class weights to a PyTorch tensor.
class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("Class weights:", class_weights)
print("Class weights tensor:", class_weights_tensor)
