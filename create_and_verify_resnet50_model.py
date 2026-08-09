"""
Create and verify the ResNet50 model used for CMMD mammogram classification
This script loads CMMD training dataset and Dataloaders , initalizes an ImageNet-pretrained ResNet50 model, replaces the final classification layer
the original fully connected layer with a two-class output layer for 
 for benign and malignant classification and performs a forward pass on one 
training batch to verify the model and data pipeline
"""

# Import the main PyTorch library
import torch

# Import neural network tools from PyTorch
# I need this to create a new final classification layer
import torch.nn as nn

# Import pretrained models from torchvision
# ResNet50 is available here
from torchvision import models

# Import DataLoader to create image batches
from torch.utils.data import DataLoader

# Import custom CMMD Dataset class
# This class reads DICOM images, preprocesses them, and returns image tensor + label.
from cmmd_resnet50_dataset import CMMDResNetDataset

# Step 1: select the computation device
# The dissertation experiments will be conducted using a CUDA-enabled GPU
# CPU is retained only as a fallback if CUDA is unavailable 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Step 2: Create the training dataset

# This dataset reads the training manifest file
# It knows where the images are, how to read them, how to preprocess them,
# and how to return each image with its label

train_dataset = CMMDResNetDataset(
    "outputs/train_manifest.csv",
    target_size=224
)

# Step 3: Create the training DataLoader

# The DataLoader divides the dataset into batches
# batch_size=16 means each batch contains 16 images
# shuffle=True means the training images are mixed before loading
# num_workers=0 means the main Python process loads the images

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0)
 
# Step 4: Load pretrained ResNet50

# Load ResNet50 with pretrained ImageNet weights
# No layers are frozen, so all model parameters remain trainable
model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

# Step 5: Check the original final layer input size

# ResNet50 has a final fully connected layer called "fc"
# model.fc.in_features gives the number of features entering this final layer

# For ResNet50, this is usually 2048
num_features = model.fc.in_features

print("Number of features entering the final layer:", num_features)

# Step 6: Replace the final layer
# The original ResNet50 final layer outputs 1000 classes for ImageNet
# my task has only 2 classes:
# 0 = benign
# 1 = malignant

# So I replace the final layer with a new layer that outputs 2 classes 
model.fc = nn.Linear(num_features, 2)

# Step 7: Move the model to GPU or CPU

# The model must be moved to the selected device
model = model.to(device)

# Step 8: Get one batch from the DataLoader

# This gets one batch of images and labels from the training DataLoader
# Because batch_size=16, it should return 16 images and 16 labels
images, labels = next(iter(train_loader))

# Step 9: Move the batch to the same device as the model

# The images and labels must be on the same device as the model
labels = labels.to(device)

# Step 10: Pass the images through the model

# This sends the image batch into ResNet50
# The model will return output scores for each image
outputs = model(images)

# Step 11: Print the shapes to check everything

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Model output shape:", outputs.shape)
