"""
Create PyTorch DataLoaders for the CMMD training, validation, and test sets.

This script loads the patient-level split manifests using the custom
CMMDResNetDataset class and creates DataLoaders for model training,
validation, and independent test evaluation.
"""
# Import Dataloader to load images in batches
from torch.utils.data import DataLoader
# Import the custom CMMD Dataset class
from cmmd_resnet50_dataset import CMMDResNetDataset

# Create the training dataset.
train_dataset = CMMDResNetDataset(
    "outputs/train_manifest.csv",
    target_size=224
)

# Create the validation dataset.
val_dataset = CMMDResNetDataset(
    "outputs/validation_manifest.csv",
    target_size=224
)

# Create the test dataset.
test_dataset = CMMDResNetDataset(
    "outputs/test_manifest.csv",
    target_size=224
)

# Create the training DataLoader.
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# Create the validation DataLoader.
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Create the test DataLoader.
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("Training batch image shape:", images.shape)
print("Training batch label shape:", labels.shape)
print("Training batch labels:", labels)
