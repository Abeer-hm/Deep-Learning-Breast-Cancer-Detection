"""
Define the CMMD PyTorch dataset and preprocessing pipeline for ResNet50.

This script reads DICOM mammograms from a labelled manifest, applies
per-image min-max normalisation, square padding, resizing, three-channel
conversion, and ImageNet normalisation.

It returns the preprocessed image tensor together with the binary
classification label for use in ResNet50 training and evaluation.
"""
#Import Path to work with file paths.
from pathlib import Path

# Import pandas to read the labelled manifest.
import pandas as pd

# Import NumPy to work with image pixel values.
import numpy as np

# Import pydicom to read DICOM mammograms.
import pydicom

# Import PyTorch.
import torch

# Import Dataset, which will be used to create the CMMD dataset class.
from torch.utils.data import Dataset

# Import PyTorch functions for padding and resizing.
import torch.nn.functional as F

# Create a custom PyTorch Dataset for the CMMD mammograms.
class CMMDResNetDataset(Dataset):

    # This function runs once when the dataset is created.
    def __init__(self, manifest_file, target_size=224):

        # Read the labelled manifest CSV file.
        self.manifest_df = pd.read_csv(manifest_file)

        # Save the selected image size.
        # This allows the dataset to be used with different input resolutions.
        self.target_size = target_size
    
   # return the total number of image records in the manifest.
    def __len__(self):

        return len(self.manifest_df)

    # Read and preprocess one image using its row number.
    def __getitem__(self, index):

        # Select one row from the manifest.
        row = self.manifest_df.iloc[index]

        # Get the DICOM image path from that row.
        image_path = row["image_path"]

        # Read the DICOM file.
        dicom = pydicom.dcmread(image_path)

        # Extract the pixel values and convert them to float32.
        pixels = dicom.pixel_array.astype(np.float32)

        # Find the minimum and maximum pixel values.
        pixel_min = pixels.min()
        pixel_max = pixels.max()

        # Scale the pixel values to a range between 0 and 1.
        if pixel_max > pixel_min:
            normalised = (
                pixels - pixel_min
            ) / (
                pixel_max - pixel_min
            )
        else:
            normalised = np.zeros_like(pixels)

        # Convert the normalised NumPy array into a PyTorch tensor.
        image_tensor = torch.from_numpy(normalised)

        # Add the channel dimension.
        # Shape changes from:
        # height x width
        # to:
        # 1 x height x width
        image_tensor = image_tensor.unsqueeze(0)

        # Get the channel count, image height, and image width.
        _, height, width = image_tensor.shape

        # Use the longer side as the square size.
        square_size = max(height, width)

        # Calculate the padding needed to make the image square.
        padding_left = (square_size - width) // 2
        padding_right = square_size - width - padding_left

        padding_top = (square_size - height) // 2
        padding_bottom = square_size - height - padding_top

        # Add black padding to make the image square.
        padded_tensor = F.pad(
            image_tensor,
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
            ),
            value=0,
        )

        # Add a temporary batch dimension, resize the image,
        # then remove the temporary batch dimension.
        resized_tensor = F.interpolate(
            padded_tensor.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Repeat the single grayscale channel three times.
        # Shape changes from:
        # 1 x target_size x target_size
        # to:
        # 3 x target_size x target_size
        three_channel_tensor = resized_tensor.repeat(3, 1, 1)

        # ImageNet mean values for the three channels.
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=three_channel_tensor.dtype
        ).view(3, 1, 1)

        # ImageNet standard deviation values for the three channels.
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=three_channel_tensor.dtype
        ).view(3, 1, 1)

        # Apply the normalisation expected by pretrained ResNet50.
        image_tensor = (
            three_channel_tensor - imagenet_mean
        ) / imagenet_std

        # Convert the classification label into a numeric value.
        # Benign = 0
        # Malignant = 1
        if row["classification"].lower() == "benign":
            label = 0
        else:
            label = 1

        # Return the processed image tensor and its numeric label.
        return image_tensor, label


# Test the custom Dataset class.
if __name__ == "__main__":

    # Create the dataset using the labelled manifest
    # and the baseline resolution of 224 x 224.
    dataset = CMMDResNetDataset(
        "outputs/image_manifest_labelled.csv",
        target_size=224
    )

    # Print the total number of images.
    print("Dataset size:", len(dataset))

    # Read and preprocess the first image.
    image, label = dataset[0]

    # Print the processed image shape and label.
    print("Image shape:", image.shape)
    print("Label:", label)

    # Print the minimum and maximum tensor values.
    print("Minimum value:", image.min().item())
    print("Maximum value:", image.max().item())
