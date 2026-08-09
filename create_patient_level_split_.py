"""
Create patient-level training, validation, and test splits for the CMMD dataset.

This script groups mammograms by patient, creates patient-level categories
from the breast-level classification labels, and performs a stratified
70/15/15 split using a fixed random seed of 42.

All images from the same patient are kept within the same split to prevent
patient-level data leakage
"""
# Import pandas to read and organise the labelled manifest.
import pandas as pd

# Import train_test_split to divide patients into groups.
from sklearn.model_selection import train_test_split

# Read the final labelled image manifest.
manifest_df = pd.read_csv(
    "outputs/image_manifest_labelled.csv"
)

# Display the number of image records and unique patients.
print("Total images:", len(manifest_df))
print(
    "Unique patients:",
    manifest_df["patient_id"].nunique()
)

# Create one patient-level category from all image labels.
patient_classification = (
    manifest_df

    # Group image labels by patient.
    .groupby("patient_id")["classification"]

    # Keep unique labels and combine them into one category.
    # Examples: benign, malignant, or benign_malignant.
    .apply(
        lambda values: "_".join(
            sorted(values.str.lower().unique())
        )
    )

    # Return a table with patient_id and patient_category.
    .reset_index(name="patient_category")
)

# Split patients into 70% training and 30% temprary data
train_patients, temp_patients = train_test_split(
    patient_classification,
    test_size=0.30,
    random_state=42,
    stratify=patient_classification["patient_category"]
)

# Split the temporary group equally into validation and test sets.
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
    stratify=temp_patients["patient_category"]
)

print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Test patients:", len(test_patients))


# Assign the images to the three patient groups 
# Store the patient IDs from each split.
train_patient_ids = set(train_patients["patient_id"])
val_patient_ids = set(val_patients["patient_id"])
test_patient_ids = set(test_patients["patient_id"])


# Select all image rows belonging to training patients.
train_df = manifest_df[
    manifest_df["patient_id"].isin(train_patient_ids)
].copy()


# Select all image rows belonging to validation patients.
val_df = manifest_df[
    manifest_df["patient_id"].isin(val_patient_ids)
].copy()


# Select all image rows belonging to test patients.
test_df = manifest_df[
    manifest_df["patient_id"].isin(test_patient_ids)
].copy()

# Print the image counts
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))
 
# Check for patient overlap between splits.
train_val_overlap = train_patient_ids.intersection(val_patient_ids)
train_test_overlap = train_patient_ids.intersection(test_patient_ids)
val_test_overlap = val_patient_ids.intersection(test_patient_ids)

print("Train-validation patient overlap:", len(train_val_overlap))
print("Train-test patient overlap:", len(train_test_overlap))
print("Validation-test patient overlap:", len(val_test_overlap))


# Check class distribution in each image-level split.
print("\nTraining class distribution:")
print(train_df["classification"].value_counts())

print("\nValidation class distribution:")
print(val_df["classification"].value_counts())

print("\nTest class distribution:")
print(test_df["classification"].value_counts())

# Save the final image-level splits.
train_df.to_csv(
    "outputs/train_manifest.csv",
    index=False
)

val_df.to_csv(
    "outputs/validation_manifest.csv",
    index=False
)

test_df.to_csv(
    "outputs/test_manifest.csv",
    index=False
)

print("\nSplit files saved successfully.")
