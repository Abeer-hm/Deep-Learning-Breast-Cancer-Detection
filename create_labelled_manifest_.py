 """
Create the final labelled image manifest for the CMMD dataset
This scripts combines the breast-level DICOM matching results with the clincial metadata using patient ID and breast laterality.
It retains only images with a valid clinical match and assigns the corresponding clinical information and classification label to each image.  
The resulting manifest is used for the patient level data splitting step.
"""
# Import the necessary libraries
from pathlib import Path
import pandas as pd

# 1. Define input and output file paths
# Clinical information and labels
clinical_file = Path(
    "data/CMMD/clinical data/CMMD_clinicaldata_revision.xlsx"
)

# Previous matching result containing all DICOM images
matching_file = Path(
    "outputs/breast_level_matching_all.csv"
)

# Final labelled image manifest
manifest_file = Path(
    "outputs/image_manifest_labelled.csv"
)

# Text summary of the result
summary_file = Path(
    "outputs/image_manifest_labelled_summary.txt"
)

# 2. Read the clinical data and matching results

clinical_df = pd.read_excel(clinical_file)
matching_df = pd.read_csv(matching_file)

# 3. Clean and rename the clinical matching columns
clinical_df["ID1"] = clinical_df["ID1"].astype(str).str.strip()
clinical_df["LeftRight"] = (
    clinical_df["LeftRight"].astype(str).str.strip()
)

clinical_df = clinical_df.rename(
    columns={
        "ID1": "patient_id",
        "LeftRight": "breast_side",
        "Age": "age",
    }
)

# 4. Keep only DICOM images with a clinical match
matched_df = matching_df[
    matching_df["clinical_match"] == True
].copy()

# 5. Add the clinical information to each matched image

manifest_df = matched_df.merge(
    clinical_df[
        [
            "patient_id",
            "breast_side",
            "age",
            "abnormality",
            "classification",
            "subtype",
        ]
    ],
    on=["patient_id", "breast_side"],
    how="left",
    validate="many_to_one",
)

# 6. Check that every retained image received a label

missing_labels = manifest_df["classification"].isna().sum()

if missing_labels > 0:
    raise ValueError(
        f"{missing_labels} matched images still have missing labels."
    )

# 7. Save the final labelled image manifest

manifest_df.to_csv(
    manifest_file,
    index=False
)

# 8. Save a short summary in the outputs folder

summary_text = (
    f"Total images in matching table: {len(matching_df)}\n"
    f"Labelled images retained: {len(manifest_df)}\n"
    f"Images excluded without breast-level labels: "
    f"{len(matching_df) - len(manifest_df)}\n"
    f"Unique patients retained: {manifest_df['patient_id'].nunique()}\n"
    f"Missing classification labels: {missing_labels}\n"
)

summary_file.write_text(summary_text)

print(summary_text)
print("Manifest saved to:", manifest_file)
print("Summary saved to:", summary_file)
