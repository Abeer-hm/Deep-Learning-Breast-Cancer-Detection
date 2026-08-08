# This script reads the CMMD clinical metadata and DICOM files
# extracts patient ID and breast laterality from each image
# and checks whether each mammogram has a corresponding breast-level clinical record

# Import the necessary libraries
from pathlib import Path
import pandas as pd
import pydicom


# Path to the clinical Excel file
clinical_file = Path(
    "data/CMMD/clinical data/CMMD_clinicaldata_revision.xlsx"
)

# Path to the folder containing all DICOM images
dicom_root = Path(
    "data/CMMD/manifest-1616439774456/CMMD"
)

# Read the clinical data
clinical_df = pd.read_excel(clinical_file)

# Create a set of valid clinical combinations:
# patient ID + breast side
clinical_keys = set(
    zip(
        clinical_df["ID1"].astype(str).str.strip(),
        clinical_df["LeftRight"].astype(str).str.strip()
    )
)


# Store the patient ID and breast side found in each DICOM image
dicom_records = []

for file_path in dicom_root.rglob("*.dcm"):

    # Read only the DICOM metadata, not the image pixels
    dataset = pydicom.dcmread(
        file_path,
        stop_before_pixels=True
    )

    # Extract patient ID and image laterality
    patient_id = str(
        dataset.get("PatientID", "")
    ).strip()

    breast_side = str(
        dataset.get("ImageLaterality", "")
    ).strip()

    # Save the image information
    dicom_records.append(
        {
            "image_path": str(file_path),
            "patient_id": patient_id,
            "breast_side": breast_side,
        }
    )


# Convert the DICOM records into a table
dicom_df = pd.DataFrame(dicom_records)


# Check whether each DICOM image matches a clinical record
dicom_df["clinical_match"] = dicom_df.apply(
    lambda row: (
        row["patient_id"],
        row["breast_side"]
    ) in clinical_keys,
    axis=1
)

# Print the matching results
print("Total DICOM images:", len(dicom_df))
print("Images matched by patient ID and breast side:",
    dicom_df["clinical_match"].sum()
)
print("Images without a clinical breast-level match:",
    (~dicom_df["clinical_match"]).sum()
)

# Create a table containing only unmatched images
unmatched_df = dicom_df[
    dicom_df["clinical_match"] == False
].copy()

# Save the full result: matched and unmatched images
dicom_df.to_csv(
    "outputs/breast_level_matching_all.csv",
    index=False
)

# Save only the unmatched images
unmatched_df.to_csv(
    "outputs/unmatched_breast_level_images.csv",
    index=False
)

print("\nSaved full matching table to:")
print("outputs/breast_level_matching_all.csv")

print("\nSaved unmatched images to:")
print("outputs/unmatched_breast_level_images.csv")
