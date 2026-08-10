
"""
Create confusion matrix figures for the ResNet50 experiments.

This script reads the confusion-matrix counts for each evaluated
input resolution and caclulate the percentages within each class . Each figure
shows both the raws counts and the percentage for benign and malignant mammograms.
"""
# Import the necessary libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CSV file containing the TN, FP, FN and TP vlaues for each input resolution

INPUT_FILE = Path("outputs/selected_confusion_matrix_values.csv")

#  Create folder to save the confuison matrix figures
OUTPUT_FOLDER = Path("outputs/selected_confusion_matrices")

# Define the main function that creates the confusion matrix figures
def main():
    # Read the confusion matrix values from the CSV file
    df = pd.read_csv(INPUT_FILE)

    # Remove any space from the column names
    df.columns = df.columns.str.strip()

    # Define the columns that must be present in the input CSV file
    required_columns = [
        "Resolution",
        "Seed",
        "Epoch",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    # Check whether any required columns are missing
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
     # Stop the script and report an error if required columns are missing
    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

     # Specify the columns that should contain numeric values
    numeric_columns = required_columns

     # Convert the required columns to numeric values
    
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

     # Process one selected resolution at a time
    for _, row in df.iterrows():

        # Read the input resolution, random seed and selected epoch
        resolution = int(row["Resolution"])
        seed = int(row["Seed"])
        epoch = int(row["Epoch"])

        # Read the four confusion-matrix counts
        tn = int(row["TN"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tp = int(row["TP"])

        # Calculate the total number of images in each actual class
        count_matrix = np.array([
            [tn, fp],
            [fn, tp],
        ])

        # Calculate the total number of images in each actual class 
        row_totals = count_matrix.sum(axis=1, keepdims=True)

        # Convert the raw counts into perecntage 
        percentage_matrix = (
            count_matrix / row_totals
        ) * 100

         # Create a figure for the confusion matrix
        fig, ax = plt.subplots(figsize=(7, 6))

         # Dispaly the percentage matrix as an image 
        image = ax.imshow(
            percentage_matrix,
            vmin=0,
            vmax=100,
        )

         # Set the x and y-axis positions for the two classes
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        # Label the predicted classes on the x-axis
        ax.set_xticklabels(
            ["Benign", "Malignant"],
            fontsize=12,
        )

         # Label the actual classes on the y-axis
        ax.set_yticklabels(
            ["Benign", "Malignant"],
            fontsize=12,
        )

         # Label the horizontial axis as the predicted class
        ax.set_xlabel(
            "Predicted class",
            fontsize=13,
        )

          # Label the vertical axis as the actual class
        ax.set_ylabel(
            "Actual class",
            fontsize=13,
        )

         # Add the input resolution , random seed and  selected epoch to the figure title 
        ax.set_title(
            f"Confusion Matrix at {resolution} × {resolution}\n"
            f"Seed {seed}, Epoch {epoch}",
            fontsize=14,
        )
        # Add the raw count and percentage inside each matrix cell
        for matrix_row in range(2):
            for column in range(2):
                # Get the raw number of cases in the current cell
                count = count_matrix[matrix_row, column]
                # Get the corresponding percentage for each row count
                percentage = percentage_matrix[
                    matrix_row,
                    column
                ]

                # Display both values in the matrix cell
                ax.text(
                    column,
                    matrix_row,
                    f"{count}\n({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=14,
                )

        # Add a colour bar representing the row count and the corresponding perecntage
        colorbar = fig.colorbar(image, ax=ax)
        # Label the colour bar
        colorbar.set_label(
            "Percentage of actual class",
            fontsize=12,
        )
        # Adjust the space to prevent the overlapping between figure elements
        plt.tight_layout()
         # Create an output filename for each resolution 
        output_file = (
            OUTPUT_FOLDER
            / f"confusion_matrix_{resolution}.png"
        )
         # Save the figure as PNG image
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )
         # Close the figure
        plt.close(fig)

        print(f"Saved: {output_file}")

# Run the main function when the script is directly executed
if __name__ == "__main__":
    main()
