"""
Create a grouped bar chart comparing test performance across the seven input resolutions.

This script stores the test-set performance metrics for the seven
input resolutions and creates a grouped bar chart comparing AUC, accuracy,
sensitivity, specificity, precision, and F1-score.

The results table is also saved as a CSV file.

"""


# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Test-set performance of the selected checkpoint for each input resolution
data = {
    "Resolution": [
        "224x224",
        "299x299",
        "448x448",
        "512x512",
        "598x598",
        "800x800",
        "1024x1024"
    ],
    "AUC": [
        0.70467, 0.74493, 0.71927, 0.76403,
        0.71585, 0.71454, 0.71877
    ],
    "Accuracy": [
        0.63751, 0.64286, 0.66071, 0.70714,
        0.67143, 0.67679, 0.65893
    ],
    "Sensitivity": [
        0.65404, 0.59091, 0.64394, 0.71212,
        0.67677, 0.69949, 0.63384
    ],
    "Specificity": [
        0.59756, 0.76829, 0.70122, 0.69512,
        0.65854, 0.62195, 0.71951
    ],
    "Precision": [
        0.79692, 0.86029, 0.83882, 0.84940,
        0.82716, 0.81711, 0.84512
    ],
    "F1-score": [
        0.71845, 0.70060, 0.72857, 0.77473,
        0.74444, 0.75374, 0.72439
    ]
}

# Convert the test performance results into  DataFrame
df = pd.DataFrame(data)

# Save the test performance results as a CSV file 
df.to_csv("selected_epoch_resolution_evaluation_metrics.csv", index=False)

# Define the evaluation metrics included in the grouped comparison
metrics = [
    "AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1-score"
]

# Create x-axis positions for the seven input resolutions
x = np.arange(len(df["Resolution"]))

# Store the number of evaluation metrics
number_of_metrics = len(metrics)

# Set the width of each bar
bar_width = 0.13

# Create the grouped bar chart 
plt.figure(figsize=(14, 7))

# Plot one set of bars for each evaluation metric.
for i, metric in enumerate(metrics):

    # Calculate the position of each metric within the resolution group
    bar_positions = (
        x - ((number_of_metrics - 1) * bar_width / 2)
        + i * bar_width
    )
    # Plot the metrics values across the seven input resolutions 
    plt.bar(
        bar_positions,
        df[metric],
        width=bar_width,
        label=metric
    )

# Add labels to the axes
plt.xlabel("Input image resolution", fontsize=12)
plt.ylabel("Metric value", fontsize=12)

# Add a title describing the comparison
plt.title(
    "Comparison of Model Performance Across Input Resolutions",
    fontsize=14
)

# Display the seven input resolutions on the x-axis
plt.xticks(
    x,
    df["Resolution"],
    rotation=45,
    ha="right"
)

 # Set the y-axis range to improve comparison between metric values 
plt.ylim(0.50, 0.90)

# Add a legand identify the evaluation metrics
plt.legend(
    title="Evaluation metric",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

# Add horizontal grid lines to improve readability
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Adjust the figure layout to avoid overlapping between the figure elements
plt.tight_layout()

# Save the grouped bar charts as PNG image
plt.savefig(
    "resolution_metrics_grouped_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)
# Close the figure after saving
plt.close()

# Confirm the saved output files
print("CSV saved as: selected_epoch_resolution_metrics.csv")
print(
    "Bar chart saved as: "
    "resolution_metrics_grouped_bar_chart.png"
)
