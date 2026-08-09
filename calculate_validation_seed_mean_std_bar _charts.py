"""
Calculate the mean of validation performance and variablilty across random seeds 

This script summarises the validation results for ResNet50 across seven input resolutions. for each resolution , the mean and   
standard deviation of AUC , accuracy, sensitivity, specificity, and F1-score are calculated across the three random seeds(42, 123, 2026)
The script also generates a grouped bar chart showing the mean of validation metrics with standard deviation error bars across input resolutions.
"""
# Import the necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Validation results from the selected epoch for each input resolution and random seed
data = {
    "Resolution": [
        "224x224", "224x224", "224x224",
        "299x299", "299x299", "299x299",
        "448x448", "448x448", "448x448",
        "512x512", "512x512", "512x512",
        "598x598", "598x598", "598x598",
        "800x800", "800x800", "800x800",
        "1024x1024", "1024x1024", "1024x1024"
    ],

    "Random seed": [
        42, 123, 2026,
        42, 123, 2026,
        42, 123, 2026,
        42, 123, 2026,
        42, 123, 2026,
        42, 123, 2026,
        42, 123, 2026
    ],

    "Selected epoch": [
        16, 4, 3,
        5, 5, 16,
        4, 7, 8,
        8, 19, 18,
        18, 6, 8,
        6, 7, 6,
        5, 5, 15
    ],

    "Validation AUC": [
        0.70427, 0.71838, 0.71905,
        0.73595, 0.75673, 0.72438,
        0.73144, 0.72484, 0.73029,
        0.77494, 0.75985, 0.74725,
        0.80355, 0.75587, 0.75481,
        0.75784, 0.77201, 0.77666,
        0.78283, 0.79363, 0.76520
    ],

   "Validation Accuracy": [
    0.65054, 0.65950, 0.66129,
    0.65950, 0.67025, 0.67384,
    0.67921, 0.66129, 0.66128,
    0.72043, 0.67384, 0.67742,
    0.73477, 0.68100, 0.69543,
    0.71505, 0.68817, 0.71326,
    0.71326, 0.71685, 0.71147
    ],

    "Sensitivity": [
        0.66327, 0.67345, 0.63776,
        0.70663, 0.63010, 0.69898,
        0.68622, 0.64796, 0.64796,
        0.75255, 0.63520, 0.69643,
        0.74745, 0.65561, 0.69388,
        0.78061, 0.66327, 0.73469,
        0.72704, 0.68878, 0.74745
    ],

    "Specificity": [
        0.62048, 0.62651, 0.71687,
        0.59639, 0.76506, 0.61446,
        0.66265, 0.69277, 0.69277,
        0.64458, 0.76506, 0.63253,
        0.70482, 0.74096, 0.69879,
        0.56025, 0.74699, 0.66265,
        0.68072, 0.78313, 0.62651
    ],

    "F1-score": [
        0.72724, 0.73538, 0.72569,
        0.75272, 0.72861, 0.75069,
        0.75035, 0.72884, 0.72884,
        0.79089, 0.76506, 0.75207,
        0.79839, 0.74277, 0.76190,
        0.79377, 0.74928, 0.78261,
        0.78082, 0.77364, 0.78441
    ]
}

df = pd.DataFrame(data)

# Ensure resolutions appear in the correct order
resolution_order = [
    "224x224",
    "299x299",
    "448x448",
    "512x512",
    "598x598",
    "800x800",
    "1024x1024"
]

df["Resolution"] = pd.Categorical(
    df["Resolution"],
    categories=resolution_order,
    ordered=True
)

metrics = [
    "Validation AUC",
    "Validation Accuracy",
    "Sensitivity",
    "Specificity",
    "F1-score"
]

# Calculate mean and standard deviation across seeds
summary = (
    df.groupby("Resolution")[metrics]
    .agg(["mean", "std"])
    .reset_index()
)

# Create simpler column names
summary.columns = [
    "Resolution",
    "AUC_mean",
    "AUC_std",
    "Accuracy_mean",
    "Accuracy_std",
    "Sensitivity_mean",
    "Sensitivity_std",
    "Specificity_mean",
    "Specificity_std",
    "F1-score_mean",
    "F1-score_std"
]

# Round only for the saved summary table
summary_rounded = summary.copy()

numeric_columns = summary_rounded.select_dtypes(include="number").columns
summary_rounded[numeric_columns] = (
    summary_rounded[numeric_columns].round(5)
)

summary_rounded.to_csv(
    "validation_metrics_mean_std_by_resolution.csv",
    index=False
)

print("\nMean and standard deviation across random seeds:\n")
print(summary_rounded.to_string(index=False))

# Create output directory
output_folder = "validation_mean_std_bar_charts"
os.makedirs(output_folder, exist_ok=True)

plot_settings = {
    "Validation AUC": ("AUC_mean", "AUC_std", "Validation AUC"),
    "Validation Accuracy": (
       "Accuracy_mean",
       "Accuracy_std",
       "Validation Accuracy"
     ),
    "Sensitivity": (
        "Sensitivity_mean",
        "Sensitivity_std",
        "Sensitivity"
    ),
    "Specificity": (
        "Specificity_mean",
        "Specificity_std",
        "Specificity"
    ),
    "F1-score": (
        "F1-score_mean",
        "F1-score_std",
        "F1-score"
    )

    } 


# Positions for the resolution groups
x = np.arange(len(summary["Resolution"]))

number_of_metrics = len(plot_settings)
bar_width = 0.16

# Create the grouped bar chart
plt.figure(figsize=(16, 8))

for i, (metric_name, columns) in enumerate(plot_settings.items()):

    mean_column, std_column, _ = columns

    positions = (
        x
        - ((number_of_metrics - 1) * bar_width / 2)
        + i * bar_width
    )

    bars = plt.bar(
        positions,
        summary[mean_column],
        width=bar_width,
        yerr=summary[std_column],
        capsize=5,
        label=metric_name
    )

plt.xlabel(
    "Input image resolution",
    fontsize=17
)

plt.ylabel(
    "Mean validation metric value",
    fontsize=17
)

plt.title(
    "Mean Validation Metrics Across Input Resolutions",
    fontsize=20,
    fontweight="bold",
    pad=15
)

plt.xticks(
    x,
    summary["Resolution"].astype(str),
    rotation=45,
    ha="right",
    fontsize=15
)

plt.yticks(fontsize=15)

plt.ylim(0, 1)

plt.legend(
    title="Validation metric",
    fontsize=14,
    title_fontsize=15,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

output_path = os.path.join(
    output_folder,
    "validation_metrics_mean_std_grouped_bar_chart.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


