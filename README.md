# CMMD-ResNet50-Resolution-study
Code for an MSc dissertation investigating the effect of input image resolution on ResNet50 performance for breast cancer classification using the CMMD mammography dataset.

## Study overview
This study evaluates how different mammogram input resolutions affect the performance of an ImageNet-pretrained ResNet50 model for binary classification of CMMD mammograms as benign and malignant.
Seven input resolutions were investigated:
224 x 224 ,
299 x 299 ,
448 x 448 ,
512 x 512 ,
598 x 598 ,
800 x 800 ,
1024 x 1024 ,

The model was trained using full fine-tuning, the Adam optimizer, class weighted CrossEntropy loss, and three random seeds.

## Dataset
The study uses the Chinese Mammography Database (CMMD)
The dataset itself is not included in this repository. 

 ## Requirements
 The code was developed in Python and requires the following packages:

- PyTorch
- torchvision
- pandas
- NumPy
- pydicom
- scikit-learn
- Matplotlib

## Code
The repository contains scripts for:
- creating breast level matching.
- creating the final labelled image manifest.
- patient-level training, validation and test splitting.
- CMMD image preprocessing and data loading.
-  Calculating class-weights from the training set.
- Creating and verifying ResNet50 model .
- Training and validation the ResNet50 model.
- Evaluating the selected model on the test set: the repository includes the 224 x 224 evaluation script as a representative example of the evaluation pipeline. the same evaluation procedures was applied to all seven input resolutions(224,299,448,512,598,800 and 1024 pixels) .For each resolution, the target size and checkpoint path were changed to correspond to the model checkpoint selected from the validation stage.
- calculating the validation performance mean and standard deviation across random seeds and presenting the results using bar charts with standard-deviation error bars
- plotting test performance metrics using bar chart .
- generating confusion matrix figures.

  ## Analysis Workflow
The scripts follow the main analysis workflow used in the dissertation:
1. Create breast-level matching between the DICOM images and clinical data.
2. Create the final labelled image manifest.
3. Create the patient-level training, validation, and test split.
4. Load and preprocess the CMMD mammograms.
5. Create the DataLoaders for the training, validation, and test sets.
6. Calculate class weights from the training set.
7. Create and verify the ResNet50 model.
8. Train and validate the model.
9. Evaluate the selected checkpoint on the independent test set.
10. Plot the test performance metrics across the seven input resolutions using bar charts.
11. Calculate the mean and standard deviation of validation performance across random seeds and presenting the results using bar charts with standard deviation error bars.
12. Generate the confusion matrix figures.

## Training settings 
- Model:ResNet50 pretrained on ImageNet
- Transfer learning: full fine-tuning
- Optimizer: Adam
- Learning rate:0.0001
- Loss function: weighted CrossEntropy
- Epochs:20
- Random seeds:42,123,2026
- Data split:70% training, 15% validation and 15% test at patient level
- Input resolutions:224,299,448,512,598,800 and 1024 pixels
- Hardware: CUDA-enabled GPU
  
  ## Evaluation
The selected model checkpoints were evaluated on the test set.
Performance was assessed using:
- AUC
- accuracy
- sensitivity
- specificity
- precision
- F1-score
- confusion matrix

## Reproducibility Notes
The CMMD dataset is not included in this repository.
The same overall training and evaluation procedure was applied across the seven input resolutions. 
The experiments reported in the dissertation were conducted using a CUDA-enabled GPU.
