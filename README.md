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
The dataset itself is not included in this repository 
## Code
The repository contains scripts for:
- creating breast level matching
- creating the labelled image manifest
- patient-level train, validation and test splitting
- CMMD preprocessing and data loading
-  Class-weight calculation
- ResNet50 model creation
- model training
- independent test evaluation
- calculation of validation mean and standard deviation across random seeds
- plotting performance metrics and confusion matrices
## Training settings 
- Model:ResNet50 pretrained on ImageNet
- Transfer learning: full fine-tuning
- Optimizer: Adam
- Learning rate:0.0001
- Loss function: weighted CrossEntropy
- Epochs:20
- Random seeds:42,123,2026
- Data split:70% training, 15% validation and 15% test at patient level 
