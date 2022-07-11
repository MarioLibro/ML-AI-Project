# ML&AI - Project (Domain Adaptation)
University project for the course of Machine Learning & Artificial Intelligence (2021/2022).

## Introduction
This project aims to provide a performance comparison between different types of classifiers (Support Vector Machines and K-Nearest Neighbours), applying Domain Adpatation on different types of datasets containing images of digits.
The datasets used for the projects are: 
- SVHN: contains images taken from Google Street View representing house numbers.
  ![svhn](/src/results/dataset_samples/svhn.png)
- MNIST-M: is a variation of MNIST. Composed of colored digits embedded on random background.
  ![mnist-m](/src/results/dataset_samples/mnistm.png)
- SYNTH: contains synthetically generated images of digits.
  ![synth](/src/results/dataset_samples/synth.png)
  
All the dataset used are in the folder original_dataset_files, instead working_dataset_files is used to save the results obtained from Resnet-38 (used as a checkpoint).

This allows us to perform a performance comparison between 36 different types of configuration, combining:
- 1x Features extractor (Resnet-38)
- 2x Features reductor (LDA and PCA)
- 2x Classifers (SVM and KNN)
- 3x Datasets (SVHN, MNIST-M and SYNTH)

## Install
Install the python libraries dependencies inside the `src/requirements.txt` file.

Execute `main.py`.

## Set-Up
Code is ready to-run! 
It will automatically:
- Load datasets (class balanced with 500 samples per class)
- Data preprocessing 
- Training/Testing set split (80/20)
- Feature extraction with Resnet-38
- Feature Normalization
- Classification using 36 different types of configurations
- Save results

## Results
After the execution of main.py all the results are saved in `src/results`. In particular:
- `src/results/accuracy`: contains a bar plot that comapres the accuracy reached for all the 36 different types of configuration
- `src/results/confusion_matrix`: contains confusions matrix plots of all the 36 different types of configuration
- `src/results/missclassified`: contains 20 random missclassified samples (predicted value/ground truth) for each different configuration
- `src/results/dataset_samples`: contains 20 random samples for each dataset
- `src/results/results_dict.csv`: contains evaluation metrics (accuracy, precision, recall) for all the 36 different types of configuration

## Authors
@MarioLibro and @FilippoNevi
