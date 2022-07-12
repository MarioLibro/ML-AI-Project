# ML&AI - Project (Domain Adaptation)
University project for the course of Machine Learning & Artificial Intelligence (2021/2022).

## Introduction
This project aims to provide a performance comparison between different combination of feature extractors, feature reductors and classifiers applying Domain Adpatation on different types of datasets containing images of digits.

The datasets used for the projects are: 
- SVHN: contains images taken from Google Street View representing house numbers.
  ![svhn](/src/results/dataset_samples/svhn.png)
- MNIST-M: is a variation of MNIST. Composed of colored digits embedded on random background.
  ![mnist-m](/src/results/dataset_samples/mnistm.png)
- SYNTH: contains synthetically generated images of digits.
  ![synth](/src/results/dataset_samples/synth.png)
  
All the dataset are stored in the folder `src/data/original_dataset_files`.

We performed a performance comparison between 72 different types of configurations, combining:
- 2x Features extractor (Resnet-34 and **H**istogram of **O**riented **G**radients)
- 2x Features reductor (**L**inear **D**iscriminant **A**nalysis and **P**rincipal **C**omponent **A**nalysis)
- 2x Classifers (**S**upport **V**ector **M**achine and **k**-**N**earest **N**eighbors)
- 3x Datasets (SVHN, MNIST-M and SYNTH)

## Install
- Install the python libraries dependencies inside the `src/requirements.txt` file.

- [Download](https://drive.google.com/file/d/1_MBIettKKF_RMmrWpeJJfSD3e3qUnHtI/view?usp=sharing) and unzip `datasets.zip` in `src/data/original_dataset_files`.

- Execute `src/main.py`.

## Set-Up
Code is ready to-run! 
It will automatically:
- Load datasets.
- Data preprocessing (class balanced with `N_SAMPLES_PER_CLASS=100`).
- Training/Testing set split (80/20).
- Features extraction with Resnet-34 or HOG (set `FEATURE_EXTRACTOR='cnn' or 'hog'` variable in `src/main.py` to decide which one to perform).
- Features Scaling.
- Features Reduction using PCA and LDA.
- Classification using 36 different types of configurations.
- Saves results.

## Results
After the execution of main.py all the results are saved in `src/results`. In particular:
- `src/results/accuracy`: contains a bar plot that comapres the accuracy reached for all the 36 different types of configuration.
- `src/results/confusion_matrix`: contains confusions matrix plots of all the 36 different types of configuration.
- `src/results/missclassified`: contains 20 random missclassified samples (predicted value/ground truth) for each different configuration.
- `src/results/dataset_samples`: contains 20 random samples for each dataset and 10 HOG extracted features visualization.
- `src/results/results_dict.csv`: contains evaluation metrics (accuracy, precision, recall) for all the 36 different types of configuration.

The results that we obtained are available in `./results`.
A detailed report is available in the folder `./report`, where we analyze the obtained results.
## Authors
@MarioLibro and @FilippoNevi
