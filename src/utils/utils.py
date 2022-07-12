import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
from torchvision import transforms
import random
import time
import json
import torch

#VERBOSE -> show plots and print
VERBOSE = False

def checkpoint(feature_train, feature_test, y_train, y_test, dataset_name):
    """
    Save extracted features
    """

    np.save('data/working_dataset_files/'+ dataset_name +'_feature_train.npy', feature_train)
    np.save('data/working_dataset_files/'+ dataset_name +'_y_train.npy', y_train)
    np.save('data/working_dataset_files/'+ dataset_name +'_feature_test.npy', feature_test)
    np.save('data/working_dataset_files/'+ dataset_name +'_y_test.npy', y_test)

def load_checkpoint(dataset_name):
    """
    Load checkpoint (pre-extracted features)
    """

    feature_train = np.load('data/working_dataset_files/'+ dataset_name +'_feature_train.npy')
    y_train = np.load('data/working_dataset_files/'+ dataset_name +'_y_train.npy')
    feature_test = np.load('data/working_dataset_files/'+ dataset_name +'_feature_test.npy')
    y_test = np.load('data/working_dataset_files/'+ dataset_name +'_y_test.npy')

    return feature_train, feature_test, y_train, y_test

def classification(pipe, x_train, x_test, y_train, y_test, step):
    """
    Classification phase, save confusion matrix
    """

    start_time = time.time()

    pipe.fit(x_train, y_train)
    y_predicted = pipe.predict(x_test)
    misclassified_idx = np.where(y_test!=y_predicted)[0]

    # Display confusion matrix using sklearn
    ConfusionMatrixDisplay.from_predictions( y_test, y_predicted)
    plt.xlabel('Predicted label', fontweight = 'bold', fontsize = 15)
    plt.ylabel('Real label', fontweight ='bold', fontsize = 15)
    plt.title(step, fontweight ='bold', fontsize = 15)
    step ='cm_' + step + '.png'
    final_path = path.join('results/confusion_matrix', step)
    plt.savefig(final_path)
    if VERBOSE:
        plt.show()
    plt.close()
    
    # Calculate the accuracy, precision, recall (of the whole classifier) and elapsed time
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='macro') #macro -> means of mean
    recall = recall_score(y_test, y_predicted, average = 'macro') #macro -> means of mean
    elapsed_time = round(time.time() - start_time,2)

    return accuracy*100, precision, recall, misclassified_idx, y_predicted, elapsed_time

def print_classifier_performance(accuracy, precision, recall, elapsed_time):
    """
    Print classifier performances
    """

    print('\t\tAccuracy: ' + '{0:.2f}'.format(accuracy) + '%' 
          + '; Precision: ' + '{0:.2f}'.format(precision)
          + '; Recall: ' + '{0:.2f}'.format(recall)
          + '; Time: ' + '{0:.2f}'.format(elapsed_time) +'s')

def pca_n_components(variance, feature):
    """
    Retrieve n_components to reach a specif variance with PCA
    """

    pca = PCA(n_components=variance)
    pca.fit(feature)

    if VERBOSE:
        # Plot the explained variance ratio
        plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Explained variance by number of components')
        plt.ylabel('Cumulative explained variance')
        plt.xlabel('Nr. of principal components')
        plt.show()

    n_components = len(pca.explained_variance_ratio_)
    
    return n_components

def save_misclassified_samples(x_test, y_test, misclassified_idx, y_predicted, step):
    """
    Save/Show 20 random misclassified samples with predicted value and ground truth
    """

    figure = plt.figure(figsize=(8,8))
    figure.suptitle(step, fontweight ='bold', fontsize = 15)
    cols, rows = 5, 4
    for i in range(1, cols * rows + 1):
        sample_idx = random.choice(misclassified_idx)
        img = torch.tensor(x_test[sample_idx])
        # Unnormalize, normalized images (back to original colors)
        img_invTransforms = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
        img = img_invTransforms(img)
        img = torch.permute(img, (1,2,0))
        figure.add_subplot(rows, cols, i)
        plt.title(f'pred:{y_predicted[sample_idx]}' 
                    f' gt:{y_test[sample_idx]}') 
        plt.axis('off')
 
        plt.imshow(img)
    step ='mc_' + step + '.png'
    final_path = path.join('results/missclassified', step)
    plt.savefig(final_path)
    if VERBOSE:
        plt.show()
    plt.close()

def plot_results(results = None):
    """
    Save/Show multiple bar plot (accuracy performance of each classifier combination)
    """

    #for debugging
    if(results is None):
        results = json.load(open('results/results_dict.json','r'))
        
    df = pd.DataFrame(results)
    
    barWidth = 0.2
    fig = plt.subplots(figsize =(18, 8))

    # --- Dataframe preprocessing ---
    df['reductor_classifier'] = df['reductor'] + '_' + df['classifier']
    df['source_target'] = df['source'] + '_' + df['target']
    df.drop(['reductor', 'classifier',  'source', 'target'], inplace=True, axis=1)
    df = df[['source_target', 'reductor_classifier', 'accuracy', 'precision', 'recall']]
    print(df)
    df.to_csv('results/results_dict.csv', index = False, header=True, sep=';')
    df.drop(['precision', 'recall'], inplace=True, axis=1)
    # -----------------
    
    sources_targets = [
        'svhn_svhn',
        'svhn_mnistm',
        'svhn_synth',
        'mnistm_svhn',
        'mnistm_mnistm',
        'mnistm_synth',
        'synth_svhn',
        'synth_mnistm',
        'synth_synth',   
    ]
    
    reductor_classifier = [
        'pca_svm',
        'lda_svm',
        'pca_knn',
        'lda_knn',
    ]

    pca_svm = df.loc[(df['reductor_classifier'] == 'pca_svm'), 'accuracy'].tolist()
    lda_svm = df.loc[(df['reductor_classifier'] == 'lda_svm'), 'accuracy'].tolist()
    pca_knn = df.loc[(df['reductor_classifier'] == 'pca_knn'), 'accuracy'].tolist()
    lda_knn = df.loc[(df['reductor_classifier'] == 'lda_knn'), 'accuracy'].tolist()

    ind = np.arange(len(pca_svm))
    br1 = ind
    plt.bar(br1, pca_svm, color ='#F9ED69', width = barWidth, edgecolor ='grey', label ='pca_svm')
    br2 = ind + barWidth
    plt.bar(br2, lda_svm, color ='#F08A5D', width = barWidth, edgecolor ='grey', label ='lda_svm')
    br3 = ind + barWidth*2
    plt.bar(br3, pca_knn, color ='#B83B5E', width = barWidth, edgecolor ='grey', label ='pca_knn')
    br4 = ind + barWidth*3
    plt.bar(br4, lda_knn, color ='#6A2C70', width = barWidth, edgecolor ='grey', label ='lda_knn')

    # Adding Xticks
    plt.xlabel('Source_Target', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks(ind + barWidth, sources_targets)
    plt.title("Results",fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.savefig('results/accuracy/accuracy.png')
    plt.show()