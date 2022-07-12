import torch
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils.dataset_loader.svhn import SVHN
from utils.dataset_loader.synth import SYNTH
from utils.dataset_loader.mnistm import MNISTM
from utils.cnn_feature_extraction import extract_features_cnn
from utils.utils import load_checkpoint, checkpoint, classification, print_classifier_performance, pca_n_components, save_misclassified_samples, plot_results
from utils.hog_feature_extraction import extract_features_hog

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SAMPLES_PER_CLASS = 500 #max 800
LOAD_CHECKPOINT = False
FEATURE_EXTRACTOR = 'hog' #hog or cnn

if not LOAD_CHECKPOINT:
    # --- Load and preprocessing SVHN dataset ---
    print('- Loading SVHN Dataset...')
    svhn_train = SVHN(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    svhn_test = SVHN(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    # -----------------
    
    # --- Load and preprocessing MNIST-M dataset ---
    print('- Loading MNIST-M Dataset...')
    mnistm_train = MNISTM(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    mnistm_test = MNISTM(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    # -----------------
    
    # --- Load and preprocessing SYNTH dataset ---
    print('- Loading SYNTH Dataset...')
    synth_train = SYNTH(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    synth_test = SYNTH(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    # -----------------

    if FEATURE_EXTRACTOR == 'cnn':
        # --- PyTorch SVHN Dataloader objects ---
        svhn_train_dataloader = torch.utils.data.DataLoader(svhn_train, batch_size=128, shuffle=True)
        svhn_test_dataloader = torch.utils.data.DataLoader(svhn_test, batch_size=128, shuffle=True)
        # -----------------

        # --- PyTorch MNIST-M Dataloader objects ---
        mnistm_train_dataloader = torch.utils.data.DataLoader(mnistm_train, batch_size=128, shuffle=True)
        mnistm_test_dataloader = torch.utils.data.DataLoader(mnistm_test, batch_size=128, shuffle=True)
        # -----------------

        # --- PyTorch SYNTH Dataloader objects ---
        synth_train_dataloader = torch.utils.data.DataLoader(synth_train, batch_size=128, shuffle=True)
        synth_test_dataloader = torch.utils.data.DataLoader(synth_test, batch_size=128, shuffle=True)
        # -----------------

        # --- SVHN Feature extraction via ResNet34 ---
        print('- SVHN extracting features via ResNet34...')
        _, svhn_feature_train, svhn_y_train = extract_features_cnn(svhn_train_dataloader)
        svhn_images_test, svhn_feature_test, svhn_y_test = extract_features_cnn(svhn_test_dataloader)
        # -----------------

        # --- MNIST-M Feature extraction via ResNet34 ---
        print('- MNIST-M extracting features via ResNet34...')
        _, mnistm_feature_train, mnistm_y_train = extract_features_cnn(mnistm_train_dataloader)
        mnistm_images_test, mnistm_feature_test, mnistm_y_test = extract_features_cnn(mnistm_test_dataloader)
        # -----------------

        # --- SYNTH Feature extraction via ResNet34 ---
        print('- SYNTH extracting features via ResNet34...')
        _, synth_feature_train, synth_y_train = extract_features_cnn(synth_train_dataloader)
        synth_images_test, synth_feature_test, synth_y_test = extract_features_cnn(synth_test_dataloader)
        # -----------------
    
    elif FEATURE_EXTRACTOR == 'hog':

        # --- Escamotage to get normalized images from dataset (used to show missclassified samples) ---
        svhn_images_test = []
        for i in range(len(svhn_test)): 
            svhn_images_test.append(svhn_test[i][0].numpy())
        # -----------------

        # --- SVHN Feature extraction via HOG ---
        print('- SVHN extracting features via HOG...')
        svhn_feature_train, svhn_y_train = extract_features_hog(svhn_train.samples, 'svhn'), svhn_train.labels
        svhn_feature_test, svhn_y_test = extract_features_hog(svhn_test.samples, 'svhn'), svhn_test.labels
        svhn_y_test = np.squeeze(svhn_y_test)
        svhn_y_train = np.squeeze(svhn_y_train)
        # -----------------

        # --- Escamotage to get normalized images from dataset (used to show missclassified samples) ---
        mnistm_images_test = []
        for i in range(len(mnistm_test)): 
            mnistm_images_test.append(mnistm_test[i][0].numpy())
        # -----------------
        
        # --- MNIST-M Feature extraction via HOG ---
        print('- MNIST-M extracting features via HOG...')
        mnistm_feature_train, mnistm_y_train = extract_features_hog(mnistm_train.samples, 'mnistm'), mnistm_train.labels
        mnistm_feature_test, mnistm_y_test = extract_features_hog(mnistm_test.samples, 'mnistm'), mnistm_test.labels
        mnistm_y_test = np.squeeze(mnistm_y_test)
        mnistm_y_train = np.squeeze(mnistm_y_train)
        # -----------------

        # --- Escamotage to get normalized images from dataset (used to show missclassified samples) ---
        synth_images_test = []
        for i in range(len(synth_test)): 
            synth_images_test.append(synth_test[i][0].numpy())
        # -----------------

        # --- SYNTH Feature extraction via HOG ---
        print('- SYNTH extracting features via HOG...')
        synth_feature_train, synth_y_train = extract_features_hog(synth_train.samples, 'synth'), synth_train.labels
        synth_feature_test, synth_y_test = extract_features_hog(synth_test.samples, 'synth'), synth_test.labels
        synth_y_test = np.squeeze(synth_y_test)
        synth_y_train = np.squeeze(synth_y_train)
        # -----------------

    # --- Checkpoint (used during project testing phase to speed-up execution) ---
    print('- Checkpoint...')
    checkpoint(feature_train=svhn_feature_train, feature_test=svhn_feature_test, y_train=svhn_y_train, y_test=svhn_y_test, dataset_name='svhn')
    checkpoint(feature_train=mnistm_feature_train, feature_test=mnistm_feature_test, y_train=mnistm_y_train, y_test=mnistm_y_test, dataset_name='mnistm')
    checkpoint(feature_train=synth_feature_train, feature_test=synth_feature_test, y_train=synth_y_train, y_test=synth_y_test, dataset_name='synth')
    # -----------------

# --- Load pre-saved extracted feature ---
if LOAD_CHECKPOINT:
    print('- Loading checkpoint...')
    svhn_feature_train, svhn_feature_test, svhn_y_train, svhn_y_test = load_checkpoint(dataset_name='svhn')
    mnistm_feature_train, mnistm_feature_test, mnistm_y_train, mnistm_y_test = load_checkpoint(dataset_name='mnistm')
    synth_feature_train, synth_feature_test, synth_y_train, synth_y_test = load_checkpoint(dataset_name='synth')
    #print(svhn_feature_train.shape, svhn_feature_test.shape, len(svhn_y_train), len(svhn_y_test))
    #print(mnistm_feature_train.shape, mnistm_feature_test.shape, len(mnistm_y_train), len(mnistm_y_test))
    #print(synth_feature_train.shape, synth_feature_test.shape, len(synth_y_train), len(synth_y_test))
# -----------------

# --- Transform extracted features by scaling each feature between 0 and 1 (x=(x-min/max-min)) ---
print('- Scaling extracted features...')
scaler = MinMaxScaler() 
svhn_scaled_feature_train = scaler.fit_transform(svhn_feature_train)
svhn_scaled_feature_test = scaler.fit_transform(svhn_feature_test)
mnistm_scaled_feature_train = scaler.fit_transform(mnistm_feature_train)
mnistm_scaled_feature_test = scaler.fit_transform(mnistm_feature_test)
synth_scaled_feature_train = scaler.fit_transform(synth_feature_train)
synth_scaled_feature_test = scaler.fit_transform(synth_feature_test)
# -----------------

# --- Check n_component used to get a variance of 0.9 ---
#obtain n_component to get a variance of 0.9
#n_components=pca_n_components(variance=0.9, feature=svhn_scaled_feature_train)
#print('ncomp:', n_components)
# -----------------

# --- Feature reductors used ---
feature_reductors = [
    ('pca', PCA(n_components=0.9, random_state=RANDOM_STATE)),
    ('lda', LinearDiscriminantAnalysis())
    ]
# -----------------

# --- Type of classifiers used ---
classifiers = [
    ('svm', SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE, decision_function_shape='ovr')),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ]
# -----------------

# --- Training set used ---
datasets_train = [
    ('svhn', svhn_scaled_feature_train, svhn_y_train),
    ('mnistm', mnistm_scaled_feature_train, mnistm_y_train),
    ('synth', synth_scaled_feature_train, synth_y_train),
]
# -----------------

# --- Testing set used ---
datasets_test = [
    ('svhn', svhn_scaled_feature_test, svhn_y_test),
    ('mnistm', mnistm_scaled_feature_test, mnistm_y_test),
    ('synth', synth_scaled_feature_test, synth_y_test),
]
# -----------------

results = []
progress = 1
total_progress = len(classifiers)*len(feature_reductors)*len(datasets_train)*len(datasets_test)

print(f'- Start Classification: {total_progress} combinations...')
for classifier in classifiers:
    for feature_reductor in feature_reductors:
        for dataset_train in datasets_train:
            for dataset_test in datasets_test:

                print(f'\t{progress}/{total_progress}: Combination({feature_reductor[0]}_{classifier[0]}) -> Datataset(train:{dataset_train[0]}; test:{dataset_test[0]})')
                
                steps = [
                    ('fred', feature_reductor[1]),
                    ('clf', classifier[1])
                ]
                pipeline = Pipeline(steps)
                
                current_step = dataset_train[0] + '_' + dataset_test[0] + '_' + feature_reductor[0] + '_' + classifier[0]
                accuracy, precision, recall, misclassified_idx, y_predicted, elapsed_time = classification(pipeline, x_train=dataset_train[1], x_test=dataset_test[1], y_train=dataset_train[2], y_test=dataset_test[2], step = current_step)
                
                print_classifier_performance(accuracy, precision, recall, elapsed_time)

                tmp_result = {'source':dataset_train[0],'target':dataset_test[0],'reductor':feature_reductor[0], 'classifier':classifier[0],'accuracy':round(accuracy,2), 'precision':round(precision,2), 'recall':round(recall,2)}
                results.append(tmp_result)

                # --- Save/Show Random misclassified sample (requires that LOAD_CHECKPOINT=False) ---
                if(LOAD_CHECKPOINT == False):
                    if dataset_test[0] == 'svhn':
                        x_test,y_test = svhn_images_test, svhn_y_test
                    elif dataset_test[0] == 'mnistm':
                        x_test,y_test = mnistm_images_test, mnistm_y_test
                    else:
                        x_test,y_test = synth_images_test, synth_y_test 
                    save_misclassified_samples(x_test=x_test, y_test=y_test, misclassified_idx=misclassified_idx, y_predicted=y_predicted, step=current_step)
                # -----------------
                progress+=1

# --- Save and plots Accuracy results ---
with open('results/results_dict.json', 'w') as fout:
    json.dump(results, fout)
plot_results(results)
# -----------------