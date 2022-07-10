import torch
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils.dataset_loader.svhn import SVHN
from utils.dataset_loader.synth import SYNTH
from utils.dataset_loader.mnistm import MNISTM

from utils.cnn_feature_extraction import extract_feature
from utils.utils import load_checkpoint, checkpoint, classification, print_classifier_performance, pca_n_components, show_misclassified, plot_results

#TODO mettili in variabili d'ambiente e toglili da argomenti classe svhn
TEST_SIZE = 0.2
N_SAMPLES_PER_CLASS = 100 #min ->1744
DATASET_READY = False

if not DATASET_READY:
    #load and preprocessing SVHN dataset
    svhn_train = SVHN(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    svhn_test = SVHN(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    print('SVHN dataset loaded')

    #load and preprocessing MNIST-M dataset
    mnistm_train = MNISTM(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    mnistm_test = MNISTM(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    print('MNISTM dataset loaded')

    #load and preprocessing SYNTH dataset
    synth_train = SYNTH(split='train', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    synth_test = SYNTH(split='test', test_size=TEST_SIZE, n_samples_per_class=N_SAMPLES_PER_CLASS)
    print('SYNTH dataset loaded')

    #PyTorch SVHN Dataloader objects
    svhn_train_dataloader = torch.utils.data.DataLoader(svhn_train, batch_size=128, shuffle=True)
    svhn_test_dataloader = torch.utils.data.DataLoader(svhn_test, batch_size=128, shuffle=True)

    #PyTorch MNIST-M Dataloader objects
    mnistm_train_dataloader = torch.utils.data.DataLoader(mnistm_train, batch_size=128, shuffle=True)
    mnistm_test_dataloader = torch.utils.data.DataLoader(mnistm_test, batch_size=128, shuffle=True)

    #PyTorch SYNTH Dataloader objects
    synth_train_dataloader = torch.utils.data.DataLoader(synth_train, batch_size=128, shuffle=True)
    synth_test_dataloader = torch.utils.data.DataLoader(synth_test, batch_size=128, shuffle=True)

    #SVHN Feature extraction via ImageNet
    svhn_images_train, svhn_feature_train, svhn_y_train = extract_feature(svhn_train_dataloader)
    svhn_images_test, svhn_feature_test, svhn_y_test = extract_feature(svhn_test_dataloader)
    print('SVHN feature extracted')

    #MNIST-M Feature extraction via ImageNet
    mnistm_images_train, mnistm_feature_train, mnistm_y_train = extract_feature(mnistm_train_dataloader)
    mnistm_images_test, mnistm_feature_test, mnistm_y_test = extract_feature(mnistm_test_dataloader)
    print('MNISTM feature extracted')

    #SYNTH Feature extraction via ImageNet
    synth_images_train, synth_feature_train, synth_y_train = extract_feature(synth_train_dataloader)
    synth_images_test, synth_feature_test, synth_y_test = extract_feature(synth_test_dataloader)
    print('SYNTH feature extracted')
    
    checkpoint(feature_train=svhn_feature_train, feature_test=svhn_feature_test, y_train=svhn_y_train, y_test=svhn_y_test, dataset_name='svhn')
    checkpoint(feature_train=mnistm_feature_train, feature_test=mnistm_feature_test, y_train=mnistm_y_train, y_test=mnistm_y_test, dataset_name='mnistm')
    checkpoint(feature_train=synth_feature_train, feature_test=synth_feature_test, y_train=synth_y_train, y_test=synth_y_test, dataset_name='synth')

svhn_feature_train, svhn_feature_test, svhn_y_train, svhn_y_test = load_checkpoint(dataset_name='svhn')
mnistm_feature_train, mnistm_feature_test, mnistm_y_train, mnistm_y_test = load_checkpoint(dataset_name='mnistm')
synth_feature_train, synth_feature_test, synth_y_train, synth_y_test = load_checkpoint(dataset_name='synth')

#print(svhn_feature_train.shape, svhn_feature_test.shape, len(svhn_y_train), len(svhn_y_test))
#print(mnistm_feature_train.shape, mnistm_feature_test.shape, len(mnistm_y_train), len(mnistm_y_test))
#print(synth_feature_train.shape, synth_feature_test.shape, len(synth_y_train), len(synth_y_test))

#data normalization -> shrink data to 0 and 1 (x-min/max-min)
scaler = MinMaxScaler() 
svhn_scaled_feature_train = scaler.fit_transform(svhn_feature_train)
svhn_scaled_feature_test = scaler.fit_transform(svhn_feature_test)
mnistm_scaled_feature_train = scaler.fit_transform(mnistm_feature_train)
mnistm_scaled_feature_test = scaler.fit_transform(mnistm_feature_test)
synth_scaled_feature_train = scaler.fit_transform(synth_feature_train)
synth_scaled_feature_test = scaler.fit_transform(synth_feature_test)

#obtain n_component to get a variance of 0.9 (100 samples per class -> 435 components)
#n_components=pca_n_components(variance=0.9, feature=svhn_scaled_feature_train)
#print('ncomp:', n_components)

feature_reductors = [
    ('pca', PCA(n_components=0.9, random_state=0)),
    ('lda', LinearDiscriminantAnalysis())
    ]
    
classifiers = [
    ('svm', SVC(kernel='rbf', gamma='scale', random_state=0, decision_function_shape='ovr')),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ]

datasets_train = [
    ('svhn', svhn_scaled_feature_train, svhn_y_train),
    ('mnistm', mnistm_scaled_feature_train, mnistm_y_train),
    ('synth', synth_scaled_feature_train, synth_y_train),
]
datasets_test = [
    ('svhn', svhn_scaled_feature_test, svhn_y_test),
    ('mnistm', mnistm_scaled_feature_test, mnistm_y_test),
    ('synth', synth_scaled_feature_test, synth_y_test),
]

results = []

progress=1
total_progress= len(classifiers)*len(feature_reductors)*len(datasets_train)*len(datasets_test)
for classifier in classifiers:
    for feature_reductor in feature_reductors:
        for dataset_train in datasets_train:
            for dataset_test in datasets_test:
                
                steps = [
                    ('fred', feature_reductor[1]),
                    ('clf', classifier[1])
                ]
                pipeline = Pipeline(steps)
                
                current_step = dataset_train[0] + '_' + dataset_test[0] + '_' + feature_reductor[0] + '_' + classifier[0]
                accuracy, precision, recall, misclassified_idx, y_predicted = classification(pipeline, x_train=dataset_train[1], x_test=dataset_test[1], y_train=dataset_train[2], y_test=dataset_test[2], step = current_step)
                
                print(f'progress -> {progress}/{total_progress}: Classifier({feature_reductor[0]};{classifier[0]}) -> (source:{dataset_train[0]};target:{dataset_test[0]})')
                #print_classifier_performance(accuracy, precision, recall)

                tmp = {'source':dataset_train[0],'target':dataset_test[0],'reductor':feature_reductor[0], 'classifier':classifier[0],'accuracy':'{0:.2f}'.format(accuracy), 'precision':'{0:.2f}'.format(precision), 'recall':'{0:.2f}'.format(recall)}
                results.append(tmp)

                #show misclassified samples (needs DATASET_READY = False)
                if(DATASET_READY == False):
                    if dataset_test[0] == 'svhn':
                        x_test,y_test = svhn_images_test, svhn_y_test
                    elif dataset_test[0] == 'mnistm':
                        x_test,y_test = mnistm_images_test, mnistm_y_test
                    else:
                        x_test,y_test = synth_images_test, synth_y_test 
                    #show_misclassified(x_test=x_test, y_test=y_test, misclassified_idx=misclassified_idx, y_predicted=y_predicted, step=current_step)
                progress+=1

#save results
with open('results/results_dict.json', 'w') as fout:
    json.dump(results, fout)
plot_results(results)