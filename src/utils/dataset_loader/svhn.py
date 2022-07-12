import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

#VERBOSE -> show plots, and print info
VERBOSE = False
RANDOM_STATE = 42

def preprocessing(samples, labels, n_samples_per_class):
    """ 
    Balance classes, transform label '10 to '0', permute shape
    """

    for i in range(1, 11):
        idx = np.where(labels == [i])        
        if (len(idx[0]) - n_samples_per_class) > 0:
            idxs_final = np.random.choice(idx[0], (len(idx[0]) - n_samples_per_class), replace=False)
            samples = np.delete(samples, idxs_final, 3)
            labels = np.delete(labels, idxs_final, 0)
        else:
            raise Exception('svhn.py_ABORT -> not enoguh samples per class')

    #labels '10' to '0'
    labels[np.where(labels == 10)] = 0

    #32,32,3,N -> N,32,32,3
    samples = np.transpose(samples, (3, 0, 1, 2))

    """
    #counter samples per class
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    """
    
    return samples, labels

def save_random_samples(samples, labels):
    """
    Show/Save 20 random samples (2 for each class)
    """

    l=[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
    
    figure = plt.figure(figsize=(8,8))
    cols, rows = 10, 2
    for i in range(0, cols * rows):
        tmp = np.where(labels == l[i])
        samples_idx = np.random.choice(tmp[0])
        img, label = samples[samples_idx], labels[samples_idx]
        figure.add_subplot(rows, cols, i+1)
        plt.title(label.item())
        plt.axis('off')
        plt.imshow(img)
        plt.subplots_adjust(bottom=0.4, top=0.7, hspace=0)
    plt.savefig('results/dataset_samples/svhn.png')
    if VERBOSE:
        plt.show()
    plt.close()

class SVHN(Dataset):
    def __init__(self, split: str = 'test', test_size = 0.20, n_samples_per_class = 300):
        
        # --- Load SVHN dataset ---
        svhn = loadmat('data/original_dataset_files/svhn.mat')
        x = np.uint8(svhn['X'])
        y = np.uint8(svhn['y'])
        # -----------------
        x, y = preprocessing(x, y, n_samples_per_class)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle = True, random_state=RANDOM_STATE)

        if split == 'test':
            self.samples = x_test
            self.labels = y_test
            save_random_samples(self.samples, self.labels)
            if VERBOSE:
                print('SVHN.PY - Test Dataset Dimensions:')
                print('samples.shape:', self.samples.shape, 'labels.shape:', len(self.labels))
        else:
            self.samples = x_train
            self.labels = y_train
            save_random_samples(self.samples, self.labels)
            if VERBOSE:
                print('SVHN.PY - Train Dataset Dimensions:')
                print('samples.shape:', self.samples.shape, 'labels.shape:', len(self.labels))

        # --- Resize, toTensor and Data normalization ((mean and std for all three channels) -> min converted to -1 and max converted to 1 ---
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # -----------------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.samples[idx], 'RGB')
        img = self.transform(img)
        label = self.labels[idx,0]
        return img, label