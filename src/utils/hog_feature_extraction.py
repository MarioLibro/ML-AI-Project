import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
import random

#VERBOSE -> show plots
VERBOSE = False

def extract_features_hog(samples, dataset_name):
    """
    Images feature extractor via HOG
    """

    features = []
    hog_images = []

    for img in samples:
        img_resize = resize(img, (128, 64))
        feature, hog_image = hog(img_resize, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=2)
        features.append(feature)
        hog_images.append(hog_image)

    sample_idx = random.sample(range(0, len(hog_images)), 20)
    imgs = [samples[i] for i in sample_idx]
    hog_imgs = [hog_images[i] for i in sample_idx]
    save_hog_samples(imgs, hog_imgs, dataset_name)
    return np.array(features)

def save_hog_samples(images, hog_images, dataset_name):
    """
    Save/Show 20 random hog samples 
    """

    _, axs = plt.subplots(2, 10, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs[:10]):
        img = resize(img, (128, 64))
        ax.axis('off')
        ax.imshow(img, cmap=plt.cm.gray)
    for img, ax in zip(hog_images, axs[10:]):
        # Rescale histogram for better display 
        img = exposure.rescale_intensity(img, in_range=(0, 10)) 
        ax.axis('off')
        ax.imshow(img, cmap=plt.cm.gray)
    plt.subplots_adjust(bottom=0.4, top=0.7, hspace=0)
    plt.savefig('results/dataset_samples/hog_'+dataset_name+'.png')
    if VERBOSE:
        plt.show()
    plt.close()