import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torchvision.models as models
import torch

VERBOSE = False

#initializing compute device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.backends.mps.is_available()):
    device = 'mps'

# Images feature extractor
class CNN_Feature_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # Remove last two layer (avgpool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        
    def forward(self, images):
        img_embeddings = self.resnet(images)   
        conv_features = torch.flatten(img_embeddings, 1)
        
        return conv_features

def extract_feature(dataloader):
    # Initialize objects needed for training
    model = CNN_Feature_Extractor().to(device)
    i = 0
    labels = []
    for batch_images, batch_labels in dataloader:
        model.eval()
        with torch.no_grad():
            # Transform the images tensor into a float
            batch_images, batch_labels = batch_images.float().to(device), batch_labels.to(device)

            # Forward pass to get output
            outputs = model(batch_images)

            labels.extend(batch_labels.detach().cpu().numpy())

            if(i==0):
                feature = outputs.data.detach().cpu().numpy()
                images = batch_images.data.detach().cpu().numpy()
                if(VERBOSE): 
                    print('CNN_FEATURE_EXTRACTION')
                    print(f'images_shape: {batch_images.shape}, labels_shape: {batch_labels.shape}\nfeature_shape: {feature.shape}')
            else:
                feature = np.concatenate((feature, outputs.data.detach().cpu().numpy()))
                images = np.concatenate((images, batch_images.data.detach().cpu().numpy()))
                if(VERBOSE): print('feature_shape: ',feature.shape)
            i+=1

    #images is used only to show the missclasifcation samples
    return images, feature, labels