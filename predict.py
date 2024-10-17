# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 


# Imports here

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

from collections import OrderedDict

import numpy as np

from torch.autograd import Variable

from PIL import Image


predict_dir = 'school_bus_dataset/test'


#Define the transform for the testing set
predict_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

# Load the dataset with ImageFolder
predict_dataset = datasets.ImageFolder(predict_dir, transform=predict_transforms)

# define dataloader for prediction using the test dataset
predictloader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=True)


# use the pretrained model (vgg16) for the classification
model = models.vgg16(weights=True)
print(model)


model_name = 'vgg16'


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# modify the fc layers of the pretrained model to suit our project
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier


# Loss function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
learning_rate = 0.003
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


# specify the model epochs

epochs = 5

#  load checkpoint and rebuild the model
def load_checkpoint(filepath):
    model = create_model()
    checkpoint2_vgg16 = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint2_vgg16['state_dict'])
    optimizer.load_state_dict(checkpoint2_vgg16['optimizer_state_dict'])
    model.eval()
    return model

def create_model():
    model = models.vgg16(pretrained=True)
    model.classifier = classifier
    return model

model = load_checkpoint('checkpoint2_vgg16')



#  function to process the image before the prediction

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
        
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        
    image = transform(image)
                      
    return image
    

#  function for displaying the image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



#  function to predict the image classification

def predict(image, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = image.view(1, 3, 224, 224)
    

    # Calculate the class probabilities (softmax) for image
    with torch.no_grad():
        output = model.forward(image)

    ps = torch.exp(output)

    top_ps, top_class = ps.topk(topk, dim=1)
                
    return top_ps, top_class


# Function to display the image along with the predicted classes

def view_classify(image, top_ps, value, label_name, version="Flowers"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    
    image = image.to('cpu')
    image = image.squeeze(0)
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    top_ps = top_ps.data.numpy().squeeze()
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,4))
    ax1.imshow(image)
    ax1.set_title(label_name)
    ax1.axis('off')
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(1.1))

    if version == "buses":
        bars = ax2.barh(value, top_ps, color='blue') 
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1.2)
        ax2.set_ylim(-1, 2)


        # Add probability values on top of the bars
        for bar, prob in zip(bars, top_ps):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob*100:.2f}%', 
                     va='center', ha='left')  # Adjust position as needed
    
    plt.show()
    

# Code that randomly provides an image from the test data set to the model for prediction

model.eval()

device = torch.device('cpu')

with torch.no_grad():
        
    data_iter = iter(predictloader)

    image, label = next(data_iter)

    model.to(device)

    image, label = image.to(device), label.to(device)
    
    topk = 2
    
    top_ps, top_class = predict(image, model, topk)

    class_name_dict = {'0': 'Not School Bus', '1': 'School Bus'}

    value = []

    for i in range(top_class.shape[1]):
        value.append(class_name_dict.get('{}'.format(top_class[0, i])))
        
    view_classify(image, top_ps, value, label_name=' ', version='buses')

            

