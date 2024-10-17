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

train_dir = 'school_bus_dataset/train'
valid_dir = 'school_bus_dataset/val'
test_dir = 'school_bus_dataset/test'


#Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validate_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validate_dataset = datasets.ImageFolder(valid_dir, transform=validate_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validateloader = torch.utils.data.DataLoader(validate_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# define a separate dataloader for prediction using the test dataset
predictloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

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

#Train the model
model.train()

epochs = 5
trn_batch = 0
trn_loss = 0

device = torch.device("cpu")

model.to(device)

# Loss function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
        
for epoch in range(epochs):
    
    print(f"Epoch {epoch+1}/{epochs}:")
    print("Training Process Started..")
    
    trn_loss = 0  
    trn_batch = 0
    
    for inputs, labels in trainloader:
        
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        trn_batch += 1
                
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        
        print(f"Training for Batch {trn_batch}/{len(trainloader)} Completed.")
            
    print(f"Training Process for Epoch {epoch+1}/{epochs} Completed.")
        
    
    model.eval()

    print("Validation Process In Progress..")

    v_batch = 0
    v_loss = 0
    v_accuracy = 0

    with torch.no_grad():

        for inputs, labels in validateloader:

            v_batch += 1

            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)              
            v_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            v_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Validation for Batch {v_batch}/{len(validateloader)} Completed.")

    print(f"Validation Process Completed.")            
    
    print(f"Training loss: {trn_loss/len(trainloader):.3f}.. "
          f"Validation loss: {v_loss/len(validateloader):.3f}.. "
          f"Validation accuracy: {v_accuracy*100/len(validateloader):.3f}%")
    
    model.train()


# testing the model

model.eval()

t_batch = 0
t_loss = 0
t_accuracy = 0

device = torch.device('cpu')

with torch.no_grad():
    
    print("Testing Process Started..")
    
    for inputs, labels in testloader:
        
        t_batch += 1
        
        print(f"Testing for Batch {t_batch}/{len(testloader)} in progerss..")
        
        model.to(device)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)       
        t_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        
        equals = top_class == labels.view(*top_class.shape)
        t_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Testing for Batch {t_batch}/{len(testloader)} completed.")

    print("Testing Process Completed.")

print(f"Testing loss: {t_loss/len(testloader):.3f}.. "
      f"Testing accuracy: {t_accuracy*100/len(testloader):.3f}%")


# Save the checkpoint 
def save_checkpoint(model, filepath, model_name, epochs, learning_rate):

    checkpoint2_vgg16 = {'input_size': (3, 224, 224),
                  'output_size': 2,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_name': model_name,
                  'epochs': epochs,
                  'learning_rate': learning_rate}

    torch.save(checkpoint2_vgg16, filepath)
    
save_checkpoint(model, 'checkpoint2_vgg16', model_name, epochs, learning_rate) 


