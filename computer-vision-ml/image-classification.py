# PyTorch: Convolutional Neural Network Image Classifier
# Author: Louis Heery

### Import required libraries ###
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor




### 1. Dataset Import & Loading ###
print("--- START ---")
print("--- 1. Dataset Import & Loading ---")
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.FashionMNIST(root='.', download=True, train=True,transform=transform)
train_image = np.array(train_set.data)
train_label = np.array(train_set.targets)
class_name = train_set.classes

test_set = torchvision.datasets.FashionMNIST(root='.', download=True, train=False)
test_image = np.array(test_set.data)
test_label = np.array(test_set.targets)




### 2. Convolutional Neural Network (CNN) ###
print("--- 2. Defining Convolutional Neural Network ---")
class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()

        self.input = nn.Conv2d(1, 6, kernel_size=5)
        self.c1 = nn.AvgPool2d(2, stride=2)
        self.s2 = nn.Conv2d(6, 16, kernel_size=5)
        self.c3 = nn.AvgPool2d(2, stride=2)
        self.s4 = nn.Linear(256, 120)
        self.c5 = nn.Linear(120, 84)
        self.f6 = nn.Linear(84, 10)

        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.input(z)
        z = self.tanh(z)
        z = self.c1(z)
        z = self.s2(z)
        z = self.tanh(z)
        z = self.c3(z)
        z = z.view(z.size(0), -1)
        z = self.s4(z)
        z = self.tanh(z)
        z = self.c5(z)
        z = self.tanh(z)
        z = self.f6(z)

        return z




### 3. Instantiate & Use Convolutional Neural Network ###

### 3a. Instantiate Convolutional Neural Network Model ###
print("--- 3a. Instantiating Convolutional Neural Network Model ---")
net = CNN()

# Hyperparameters
learnrate = 0.01
epochs = 100
batchsize = 128

# Loss Function & Optimizer
lossfunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learnrate)


### 3b. Train CNN ###
print("--- 3b. Training Convolutional Neural Network Model ---")
loss_values = []

# Define dataloader to manage the selecting of random batches from training dataset
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

start = time.time()

# Loop through each epoch
for epoch in range(epochs):

    # For each epoch, training in batches of size 'batch_size'
    for i, (input_val, targets) in enumerate(train_loader):

        output = net(input_val)
        loss_value = lossfunction(output, targets)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_values.append(loss_value.item())

    average_loss = sum(loss_values)/len(loss_values)

    print('Epoch: ', epoch, '/', epochs, ' - Loss: ', average_loss)

print('Training Finished. Time Taken: ', time.time() - start, ' seconds.')


### 3c. Test & Evaluate Performance of CNN ###
print("--- 3c. Testing & Evaluting Convolutional Neural Network Model ---")
test_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

# Keep track of number of correctly classified tests
correct_classified = 0
total_classified = 0

start = time.time()

# Loop through all samples in test dataset and record if they are classified correctly
for (input_val, targets) in test_loader:

    output = net(input_val)

    ignore, predicted = torch.max(output.data, 1)

    correct_classified = correct_classified + (predicted == targets).sum().item()
    total_classified = total_classified + targets.size(0)

print('Testing Finished. Time Taken: ', time.time() - start, ' seconds.')

# Print out confusion matrix
# Convert Tensors to Numpy arrays
y_target = targets.detach().numpy()
y_predicted = predicted.detach().numpy()

# Plot Confusion Matrix using Matplotlib
matrix = metrics.confusion_matrix(y_target, y_predicted)
fig, ax = plt.subplots()
ax.matshow(matrix)

# Add Text labels to Confusion matrix
for (i, j), z in np.ndenumerate(matrix):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.ylabel("Image Class")
plt.title("Confusion Matrix of CNN Classifier", pad=20)
fig.set_size_inches(6, 6)
#plt.show()
plt.savefig("cnn_test_confusion_matrix.pdf", bbox_inches='tight',dpi=300)

# Calculate Accuracy
class_accuracy = correct_classified / total_classified
print('Classification Accuracy ', class_accuracy)


#### 3d. Apply CNN to Other Images ###
print("--- 3d. Applying Convolutional Neural Network Model to Other Images ---")
# Define image trainsformation
transformation = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Load images
own_test_data = torchvision.datasets.ImageFolder('images', transform=transformation)

# Print Class Labels
print(own_test_data.classes)

# Keep track of number of correctly classfied images
correct_classified = 0
total_classified = 0

start = time.time()

print('Predicted Class - Actual Class')

# Loop through all test samples in test dataset
for (input_val, targets) in own_testloader:

    output = net(input_val)

    ignore, predicted = torch.max(output.data, 1)

    print(predicted, ' - ',targets)

    correct_classified = correct_classified + (predicted == targets).sum().item()
    total_classified = total_classified + targets.size(0)

print('Testing Finished. Time Taken: ', time.time() - start, ' seconds.')

# Calculate accuracy and print out
class_accuracy = correct_classified / total_classified
print('Classification Accuracy ', correct_classified)

print("--- END ---")
