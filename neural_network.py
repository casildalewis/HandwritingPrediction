import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    arguments: 
        An optional boolean argument (default value is True for training dataset)

    returns:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False, transform=custom_transform)
  
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 50)
    
    return loader
  
def build_model():
    """
    arguments: 
        None

    returns:
        An untrained neural network model
    """

    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    )
    
    return model


def train_model(model, train_loader, criterion, T):
    """
    arguments: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    returns:
        None
    """

    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
    for i in range(T):

        correct = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in train_loader:

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # to print stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        datasetSize = len(train_loader)
        print("Train Epoch: {:d} Accuracy: {:d}/{:d}({:.2f}%) Loss: {:.3f}".format(i, correct, total, 100*(correct/total), running_loss/datasetSize))
            

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    arguments: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    returns:
        None
    """
    model.eval()
  
    correct = 0
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            # to print stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    if show_loss:
        print("Average Loss: {:.4f}".format(running_loss/total))
    print("Accuracy: {:.2f}%".format(100*(correct/total)))


def predict_label(model, test_images, index):
    """
    arguments: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    returns:
        None
    """
    logits = model(test_images)
    prob = F.softmax(logits, dim=1)

    probAtIndex, indices  = torch.sort(prob[index], descending=True)
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    for i in range(3):
        print("{}: {:.2f}%" .format(class_names[indices[i]], 100*probAtIndex[i]))


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()

    train_loader = get_data_loader()
    test_loader = get_data_loader(False)

    # print(type(train_loader))
    # print(train_loader.dataset)
  
    model = build_model()
    # print(model)

    train_model(model, train_loader, criterion, T = 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)

    test_images = []
    for (images, labels) in test_loader:
        test_images.append(images)
    test_images = torch.cat(test_images, dim=0)

    predict_label(model, test_images, 1)