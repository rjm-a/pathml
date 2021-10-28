import sys
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.optim import lr_scheduler

from cnn_util import TrainInputs, trainModel

def doTraining(data_root, model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    isHuman_model = models.resnet50(pretrained=False)
    # num_features = isHuman_model.fc.in_features
    # isHuman_model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, num_classes))
   
    # isHuman_model.load_state_dict(torch.load('/data/models/isHuman.pt'))

    for param in isHuman_model.parameters():
        param.requires_grad = True

    isHuman_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, num_classes))

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalizer
    ])
    
    test_val_transformer = transforms.Compose([
            transforms.Resize(size=(224,224)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalizer
    ])

    train_data = datasets.ImageFolder(root=data_root + '/train', transform=test_val_transformer)
    val_data = datasets.ImageFolder(root=data_root + '/val', transform=test_val_transformer)
    #test_data = datasets.ImageFolder(root=data_root + '/test', transform=test_val_transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)

    class_names = train_data.classes
    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {
        'train' : len(train_data),
        'val' : len(val_data)
    }

    isHuman_model.cuda()

    # optimizer = optim.SGD(isHuman_model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    optimizer = optim.Adam(base_model.parameters())
    criterion = nn.CrossEntropyLoss()

    inputs = TrainInputs(num_epochs=32,
                         optimizer=optimizer,
                         criterion=criterion,
                         scheduler=None,
                         dataloaders=dataloaders,
                         dataset_sizes=dataset_sizes)

    model = trainModel(inputs, isHuman_model, device)

    torch.save(model.state_dict(), "/data/models/" + model_name + ".pt")

def main():
    label_types = ["coding" , "behavior", "posture", "intensity"]
    classes = {'coding': 2, 'posture': , 'intensity': }
    if sys.argv[1] not in label_types:
        print("Incorrect Label Type")
    else:
        doTraining("/data/" + sys.argv[1], sys.argv[1])

if __name__ == '__main__':
    main()
