# define the network modules

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # load the VGG16 models with pretrained weights
        vgg16 = models.vgg16(pretrained=True)

        # freez the network weights so they won't get updated during training
        for parameter in vgg16.parameters():
            parameter.requires_grad_(False)

        # get all the network modules except the last part which is the fully
        # connected layers
        modules = list(vgg16.children())[:-1]

        # combine all the modules in one sequential
        self.vgg16 = nn.Sequential(*modules)

        # create two fully connected layers with drop layer for regularization
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.fc_drop = nn.Dropout(0.5)


    def forward(self, x):
        # pass the input through the prtrained net
        x = self.vgg16(x)
        # flatten x to feed it to the fully connected layers
        x = x.view(x.size(0), -1)

        # pass x through the fully connected layers with relu activation function
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)

        return x