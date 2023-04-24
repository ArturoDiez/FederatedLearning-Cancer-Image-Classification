
import torch
import torch.nn as nn
import torch.nn.functional as F


class BreastCancerNet(nn.Module):
    def __init__(self,):
        super(BreastCancerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=16, kernel_size=5, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels= 32, kernel_size= 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size= 3, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size= 3)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 1)

        ## define the model layer according to the architecture
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
