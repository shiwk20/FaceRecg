from torch import nn
import torch

class FaceRecg(nn.Module):
    def __init__(self):
        super(FaceRecg, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)