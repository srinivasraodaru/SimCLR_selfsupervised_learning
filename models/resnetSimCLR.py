import torch.nn as nn
import torchvision.models as models
from resnet50 import ResNet50

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        
        ###make the model
        self.model = ResNet50(out_dim)
        in_dim = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), self.model.fc)


    def forward(self, x):
        return self.model(x)
