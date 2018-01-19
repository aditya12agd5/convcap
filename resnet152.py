import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


resnet152 = models.resnet152(pretrained=True)

class Resnet152Feats(nn.Module):
  def __init__(self):
    super(Resnet152Feats, self).__init__()
    self.features = nn.Sequential(
      resnet152.conv1, 
      resnet152.bn1, 
      resnet152.relu, 
      resnet152.maxpool, 
      resnet152.layer1,
      resnet152.layer2,
      resnet152.layer3,
      resnet152.layer4,
    ) 
    self.avgpool = resnet152.avgpool
    self.fc = resnet152.fc 

  def forward(self, x):
    x = self.features(x)
    y = self.avgpool(x)
    y = y.view(y.size(0), -1)
    return x, y
