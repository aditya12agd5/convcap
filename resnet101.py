import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


resnet101 = models.resnet101(pretrained=True)

class Resnet101Feats(nn.Module):
  def __init__(self):
    super(Resnet101Feats, self).__init__()
    self.features = nn.Sequential(
      resnet101.conv1, 
      resnet101.bn1, 
      resnet101.relu, 
      resnet101.maxpool, 
      resnet101.layer1,
      resnet101.layer2,
      resnet101.layer3,
      resnet101.layer4,
    ) 
    self.avgpool = resnet101.avgpool
    self.fc = resnet101.fc 

  def forward(self, x):
    x = self.features(x)
    y = self.avgpool(x)
    y = y.view(y.size(0), -1)
    return x, y
