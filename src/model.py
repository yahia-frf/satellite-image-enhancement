# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=10):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
