import torch.nn as nn
import torchvision.models as models

def get_mobilenet(num_classes=4):
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features[:14].parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
