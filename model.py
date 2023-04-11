import torch
from torchvision.models import resnet50
import torch.nn as nn

num_classes = 5

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # ResNet layers
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True)
        
        # FC layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # CNN
        with torch.no_grad():
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x = self.resnet.avgpool(x)
            x = torch.flatten(x, 1)
        
        # LSTM
        x = x.unsqueeze(0)
        x, _ = self.lstm1(x)
        x = x.squeeze(0)
        
        # FC
        x = self.fc(x)
        
        return x
