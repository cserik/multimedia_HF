import torch
from torchvision.models import resnet50
from torchvision import models
import torch.nn as nn

num_classes = 6

#kipróbálni, hogy csak onnan szedek ki 30 kepet (grayscale + facecrop) ahol a legnagyobb valtozas tortent a videoban, majd az egymas utani harmat
#osszerakom egy 3 channeles keppe, hogy jo resnet input legyen és akkor 10-esevel fogom beadni a kepet a neuralis halonak vagy ebbol az informaciot 3x256x256-ra tomoriteni

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
    

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super(ResNetLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        # Freeze model parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        fc_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes), 
            nn.LogSoftmax(dim=1) 
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # ResNet forward pass
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        
        # LSTM forward pass
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        x, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        
        # Output layer forward pass
        x = self.fc(x[:, -1, :])
        
        return x
    

class ResNetLSTM2(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super(ResNetLSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove the last FC layer of the ResNet-50 model
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # LSTM layers
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # ResNet forward pass
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        
        # LSTM forward pass
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        x, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        
        # Output layer forward pass
        x = self.fc(x[:, -1, :])
        
        return x
    
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x

