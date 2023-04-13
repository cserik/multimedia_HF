import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, hidden_size=256, num_classes=5):
        super(LSTMNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.lstm = nn.LSTM(input_size=512*6*6, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv(x)
        
        # Flatten output and add sequence dimension
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0)

        print(x.shape)

        # Apply LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        # Apply fully connected layer
        x = self.fc(x)
        
        return x