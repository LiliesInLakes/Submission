import torch
import torch.nn as nn

class ConvNeuralNet(nn.Module):
    def __init__(self): # Fixed double underscores
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 0, 1, 2
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 3, 4, 5
            nn.MaxPool2d(2), # 6
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 7, 8, 9
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 10, 11, 12
            nn.MaxPool2d(2), # 13
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 14, 15, 16
            nn.MaxPool2d(2), # 17
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

