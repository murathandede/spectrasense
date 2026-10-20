import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(cin, cout, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class Conv1DAMCNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=11, dropout=0.3):
        super().__init__()
        if in_channels not in (2, 4):
            raise ValueError("Expected 2 or 4 input channels")
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 24, 7, padding=3, bias=False), nn.BatchNorm1d(24), nn.ReLU(),
            ResidualBlock(24, 24), ResidualBlock(24, 48, 2), ResidualBlock(48, 64, 2),
            nn.AdaptiveAvgPool1d(4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
