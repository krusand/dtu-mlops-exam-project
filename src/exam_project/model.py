import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """Our custom CNN to classify facial expressions."""
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.6)   # increased dropout
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        # Fully connected layers
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)                # GAP reduces to (B, 256, 1, 1)
        x = torch.flatten(x, 1)        # flatten to (B, 256)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CustomCNN()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
