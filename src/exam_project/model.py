import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class BaseCNN(LightningModule):
    """Our custom CNN to classify facial expressions."""
    def __init__(self, img_size: int, output_dim: int):
        super(BaseCNN, self).__init__()

        self.img_size = img_size
        self.output_dim = output_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.6)   # increased dropout

        # Fully connected layers
        self.fc1 = nn.Linear(3*3*256, 256)   # multiplying 
        self.fc2 = nn.Linear(256, output_dim)

        # Loss function
        self.loss_fn = nn.NLLLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))              
        x = torch.flatten(x, 1)        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch):
        img, target = batch
        y_pred = self(img)
        return self.loss_fn(y_pred, target)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class BaseANN(LightningModule):
    """Simple MLP for 48x48 grayscale images (7 classes)."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = 48 * 48 

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = h

        layers.append(nn.Linear(prev, num_classes)) 
        self.net = nn.Sequential(*layers)

        self.loss_fn = nn.NLLLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 48, 48] -> [B, 2304]
        x = x.reshape(x.size(0), -1)
        logits = self.net(x)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs

    def training_step(self, batch):
        """
        Expects batch to be (images, targets)
        - images: [B, 1, 48, 48]
        - targets: [B] with class indices 0..6
        """
        img, target = batch
        y_pred = self(img)  
        return self.loss_fn(y_pred, target)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == "__main__":
    model = BaseCNN(img_size=48, output_dim=7)
    # Create a random input tensor with shape (batch=1, channels=1, H=img_size, W=img_size)
    x = torch.rand(2, 1, model.img_size, model.img_size)
    output = model(x)
    print(f"Input: {x}")
    print(f"Output shape of model: {output.shape}")
    print(f"Output: {torch.exp(output)}")



    ann = BaseANN(num_classes=7, hidden=(512, 256), dropout=0.3)

    # Create a random input tensor: (batch=2, channels=1, H=48, W=48)
    x = torch.rand(2, 1, 48, 48)
    output = ann(x)

    print(f"Output shape of model: {output.shape}")  # [2, 7]
    print(f"Output (probabilities): {torch.exp(output)}")

    # Example "batch" for training_step
    y = torch.randint(0, 7, (2,), dtype=torch.long)
    loss = ann.training_step((x, y))
    print(f"Loss: {loss.item():.4f}")
