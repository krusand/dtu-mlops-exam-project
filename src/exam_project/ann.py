import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNClassifier(nn.Module):
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
    model = ANNClassifier(num_classes=7, hidden=(512, 256), dropout=0.3)

    # Create a random input tensor: (batch=2, channels=1, H=48, W=48)
    x = torch.rand(2, 1, 48, 48)
    output = model(x)

    print(f"Output shape of model: {output.shape}")  # [2, 7]
    print(f"Output (probabilities): {torch.exp(output)}")

    # Example "batch" for training_step
    y = torch.randint(0, 7, (2,), dtype=torch.long)
    loss = model.training_step((x, y))
    print(f"Loss: {loss.item():.4f}")
