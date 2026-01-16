import hydra
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score
class BaseCNN(LightningModule):
    """Our custom CNN to classify facial expressions."""
    def __init__(self, 
                 img_size: int, 
                 output_dim: int, 
                 lr: float,
                 dropout: float = 0.2,
                 ) -> None:
        super(BaseCNN, self).__init__()

        self.img_size = img_size
        self.output_dim = output_dim
        self.lr = lr
        self.dropout = dropout

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
        self.dropout = nn.Dropout(self.dropout)   # increased dropout

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
        """
        Expects batch to be (images, targets)
        - images: [B, 1, 48, 48]
        - targets: [B] with class indices 0..6
        """
        img, target = batch
        y_pred = self(img) 
        loss = self.loss_fn(y_pred, target)
        y_pred_class = torch.argmax(y_pred, dim=1)

        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy_score(y_true=target, y_pred=y_pred_class))
        return loss
    
    def validation_step(self, batch):
        """
        Expects batch to be (images, targets)
        - images: [B, 1, 48, 48]
        - targets: [B] with class indices 0..6
        """
        img, target = batch
        y_pred = self(img)  
        loss = self.loss_fn(y_pred, target)
        y_pred_class = torch.argmax(y_pred, dim=1)

        self.log("validation_loss", loss)
        self.log("validation_accuracy", accuracy_score(y_true=target, y_pred=y_pred_class))
        return loss

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class BaseANN(LightningModule):
    """Simple MLP for 48x48 grayscale images (7 classes)."""

    def __init__(
        self,
        img_size: int,
        output_dim: int,
        hidden: tuple[int, ...],
        dropout: float,
        lr: float,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.output_dim = output_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr

        layers: list[nn.Module] = []
        prev = img_size * img_size

        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = h

        layers.append(nn.Linear(prev, output_dim)) 
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
        loss = self.loss_fn(y_pred, target)
        y_pred_class = torch.argmax(y_pred, dim=1)

        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy_score(y_true=target, y_pred=y_pred_class))
        return loss
    
    def validation_step(self, batch):
        """
        Expects batch to be (images, targets)
        - images: [B, 1, 48, 48]
        - targets: [B] with class indices 0..6
        """
        img, target = batch
        y_pred = self(img)  
        loss = self.loss_fn(y_pred, target)
        y_pred_class = torch.argmax(y_pred, dim=1)

        self.log("validation_loss", loss)
        self.log("validation_accuracy", accuracy_score(y_true=target, y_pred=y_pred_class))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ViTClassifier(LightningModule):
    """Vision Transformer (ViT) for image classification using Hugging Face."""

    def __init__(
        self,
        model_name: str,
        lr: float,
        freeze_backbone: bool,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self. output_dim = output_dim

        # Load pretrained ViT model with custom number of classes
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=output_dim,
            ignore_mismatched_sizes=True,
        )

        # Load image processor for preprocessing
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Optionally freeze the backbone (only train classifier head)
        if freeze_backbone:
            for param in self.vit.vit.parameters():
                param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for ViT.
        Converts grayscale [B, 1, H, W] to RGB [B, 3, 224, 224].
        ViT model expects RGB images of size 224x224.
        """
        # Convert grayscale to RGB by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to ViT expected size (224x224)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize using ViT's expected normalization
        mean = torch.tensor(self.processor.image_mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        outputs = self.vit(pixel_values=x)
        return outputs.logits

    def training_step(self, batch):
        img, target = batch
        logits = self(img)
        loss = self.loss_fn(logits, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    # Create a random input tensor with shape (batch=1, channels=1, H=img_size, W=img_size)
    x = torch.rand(2, 1, cfg.models.img_size, cfg.models.img_size)

    # Instantiate model
    model = instantiate(cfg.models)

    # Sample output for CNN
    output = model(x)
    print(f"Input: {x}")
    print(f"Output shape of model: {output.shape}")
    print(f"Output: {torch.exp(output)}")


if __name__ == "__main__":
    main()