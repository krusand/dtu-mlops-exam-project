import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from exam_project.data import load_data
from exam_project.model import ViTClassifier, BaseCNN


ROOT = Path(__file__).resolve().parents[2]    # go two levels up to project root
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
def load_model(model_file_name: str = "checkpoint.pth", device: str = DEVICE) -> None:
    """
    Loads a trained image classification model.

    Params: 
    - model_file_name:      Name of file containing trained model object.
    - device:               Device on which to store the test data.

    Returns:
    - model:                Retrieved model object.
    """
    #model = BaseANN()
    model = ViTClassifier()
    model_path = str(ROOT / "models" / f"{model_file_name}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model

class DistillationModule(LightningModule):
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, lr: float = 1e-4, alpha: float = 0.5, temperature: float = 2.0):
        """
        Knowledge distillation module.

        Params:
        - teacher_model: pretrained ViT teacher
        - student_model: smaller ViT student
        - lr: learning rate
        - alpha: weight between distillation loss and cross-entropy
        - temperature: softening parameter
        """
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()  # freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student_model
        self.lr = lr
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Distillation loss (KL divergence)
        T = self.temperature
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        # Standard supervised loss
        ce_loss = F.cross_entropy(student_logits, y)

        # Combine losses
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        return optimizer


# Load the teacher model (pretrained, large)
teacher_model = load_model("models_vit_vit_production_model.ckpt", device=DEVICE)

# Define a smaller student model
student_model = BaseCNN()

# Wrap in distillation module
distill_module = DistillationModule(
    teacher_model=teacher_model,
    student_model=student_model,
    lr=1e-4,
    alpha=0.7,         # weight for distillation loss
    temperature=3.0
)

# Create dataloaders
_, val, train_dataset = load_data(processed_dir="data/processed/")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=0)

# Save the best student model based on validation loss
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",          # track validation loss
    dirpath="models/",      # folder to save
    filename="student-model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,                # keep only the best model
    mode="min"                   # minimize val_loss
)

early_stop_callback = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        mode='min',                 # we want to minimize validation loss
        min_delta=0.00,             # minimum change to qualify as improvement
        patience=1,                 # stop if no improvement after N epochs
    )

# Trainer
trainer = Trainer(
    max_epochs=10,
    accelerator=DEVICE,
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback]
)

trainer.fit(distill_module, train_loader, val_loader)

# Best model path
best_model_path = checkpoint_callback.best_model_path
print("Best model saved at:", best_model_path)

'''
student_model = BaseCNN()
student_model.load_state_dict(torch.load(best_model_path)['state_dict'])
student_model.to(DEVICE).eval()
'''
