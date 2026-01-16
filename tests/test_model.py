import pytest
import torch
from exam_project.model import BaseCNN, BaseANN, ViTClassifier


def set_global_seed(seed: int = 42) -> None:
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


MODELS = [
    ("BaseCNN", BaseCNN(img_size=48, output_dim=7, lr=1e-3)),
    ("BaseANN", BaseANN(output_dim=7, hidden=(512, 256), dropout=0.3, lr=1e-3)),
    ("ViTClassifier", ViTClassifier(output_dim=7, lr=1e-4)),
]


@pytest.mark.parametrize(
    "model_name,model",
    MODELS,
    ids=["BaseCNN", "BaseANN", "ViTClassifier"]
)
class TestModels:
    """Combined test suite for all models."""

    def test_initialization(self, model_name, model):
        """Test model initialization."""
        try:
            assert isinstance(model, (BaseCNN, BaseANN, ViTClassifier))
            assert hasattr(model, 'forward')
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Initialization failed: {e}")

    def test_forward_pass(self, model_name, model):
        """Test forward pass with batch of images."""
        x = torch.randn(2, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            output = model(x)

        try:
            assert output.shape == (2, 7), f"Shape {output.shape} != (2, 7)"
            assert output.dtype == torch.float32
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Forward pass failed: {e}")

    def test_training_step(self, model_name, model):
        """Test training step returns valid loss."""
        x = torch.randn(2, 1, 48, 48)
        y = torch.randint(0, 7, (2,))
        loss = model.training_step((x, y))

        try:
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert loss.item() > 0
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Training step failed: {e}")

    def test_optimizer_config(self, model_name, model):
        """Test optimizer configuration."""
        optimizer = model.configure_optimizers()
        try:
            assert optimizer is not None
            assert hasattr(optimizer, 'step')
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Optimizer config failed: {e}")

    def test_reproducibility(self, model_name, model):
        """Test reproducibility with seeding and dropout."""
        x = torch.randn(2, 1, 48, 48)

        # Create models with same seed
        set_global_seed(42)
        if model_name == "BaseCNN":
            m1 = BaseCNN(img_size=48, output_dim=7, lr=1e-3)
        elif model_name == "BaseANN":
            m1 = BaseANN(output_dim=7, hidden=(512, 256), dropout=0.3, lr=1e-3)
        else:
            m1 = ViTClassifier(output_dim=7, lr=1e-4)

        set_global_seed(42)
        if model_name == "BaseCNN":
            m2 = BaseCNN(img_size=48, output_dim=7, lr=1e-3)
        elif model_name == "BaseANN":
            m2 = BaseANN(output_dim=7, hidden=(512, 256), dropout=0.3, lr=1e-3)
        else:
            m2 = ViTClassifier(output_dim=7, lr=1e-4)

        try:
            m1.eval()
            m2.eval()
            with torch.no_grad():
                out1 = m1(x)
                out2 = m2(x)
            assert torch.allclose(out1, out2, atol=1e-5), (
                "Seeding failed: outputs differ"
            )

            if model_name in ["BaseCNN", "BaseANN"]:
                m1.train()
                assert not torch.allclose(m1(x), m1(x)), (
                    "Dropout inactive in training"
                )

            m1.eval()
            with torch.no_grad():
                assert torch.allclose(m1(x), m1(x), atol=1e-7)
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Reproducibility failed: {e}")

    def test_parameters(self, model_name, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        trainable = [p for p in params if p.requires_grad]
        try:
            assert len(params) > 0, "No parameters"
            assert len(trainable) > 0, "No trainable parameters"
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Parameters failed: {e}")


class TestModelComparison:
    """Cross-model consistency tests."""

    def test_output_shape_consistency(self):
        """Test all models produce same output shape."""
        models = [
            ("BaseCNN", BaseCNN(img_size=48, output_dim=7)),
            ("BaseANN", BaseANN(output_dim=7)),
            ("ViTClassifier", ViTClassifier(output_dim=7)),
        ]
        x = torch.randn(2, 1, 48, 48)

        for name, model in models:
            try:
                model.eval()
                with torch.no_grad():
                    output = model(x)
                assert output.shape == (2, 7), f"[{name}] Wrong shape"
            except AssertionError as e:
                pytest.fail(str(e))

    def test_device_compatibility(self):
        """Test model device movement."""
        model = BaseCNN(img_size=48, output_dim=7)
        x = torch.randn(2, 1, 48, 48)

        try:
            model = model.cpu()
            x = x.cpu()
            output = model(x)
            assert output.device.type == 'cpu', "Device mismatch"
        except Exception as e:
            pytest.fail(f"[BaseCNN] Device compatibility failed: {e}")

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = BaseCNN(img_size=48, output_dim=7)
        x = torch.randn(2, 1, 48, 48, requires_grad=True)
        y = torch.randint(0, 7, (2,))

        try:
            output = model(x)
            loss = model.loss_fn(output, y)
            loss.backward()

            assert x.grad is not None, "[BaseCNN] No input gradients"
            assert any(
                p.grad is not None for p in model.parameters() if p.requires_grad
            ), "[BaseCNN] No parameter gradients"
        except Exception as e:
            pytest.fail(f"[BaseCNN] Gradient flow failed: {e}")

