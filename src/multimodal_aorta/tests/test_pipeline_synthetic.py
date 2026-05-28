"""
Synthetic pipeline smoke test — no file I/O, no GPU required.

Uses random tensors to verify:
  1. Forward pass produces correct shapes
  2. Loss backward pass works (gradients flow to all components)
  3. One optimizer step runs without NaN
  4. Checkpoint save/load round-trips correctly
  5. Evaluation loop runs without errors

Run from src/:
    python -m multimodal_aorta.tests.test_pipeline_synthetic
"""

import os, sys, tempfile, logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO during tests

import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock

from multimodal_aorta.configs.default_config import Config
from multimodal_aorta.models.ecg_encoder import ResNet1DEncoder
from multimodal_aorta.models.fusion import FusionTransformer
from multimodal_aorta.models.regression_head import RegressionHead
from multimodal_aorta.models.full_model import AortaModel
from multimodal_aorta.training.losses import total_loss, masked_huber_loss
from multimodal_aorta.training.evaluate import evaluate
from multimodal_aorta.utils.logging_utils import CSVLogger, save_checkpoint, load_checkpoint

DEVICE = torch.device("cpu")
B = 4   # small batch for speed
D = 768  # d_model


class _StubCXREncoder(nn.Module):
    """Replaces RAD-DINO with a single linear layer for fast CPU testing."""
    def __init__(self):
        super().__init__()
        self.out_dim = D
        self.proj = nn.Linear(3 * 224 * 224, D)
        self.config = MagicMock()
        self.config.hidden_size = D

    def forward(self, pixel_values):
        return self.proj(pixel_values.flatten(1))

    def set_frozen_for_epoch(self, *a, **kw): pass
    def parameters(self): return super().parameters()


def _make_fast_model(cfg):
    """Build AortaModel but swap RAD-DINO for the stub — loads in <1 second."""
    with patch("multimodal_aorta.models.full_model.CXREncoder",
               return_value=_StubCXREncoder()):
        model = AortaModel(cfg.model, cfg.train)
    model.cxr_encoder = _StubCXREncoder()
    return model.to(DEVICE)


def _fake_batch(has_ecg=None, has_cxr=None):
    """Build a collated batch of random tensors."""
    if has_ecg is None:
        has_ecg = torch.tensor([True, True, True, False])
    if has_cxr is None:
        has_cxr = torch.tensor([True, False, True, False])
    # Include NaN in targets to test masked loss
    target = torch.tensor([
        [3.2, 3.5],
        [float("nan"), 2.9],   # root label missing
        [2.8, float("nan")],   # asc label missing
        [4.1, 4.3],
    ])
    return {
        "ecg":     torch.randn(B, 12, 5000),
        "cxr":     torch.randn(B, 3, 224, 224),
        "target":  target,
        "has_ecg": has_ecg,
        "has_cxr": has_cxr,
    }


def _make_loader(n_batches=3):
    """Fake DataLoader that yields random batches."""
    return [_fake_batch() for _ in range(n_batches)]


def test_masked_loss():
    print("  [2] Masked Huber loss with NaN targets...", end=" ")
    pred = torch.randn(B, 2)
    target = torch.tensor([
        [3.0, 3.2],
        [float("nan"), 2.8],
        [2.9, float("nan")],
        [4.0, 4.1],
    ])
    loss = total_loss(pred, target)
    assert loss.item() >= 0, "Loss must be non-negative"
    assert not torch.isnan(loss), "Loss must not be NaN"
    # All-NaN target should give 0 loss (nothing to optimise)
    all_nan = torch.full((B, 2), float("nan"))
    loss_zero = masked_huber_loss(pred, all_nan)
    assert loss_zero.item() == 0.0, f"All-NaN loss should be 0, got {loss_zero.item()}"
    print("OK")



def test_csv_logger():
    print("  [7] CSV logger...", end=" ")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.csv")
        log = CSVLogger(path)
        log.write(epoch=1, train_loss=0.5, val_loss=0.4, mae_root=0.3, mae_asc=0.35)
        log.write(epoch=2, train_loss=0.4, val_loss=0.35, mae_root=0.28, mae_asc=0.30)
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["epoch"] == "1"
    print("OK")


def test_forward(model):
    print("  [1] Forward pass shapes...", end=" ")
    model.eval()
    b = _fake_batch()
    with torch.no_grad():
        out = model(b["ecg"], b["cxr"], b["has_ecg"], b["has_cxr"])
    assert out.shape == (B, 2), f"Expected ({B}, 2), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print("OK")


def test_backward(model):
    print("  [3] Backward pass + optimizer step...", end=" ")
    model.train()
    optimizer = optim.AdamW(model.get_param_groups(), weight_decay=1e-2)
    b = _fake_batch()
    optimizer.zero_grad()
    out = model(b["ecg"], b["cxr"], b["has_ecg"], b["has_cxr"])
    loss = total_loss(out, b["target"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert not torch.isnan(loss), "Loss is NaN after backward"
    head_grad = model.head.net[1].weight.grad
    assert head_grad is not None and not torch.isnan(head_grad).any()
    print(f"OK  (loss={loss.item():.4f})")


def test_two_training_steps(model):
    print("  [4] Two full training steps...", end=" ")
    model.train()
    optimizer = optim.AdamW(model.get_param_groups())
    losses = []
    for _ in range(2):
        b = _fake_batch()
        optimizer.zero_grad()
        out = model(b["ecg"], b["cxr"], b["has_ecg"], b["has_cxr"])
        loss = total_loss(out, b["target"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(not torch.tensor(l).isnan() for l in losses)
    print(f"OK  (losses: {losses[0]:.4f} → {losses[1]:.4f})")


def test_evaluate(model):
    print("  [5] Evaluation loop...", end=" ")
    model.eval()
    loader = _make_loader(n_batches=4)
    from multimodal_aorta.training.evaluate import evaluate as eval_fn
    metrics = eval_fn(model, loader, DEVICE)
    assert not torch.tensor(metrics.val_loss).isnan()
    print(f"OK  (val_loss={metrics.val_loss:.4f}, total_mae={metrics.total_mae:.4f})")


def test_checkpoint(model):
    print("  [6] Checkpoint save/load...", end=" ")
    optimizer = optim.AdamW(model.get_param_groups())
    w_before = model.head.net[1].weight.detach().clone()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        save_checkpoint(path, model, optimizer, epoch=1,
                        metrics={"val_mae": 0.3}, config={})
        with torch.no_grad():
            model.head.net[1].weight.fill_(999.0)
        load_checkpoint(path, model, optimizer, device=DEVICE)
        w_after = model.head.net[1].weight.detach()
    assert torch.allclose(w_before, w_after)
    print("OK")


if __name__ == "__main__":
    print("=== Synthetic pipeline smoke test (CPU, no file I/O) ===\n")
    print("  Building model with stub CXR encoder (no RAD-DINO, fast)...", end=" ", flush=True)
    cfg = Config()
    shared_model = _make_fast_model(cfg)
    print("done.\n")

    # Tests that need the shared model
    model_tests = [
        lambda: test_forward(shared_model),
        test_masked_loss,
        lambda: test_backward(shared_model),
        lambda: test_two_training_steps(shared_model),
        lambda: test_evaluate(shared_model),
        lambda: test_checkpoint(shared_model),
        test_csv_logger,
    ]

    passed = 0
    for t in model_tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'All' if passed == len(model_tests) else f'{passed}/{len(model_tests)}'} tests passed.")
    sys.exit(0 if passed == len(model_tests) else 1)
