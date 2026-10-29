"""Correctness and integration tests for Query/Value HRA."""

import torch
from torch.nn import functional as F

from experiments.hra.config import CLASSIFIER_LEARNING_RATE, HRA_LEARNING_RATE, HRA_RANK
from experiments.hra.hra_layer import HRALinear
from experiments.hra.model import ViTHRAClassifier
from experiments.hra.run_hra import build_optimizer, build_scheduler
from vit_core.pretrained import load_pretrained_vit


def test_hra_layer_math() -> None:
    torch.manual_seed(0)
    layer = HRALinear(16, 11, rank=HRA_RANK)
    inputs = torch.randn(3, 5, 16)
    identity = torch.eye(16)
    rotation = layer.orthogonal_matrix()

    torch.testing.assert_close(rotation, identity, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(rotation.T @ rotation, identity, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(layer(inputs), layer.base(inputs), atol=5e-6, rtol=5e-6)
    dense_output = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    torch.testing.assert_close(layer(inputs), dense_output, atol=5e-6, rtol=5e-6)

    layer(inputs).sum().backward()
    assert layer.householder_vectors.grad is not None
    assert torch.isfinite(layer.householder_vectors.grad).all()


def test_hra_model_and_optimizer() -> None:
    num_classes = 10
    model = ViTHRAClassifier(num_classes=num_classes)
    assert load_pretrained_vit(model) == 198

    adapter_parameters = 12 * 2 * 768 * HRA_RANK
    classifier_parameters = 768 * num_classes + num_classes
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable_parameters == adapter_parameters + classifier_parameters

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(name.endswith("householder_vectors") or name.startswith("head.classifier.") for name in trainable_names)
    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [HRA_LEARNING_RATE, CLASSIFIER_LEARNING_RATE]
    assert build_scheduler(optimizer, num_epochs=100, warmup_epochs=10) is not None


if __name__ == "__main__":
    test_hra_layer_math()
    test_hra_model_and_optimizer()
    print("Mini-VTAB HRA tests passed.")
