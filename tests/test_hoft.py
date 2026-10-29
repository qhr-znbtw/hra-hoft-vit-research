"""Correctness and integration tests for Query/Value HOFT."""

import torch
from torch.nn import functional as F

from experiments.hoft.config import CLASSIFIER_LEARNING_RATE, HOFT_LEARNING_RATE, HOFT_RANK
from experiments.hoft.hoft_layer import HOFTLinear
from experiments.hoft.model import ViTHOFTClassifier
from experiments.hoft.run_hoft import build_optimizer, build_scheduler
from vit_core.pretrained import load_pretrained_vit


def test_hoft_layer_math() -> None:
    torch.manual_seed(0)
    layer = HOFTLinear(64, 48, rank=HOFT_RANK)
    inputs = torch.randn(3, 5, 64)
    dense_output = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    torch.testing.assert_close(layer(inputs), dense_output, atol=2e-5, rtol=2e-5)

    # The paper's first-order inverse is approximate, so Q is nearly rather
    # than exactly orthogonal. This guards against a broken, exploding factor.
    left = layer.dense_matrix(layer.left_vectors)
    relative_error = torch.linalg.norm(left.T @ left - torch.eye(48)) / torch.linalg.norm(torch.eye(48))
    assert relative_error < 0.25

    layer(inputs).sum().backward()
    assert layer.left_vectors.grad is not None
    assert layer.right_vectors.grad is not None
    assert torch.isfinite(layer.left_vectors.grad).all()
    assert torch.isfinite(layer.right_vectors.grad).all()


def test_hoft_model_and_optimizer() -> None:
    num_classes = 10
    model = ViTHOFTClassifier(num_classes=num_classes)
    assert load_pretrained_vit(model) == 198

    adapter_parameters = 12 * 2 * (768 + 768) * HOFT_RANK
    classifier_parameters = 768 * num_classes + num_classes
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable_parameters == adapter_parameters + classifier_parameters

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(
        name.endswith("left_vectors") or name.endswith("right_vectors") or name.startswith("head.classifier.")
        for name in trainable_names
    )
    optimizer = build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [HOFT_LEARNING_RATE, CLASSIFIER_LEARNING_RATE]
    assert build_scheduler(optimizer, num_epochs=100, warmup_epochs=10) is not None


if __name__ == "__main__":
    test_hoft_layer_math()
    test_hoft_model_and_optimizer()
    print("Mini-VTAB HOFT tests passed.")
