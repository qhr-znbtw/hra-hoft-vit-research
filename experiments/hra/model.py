"""ViT-B/16 classifier with HRA in Query and Value projections."""

from functools import partial

from experiments.hra.config import HRA_RANK
from experiments.hra.hra_layer import HRALinear
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTHRAClassifier(ViTForImageClassification):
    """Train Query/Value HRA vectors and a task-specific classifier."""

    def __init__(self, num_classes: int) -> None:
        hra_factory = partial(HRALinear, rank=HRA_RANK)
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=hra_factory,
            value_factory=hra_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, HRALinear):
                module.householder_vectors.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
