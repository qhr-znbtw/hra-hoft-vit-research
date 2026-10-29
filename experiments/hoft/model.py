"""ViT-B/16 classifier with HOFT in Query and Value projections."""

from functools import partial

from experiments.hoft.config import HOFT_RANK
from experiments.hoft.hoft_layer import HOFTLinear
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


class ViTHOFTClassifier(ViTForImageClassification):
    """Train Query/Value HOFT vectors and a task-specific classifier."""

    def __init__(self, num_classes: int) -> None:
        hoft_factory = partial(HOFTLinear, rank=HOFT_RANK)
        super().__init__(
            num_classes=num_classes,
            config=VIT_BASE_PATCH16_224,
            query_factory=hoft_factory,
            value_factory=hoft_factory,
        )
        self.set_trainable_parameters()

    def set_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, HOFTLinear):
                module.left_vectors.requires_grad = True
                module.right_vectors.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True
