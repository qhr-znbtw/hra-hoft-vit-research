"""Configuration for the Mini-VTAB ViT-B/16 HOFT experiments."""

# HOFT has vectors on both sides of W. Rank 4 therefore matches the adapter
# parameter count of rank-8 HRA for square 768 x 768 Query/Value projections.
HOFT_RANK = 4
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10
HOFT_LEARNING_RATE = 1e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
