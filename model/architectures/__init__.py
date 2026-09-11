"""SignalScope model architectures package."""

from model.architectures.convnext import ConvNeXtTinyDetector, build_convnext_tiny
from model.architectures.frequency import FrequencyFeatureExtractor
from model.architectures.fusion import DualBranchFusionDetector

__all__ = [
    "ConvNeXtTinyDetector",
    "build_convnext_tiny",
    "FrequencyFeatureExtractor",
    "DualBranchFusionDetector",
]
