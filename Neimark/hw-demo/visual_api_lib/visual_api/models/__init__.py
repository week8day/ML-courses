"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from .base_model import Model
from .image_model import ImageModel
from .classification import Classification
from .detection import Detection

__all__ = [
    "Model",
    "ImageModel",
    "Classification",
	"Detection"
]
