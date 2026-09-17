from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.models import RecognizedPage


class OCRProvider(ABC):
    @abstractmethod
    def recognize(self, aligned_rgb: np.ndarray) -> RecognizedPage:
        raise NotImplementedError
