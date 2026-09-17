from __future__ import annotations

from pathlib import Path

import pymupdf
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class DocumentLoader:
    def __init__(self, path: str | Path, dpi: int = 160):
        self.path = Path(path)
        self.dpi = dpi
        self._pdf: pymupdf.Document | None = None
        if self.path.suffix.lower() == ".pdf":
            self._pdf = pymupdf.open(self.path)
            self.page_count = len(self._pdf)
        elif self.path.suffix.lower() in SUPPORTED_IMAGES:
            self.page_count = 1
        else:
            raise ValueError(f"Неподдерживаемый формат: {self.path.suffix}")

    def render_page(self, page_index: int) -> np.ndarray:
        if not 0 <= page_index < self.page_count:
            raise IndexError(page_index)
        if self._pdf is not None:
            page = self._pdf[page_index]
            scale = self.dpi / 72.0
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            image = Image.open(self.path).convert("RGB")
            image = ImageOps.exif_transpose(image)
        if image.height > image.width:
            image = image.rotate(90, expand=True)
        return np.asarray(image)

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def __enter__(self) -> "DocumentLoader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
