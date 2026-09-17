from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from app.core.alignment import erase_grid_lines
from app.core.layout import (
    FOOTER_RECTS,
    HEADER_RECTS,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    TABLE_FIELDS,
    TABLE_X,
    TABLE_Y,
)
from app.models import FooterData, HeaderData, JournalRow, RecognizedPage
from app.ocr.base import OCRProvider


class PaddleLocalOCR(OCRProvider):
    def __init__(self):
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_onednn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        try:
            import paddle
            paddle.set_flags({
                "FLAGS_use_mkldnn": False,
                "FLAGS_use_onednn": False,
            })
        except Exception:
            pass

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "Локальный OCR не установлен. Запустите «Установить_локальный_OCR.bat»."
            ) from exc

        self.engine = PaddleOCR(
            lang="ru",
            text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
        )

    def recognize(
        self,
        aligned_rgb: np.ndarray,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RecognizedPage:
        height, width = aligned_rgb.shape[:2]
        x_scale = width / REFERENCE_WIDTH
        y_scale = height / REFERENCE_HEIGHT

        _emit(progress_callback, "Локальное OCR: распознавание шапки (1 из 10)…")
        header_box = (0, max(0, round(10 * y_scale)), width, min(height, round(90 * y_scale)))
        header_crop = _crop(aligned_rgb, header_box)
        header_texts, header_boxes = self._recognize_crop(header_crop)
        header_items = _offset_items(header_texts, header_boxes, header_box[0], header_box[1])
        header_values = {
            name: _text_inside(header_items, rect.pixels(width, height))
            for name, rect in HEADER_RECTS.items()
        }

        dates = _split_dates(header_values.pop("drilling_dates", ""))
        header = HeaderData(
            **header_values,
            drilling_start=dates[0],
            drilling_end=dates[1],
        )

        rows = [JournalRow() for _ in range(19)]
        body_top = round(TABLE_Y[0] * y_scale)
        body_bottom = round(TABLE_Y[-1] * y_scale)
        row_height = (body_bottom - body_top) / 19.0
        for column, field_name in enumerate(TABLE_FIELDS):
            _emit(
                progress_callback,
                f"Локальное OCR: столбец «{field_name}» ({column + 2} из 10)…",
            )
            x1 = max(0, round(TABLE_X[column] * x_scale) + 2)
            x2 = min(width, round(TABLE_X[column + 1] * x_scale) - 2)
            strip = erase_grid_lines(aligned_rgb[body_top:body_bottom, x1:x2])
            texts, boxes = self._recognize_crop(strip)
            grouped: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
            for text, box in zip(texts, boxes):
                value = text.strip()
                if not value:
                    continue
                x_center = (box[0] + box[2]) / 2.0
                y_center = (box[1] + box[3]) / 2.0
                row_index = min(18, max(0, int(y_center / max(1.0, row_height))))
                grouped[row_index].append((y_center, x_center, value))
            for row_index, pieces in grouped.items():
                value = " ".join(piece[2] for piece in sorted(pieces))
                setattr(rows[row_index], field_name, value)

        _emit(progress_callback, "Локальное OCR: распознавание подвала (10 из 10)…")
footer_box = (0, max(0, round(690 * y_scale)), width, min(height, round(870 * y_scale)))
footer_crop = _crop(aligned_rgb, footer_box)
# Подвал самый широкий — разбиваем его на 3 горизонтальные части,
# чтобы детектор не захлёбывался на полосе во всю ширину.
footer_texts: list[str] = []
footer_boxes: list[list[float]] = []
chunks = 3
chunk_w = footer_crop.shape[1] // chunks
for _i in range(chunks):
    cx1 = _i * chunk_w
    cx2 = footer_crop.shape[1] if _i == chunks - 1 else (_i + 1) * chunk_w
    piece = footer_crop[:, cx1:cx2]
    piece_texts, piece_boxes = self._recognize_crop(piece)
    for text, box in zip(piece_texts, piece_boxes):
        footer_texts.append(text)
        footer_boxes.append([box[0] + cx1, box[1], box[2] + cx1, box[3]])
footer_items = _offset_items(footer_texts, footer_boxes, footer_box[0], footer_box[1])
        footer_values = {
            name: _text_inside(footer_items, rect.pixels(width, height))
            for name, rect in FOOTER_RECTS.items()
        }

        return RecognizedPage(
            header=header,
            rows=rows,
            footer=FooterData(**footer_values),
            recognition_mode="local",
        )

    def _recognize_crop(self, crop_rgb: np.ndarray) -> tuple[list[str], list[list[float]]]:
        if crop_rgb.size == 0:
            return [], []
        enhanced = _enhance_handwriting(crop_rgb)
        height, width = crop_rgb.shape[:2]
        base_scale = 2.5 if height < 400 else 2.0
        max_scale = 3800.0 / max(width, 1)
        scale = max(1.0, min(base_scale, max_scale))
        enlarged = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result = self.engine.predict(enlarged)
        texts: list[str] = []
        boxes: list[list[float]] = []
        for item in result:
            payload = _result_payload(item)
            rec_texts = payload.get("rec_texts", [])
            rec_scores = payload.get("rec_scores", [])
            rec_boxes = payload.get("rec_boxes", [])
            for index, text in enumerate(rec_texts):
                score = float(rec_scores[index]) if index < len(rec_scores) else 1.0
                if score < 0.38:
                    continue
                texts.append(str(text))
                if index < len(rec_boxes):
                    raw = list(map(float, rec_boxes[index]))
                    boxes.append([value / scale for value in raw[:4]])
                else:
                    boxes.append([0.0, float(index), 1.0, float(index + 1)])
        order = sorted(range(len(texts)), key=lambda i: (boxes[i][1], boxes[i][0]))
        texts = [texts[i] for i in order]
        boxes = [boxes[i] for i in order]
        return texts, boxes


def _result_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        payload = item
    else:
        payload = getattr(item, "json", {})
        if callable(payload):
            payload = payload()
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _offset_items(
    texts: list[str], boxes: list[list[float]], offset_x: float, offset_y: float
) -> list[tuple[str, list[float]]]:
    return [
        (text, [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y])
        for text, box in zip(texts, boxes)
    ]


def _text_inside(items: list[tuple[str, list[float]]], region: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = region
    selected: list[tuple[float, float, str]] = []
    for text, box in items:
        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0
        if x1 <= x_center <= x2 and y1 <= y_center <= y2 and text.strip():
            selected.append((box[1], box[0], text.strip()))
    return " ".join(item[2] for item in sorted(selected))


def _interval_index(value: float, boundaries: list[int], scale: float) -> int | None:
    for index in range(len(boundaries) - 1):
        if boundaries[index] * scale <= value <= boundaries[index + 1] * scale:
            return index
    return None


def _enhance_handwriting(image_rgb: np.ndarray) -> np.ndarray:
    """Increase contrast of faint pencil/pen strokes without hard binarization."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clahe, 1.45, blurred, -0.45, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def _crop(image: np.ndarray, box: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    return image[y1:y2, x1:x2]


def _split_dates(value: str) -> tuple[str, str]:
    text = value.strip()
    for separator in (" / ", "/", " - ", "—"):
        if separator in text:
            left, right = text.split(separator, 1)
            return left.strip(), right.strip()
    return text, ""
