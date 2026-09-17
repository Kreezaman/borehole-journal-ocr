from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlays: list[QGraphicsRectItem] = []

    def set_image(self, rgb: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(rgb)
        height, width, channels = contiguous.shape
        image = QImage(contiguous.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.scene().clear()
        self._overlays.clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlays(self, rectangles: list[tuple[tuple[int, int, int, int], str, int]]) -> None:
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()
        colors = {
            "blue": QColor(39, 132, 196, 190),
            "orange": QColor(242, 139, 48, 220),
            "red": QColor(210, 55, 55, 220),
        }
        for (x1, y1, x2, y2), color_name, width in rectangles:
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            item.setPen(QPen(colors.get(color_name, colors["blue"]), width))
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(10)
            self.scene().addItem(item)
            self._overlays.append(item)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap_item is not None and self.transform().m11() == 1.0:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
