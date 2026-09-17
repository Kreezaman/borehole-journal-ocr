from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RectN:
    """Rectangle with coordinates normalized to the reference image size."""

    x1: float
    y1: float
    x2: float
    y2: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )


REFERENCE_WIDTH = 1293
REFERENCE_HEIGHT = 895


def nr(x1: int, y1: int, x2: int, y2: int) -> RectN:
    return RectN(x1 / REFERENCE_WIDTH, y1 / REFERENCE_HEIGHT, x2 / REFERENCE_WIDTH, y2 / REFERENCE_HEIGHT)


HEADER_RECTS = {
    "borehole_no": nr(126, 22, 275, 56),
    "drilling_dates": nr(594, 20, 852, 58),
    "object_name": nr(83, 50, 956, 82),
    "coordinate_n": nr(1100, 20, 1272, 51),
    "coordinate_e": nr(1100, 48, 1272, 79),
}

FOOTER_RECTS = {
    "groundwater_exposed": nr(172, 727, 242, 760),
    "groundwater_stabilized": nr(172, 752, 242, 786),
    "specific_odor": nr(24, 785, 243, 844),
    "drilling_rig": nr(460, 704, 679, 736),
    "drilling_diameter": nr(460, 729, 679, 761),
    "drilling_depth": nr(460, 754, 679, 787),
    "drill_master": nr(460, 794, 679, 825),
    "engineer_geologist": nr(460, 818, 679, 852),
    "sketch_notes": nr(690, 693, 1273, 866),
}

TABLE_X = [14, 44, 109, 174, 244, 1085, 1118, 1215, 1273]
TABLE_Y = [193, 219, 245, 271, 297, 323, 349, 375, 401, 427, 453, 480, 506, 532, 558, 584, 610, 636, 662, 689]
TABLE_FIELDS = [
    "layer_no",
    "depth_from",
    "depth_to",
    "frozen_interval",
    "description",
    "sample_no",
    "sample_depth",
    "water",
]


def table_cell_rect(row: int, column: int) -> RectN:
    return nr(TABLE_X[column], TABLE_Y[row], TABLE_X[column + 1], TABLE_Y[row + 1])


def template_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "template.png"
