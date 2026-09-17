from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class HeaderData(BaseModel):
    borehole_no: str = ""
    drilling_start: str = ""
    drilling_end: str = ""
    object_name: str = ""
    coordinate_n: str = ""
    coordinate_e: str = ""


class JournalRow(BaseModel):
    layer_no: str = ""
    depth_from: str = ""
    depth_to: str = ""
    frozen_interval: str = ""
    description: str = ""
    sample_no: str = ""
    sample_depth: str = ""
    water: str = ""


class FooterData(BaseModel):
    groundwater_exposed: str = ""
    groundwater_stabilized: str = ""
    specific_odor: str = ""
    drilling_rig: str = ""
    drilling_diameter: str = ""
    drilling_depth: str = ""
    drill_master: str = ""
    engineer_geologist: str = ""
    sketch_notes: str = ""


class OCRWarning(BaseModel):
    field: str = ""
    reason: str = ""
    alternative: str = ""


class RecognizedPage(BaseModel):
    header: HeaderData = Field(default_factory=HeaderData)
    rows: list[JournalRow] = Field(default_factory=lambda: [JournalRow() for _ in range(19)])
    footer: FooterData = Field(default_factory=FooterData)
    page_notes: str = ""
    ocr_warnings: list[OCRWarning] = Field(default_factory=list)
    recognition_mode: Literal["none", "local", "cloud", "gemini"] = "none"
    alignment_quality: float = 0.0

    def normalized_rows(self, count: int = 19) -> list[JournalRow]:
        rows = list(self.rows[:count])
        rows.extend(JournalRow() for _ in range(count - len(rows)))
        return rows


class ProjectPage(BaseModel):
    page_index: int
    source_name: str
    source_path: str = ""
    source_page: int = 0
    data: RecognizedPage = Field(default_factory=RecognizedPage)


class JournalProject(BaseModel):
    format_version: int = 1
    source_path: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pages: list[ProjectPage] = Field(default_factory=list)

    def touch(self) -> None:
        self.modified_at = datetime.now(timezone.utc).isoformat()

    @property
    def display_name(self) -> str:
        return Path(self.source_path).name if self.source_path else "Новый проект"
