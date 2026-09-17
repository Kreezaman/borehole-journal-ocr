from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from openpyxl import load_workbook
from PIL import Image

from app.core.alignment import PageAligner
from app.core.export_xlsx import export_project
from app.core.project_io import load_project, save_project
from app.models import HeaderData, JournalProject, JournalRow, ProjectPage, RecognizedPage
from app.ocr.cloud_gemini import GeminiCloudOCR
from app.ocr.local_paddle import PaddleLocalOCR, _interval_index


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_entrypoint_diverts_multiprocessing_before_gui_import() -> None:
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert source.index("multiprocessing.freeze_support()") < source.index("from app.ui.main_window import run_app")


def test_alignment_of_template() -> None:
    template = np.asarray(Image.open(ROOT / "app/assets/template.png").convert("RGB"))
    result = PageAligner(template).align(template)
    assert result.image.shape == template.shape
    assert result.quality >= 0.8


def test_project_roundtrip(tmp_path: Path) -> None:
    project = JournalProject(
        source_path="sample.pdf",
        pages=[
            ProjectPage(
                page_index=0,
                source_name="sample.pdf — стр. 1",
                source_path="sample.pdf",
                data=RecognizedPage(
                    header=HeaderData(borehole_no="17"),
                    rows=[JournalRow(depth_from="0", depth_to="0,1", description="ПРС")],
                ),
            )
        ],
    )
    output = tmp_path / "sample.bjproj"
    save_project(project, output)
    restored = load_project(output)
    assert restored.pages[0].data.header.borehole_no == "17"
    assert restored.pages[0].data.rows[0].description == "ПРС"


def test_excel_export(tmp_path: Path) -> None:
    project = JournalProject(
        pages=[
            ProjectPage(
                page_index=0,
                source_name="one",
                data=RecognizedPage(
                    header=HeaderData(borehole_no="18А"),
                    rows=[JournalRow(depth_from="0", depth_to="2,2", description="Песок мелкий")],
                ),
            )
        ]
    )
    output = tmp_path / "result.xlsx"
    export_project(project, output)
    workbook = load_workbook(output)
    sheet = workbook[workbook.sheetnames[0]]
    assert sheet["B3"].value == "18А"
    assert sheet["E8"].value == "Песок мелкий"


def test_local_ocr_disables_onednn(monkeypatch) -> None:
    received: dict = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))
    PaddleLocalOCR()
    assert received["enable_mkldnn"] is False
    assert received["device"] == "cpu"
    assert received["text_recognition_model_name"] == "cyrillic_PP-OCRv5_server_rec"


def test_interval_mapping_uses_template_scale() -> None:
    assert _interval_index(50.0, [0, 100, 200], 1.0) == 0
    assert _interval_index(150.0, [0, 100, 200], 1.0) == 1
    assert _interval_index(250.0, [0, 100, 200], 1.0) is None


def test_local_worker_forces_native_process_exit() -> None:
    source = (ROOT / "app/ocr/worker_process.py").read_text(encoding="utf-8")
    assert "temporary_output.replace(output_path)" in source
    assert "os._exit(exit_code)" in source


def test_gemini_ocr_uses_inline_image_and_structured_output() -> None:
    received: dict = {}
    progress: list[str] = []

    class FakeInteractions:
        def create(self, **kwargs):
            received.update(kwargs)
            return SimpleNamespace(
                output_text='''{
                    "header": {"borehole_no": "2"},
                    "rows": [{"depth_from": "0", "depth_to": "0,1", "description": "ПРС"}],
                    "footer": {"drilling_depth": "10"},
                    "page_notes": ""
                }'''
            )

    client = SimpleNamespace(interactions=FakeInteractions())
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    result = GeminiCloudOCR("test-key", client=client).recognize(image, progress.append)

    assert received["model"] == "gemini-3.8-flash"
    assert received["input"][1]["type"] == "image"
    assert received["input"][1]["mime_type"] == "image/jpeg"
    assert received["response_format"]["mime_type"] == "application/json"
    assert result.header.borehole_no == "2"
    assert result.rows[0].description == "ПРС"
    assert len(result.rows) == 19
    assert result.recognition_mode == "gemini"
    assert progress[0].startswith("Этап 1 из 3")
    assert progress[-1].startswith("Этап 3 из 3")


def test_project_still_accepts_legacy_cloud_mode() -> None:
    assert RecognizedPage(recognition_mode="cloud").recognition_mode == "cloud"
