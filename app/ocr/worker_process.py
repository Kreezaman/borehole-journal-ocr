from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


def main(arguments: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if len(args) not in (3, 4):
        return 2
    input_path, output_path, error_path = map(Path, args[:3])
    progress_path = Path(args[3]) if len(args) == 4 else None
    warnings.filterwarnings("ignore", message="No ccache found.*")
    try:
        _progress(progress_path, "Локальное OCR: загрузка модулей…")
        from app.ocr.local_paddle import PaddleLocalOCR

        image = np.asarray(Image.open(input_path).convert("RGB"))
        _progress(progress_path, "Локальное OCR: подготовка моделей…")
        engine = PaddleLocalOCR()
        result = engine.recognize(image, lambda text: _progress(progress_path, text))
        _progress(progress_path, "Локальное OCR: сохранение результата…")
        temporary_output = output_path.with_suffix(".tmp")
        temporary_output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary_output.replace(output_path)
        return 0
    except BaseException:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


def _progress(path: Path | None, message: str) -> None:
    if path is not None:
        path.write_text(message, encoding="utf-8")


if __name__ == "__main__":
    exit_code = main()
    # Paddle/PaddleX can leave non-daemon native worker threads alive on
    # Windows. All result files are already closed here, so bypass teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
