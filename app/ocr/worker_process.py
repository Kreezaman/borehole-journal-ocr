from __future__ import annotations

import os
import sys
from datetime import datetime

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


def _log_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BoreholeJournalOCR"
    base.mkdir(parents=True, exist_ok=True)
    return base


_timeline_path = _log_dir() / "worker_timeline.txt"


def _log_timeline(message: str) -> None:
    try:
        with _timeline_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    except Exception:
        pass


def main(arguments: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if len(args) not in (3, 4):
        return 2
    input_path, output_path, error_path = map(Path, args[:3])
    progress_path = Path(args[3]) if len(args) == 4 else None
    warnings.filterwarnings("ignore", message="No ccache found.*")

    try:
        _timeline_path.write_text(
            f"=== Запуск worker: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    try:
        _progress(progress_path, "Локальное OCR: загрузка модулей…")
        _log_timeline("Старт: импорт PaddleLocalOCR")
        from app.ocr.local_paddle import PaddleLocalOCR

        _progress(progress_path, "Локальное OCR: подготовка моделей…")
        _log_timeline("Импорт завершён, создаём движок")
        engine = PaddleLocalOCR()
        _log_timeline("Движок создан, читаем картинку")
        image = np.asarray(Image.open(input_path).convert("RGB"))
        _log_timeline(f"Картинка загружена: {image.shape}")
        _progress(progress_path, "Локальное OCR: распознавание…")
        result = engine.recognize(image, lambda text: _progress(progress_path, text))
        _log_timeline("recognize() завершён")
        _progress(progress_path, "Локальное OCR: сохранение результата…")
        temporary_output = output_path.with_suffix(".tmp")
        temporary_output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary_output.replace(output_path)
        _log_timeline("Результат сохранён, worker завершается успешно")
        return 0
    except BaseException:
        tb = traceback.format_exc()
        _log_timeline(f"ОШИБКА:\n{tb}")
        try:
            error_path.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        try:
            fallback = _log_dir() / "worker_error.txt"
            fallback.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        return 1


def _progress(path: Path | None, message: str) -> None:
    _log_timeline(message)
    if path is not None:
        path.write_text(message, encoding="utf-8")


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)