from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import RecognizedPage


def run_local_ocr_process(
    aligned_rgb: np.ndarray,
    timeout_seconds: int = 900,
    progress_callback: Callable[[str], None] | None = None,
) -> RecognizedPage:
    """Run PaddleOCR outside the GUI process so native failures cannot close Qt."""
    project_root, python_exe = _find_worker_python()
    with tempfile.TemporaryDirectory(prefix="borehole_ocr_") as temporary:
        temp_dir = Path(temporary)
        input_path = temp_dir / "page.png"
        output_path = temp_dir / "result.json"
        error_path = temp_dir / "error.txt"
        progress_path = temp_dir / "progress.txt"
        console_path = temp_dir / "console.txt"
        Image.fromarray(aligned_rgb).save(input_path, format="PNG")

        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONPATH"] = str(project_root)
        environment["FLAGS_use_mkldnn"] = "0"
        environment["FLAGS_use_onednn"] = "0"
        command = [
            str(python_exe),
            "-m",
            "app.ocr.worker_process",
            str(input_path),
            str(output_path),
            str(error_path),
            str(progress_path),
        ]
        flags = 0  # раньше было CREATE_NO_WINDOW — временно отключено для диагностики
        started = time.monotonic()
        last_progress = ""
        with console_path.open("w", encoding="utf-8", errors="replace") as console_stream:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=None,
                stderr=None,,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            while process.poll() is None:
                if output_path.exists():
                    try:
                        result = RecognizedPage.model_validate_json(
                            output_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        pass
                    else:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return result
                if progress_path.exists():
                    progress = progress_path.read_text(encoding="utf-8", errors="replace").strip()
                    if progress and progress != last_progress:
                        last_progress = progress
                        if progress_callback is not None:
                            progress_callback(progress)
                if time.monotonic() - started > timeout_seconds:
                    process.kill()
                    process.wait()
                    raise TimeoutError(
                        "Локальное OCR не завершилось за 15 минут. Повторно запустите "
                        "install_local_ocr.bat и проверьте доступ к интернету."
                    )
                time.sleep(0.25)
            return_code = process.returncode

        if return_code != 0:
            details = error_path.read_text(encoding="utf-8", errors="replace") if error_path.exists() else ""
            console = console_path.read_text(encoding="utf-8", errors="replace") if console_path.exists() else ""
            suffix = details.strip() or console.strip() or "Нативный процесс PaddleOCR завершился без сообщения."
            raise RuntimeError(
                f"Локальный OCR завершился с кодом {return_code}.\n\n{suffix[-5000:]}"
            )
        if not output_path.exists():
            raise RuntimeError("Локальный OCR не создал файл результата.")
        return RecognizedPage.model_validate_json(output_path.read_text(encoding="utf-8"))


def _find_worker_python() -> tuple[Path, Path]:
    override = os.getenv("BOREHOLE_OCR_PYTHON", "").strip()
    if override:
        executable = Path(override).expanduser().resolve()
        if executable.is_file():
            root = executable.parents[2] if len(executable.parents) > 2 else Path.cwd()
            return root, executable

    candidates: list[Path] = [Path.cwd(), Path(__file__).resolve().parents[2]]
    executable_path = Path(sys.executable).resolve()
    candidates.extend(executable_path.parents[:4])
    checked: list[str] = []
    for root in candidates:
        root = root.resolve()
        if str(root) in checked:
            continue
        checked.append(str(root))
        python_exe = root / ".venv" / "Scripts" / "python.exe"
        worker_module = root / "app" / "ocr" / "worker_process.py"
        if python_exe.is_file() and worker_module.is_file():
            return root, python_exe
    raise RuntimeError(
        "Не найдена среда локального OCR. Запускайте EXE из папки "
        "dist\\BoreholeJournalOCR внутри проекта и предварительно выполните install_local_ocr.bat."
    )
