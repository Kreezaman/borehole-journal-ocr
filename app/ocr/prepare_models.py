from __future__ import annotations

import os


def main() -> int:
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    print("Подготовка моделей PaddleOCR. Загрузка может занять несколько минут...")
    from app.ocr.local_paddle import PaddleLocalOCR

    PaddleLocalOCR()
    print("Модели локального OCR готовы.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
