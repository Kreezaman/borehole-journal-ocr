from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
)


class OCRProgressDialog(QDialog):
    """Visible heartbeat for OCR APIs that do not report percentage progress."""

    def __init__(self, parent=None, timeout_seconds: int = 120):
        super().__init__(parent)
        self.timeout_seconds = timeout_seconds
        self._last_message = ""
        self._last_wait_attempt = ""
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_time)

        self.setWindowTitle("Gemini OCR — выполнение запроса")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(560)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self.stage_label = QLabel("Подготовка изображения…")
        self.stage_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note = QLabel(
            "Полоса движется, пока программа ожидает Gemini. "
            "При временном лимите появится обратный отсчёт до автоматического повтора."
        )
        self.note.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(30)
        self.log.setFixedHeight(105)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.time_label)
        layout.addWidget(self.note)
        layout.addWidget(self.log)

    def start(self, initial_message: str) -> None:
        self._elapsed.start()
        self._timer.start()
        self.set_stage(initial_message)
        self._update_time()
        self.show()

    def set_stage(self, message: str) -> None:
        message = message.strip()
        if not message or message == self._last_message:
            return
        self._last_message = message
        self.stage_label.setText(message)
        seconds = max(0, self._elapsed.elapsed() // 1000) if self._elapsed.isValid() else 0
        if message.startswith("⏳"):
            attempt = message.split("(", 1)[-1] if "(" in message else message
            if attempt == self._last_wait_attempt and "через 1 сек" not in message:
                return
            self._last_wait_attempt = attempt
        self.log.appendPlainText(f"{_clock(seconds)}  {message}")

    def finish(self) -> None:
        self._timer.stop()
        super().accept()
        self.deleteLater()

    def _update_time(self) -> None:
        elapsed = max(0, self._elapsed.elapsed() // 1000)
        remaining = max(0, self.timeout_seconds - elapsed)
        self.time_label.setText(
            f"Прошло: {_clock(elapsed)}    •    "
            f"предельное время операции: {_clock(remaining)}"
        )

    def reject(self) -> None:
        # The synchronous HTTPS call cannot be safely interrupted from the GUI
        # thread. It is bounded by a short hard timeout instead.
        return


def _clock(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"
