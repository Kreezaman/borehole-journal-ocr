from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class OCRProgressDialog(QDialog):
    """Visible heartbeat for OCR APIs that do not report percentage progress."""

    stop_requested = Signal()

    def __init__(
        self,
        parent=None,
        timeout_seconds: int | None = 120,
        allow_stop: bool = False,
    ):
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
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.hide()
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
        self.stop_button = QPushButton("Остановить после текущей страницы")
        self.stop_button.clicked.connect(self._request_stop)
        self.stop_button.setVisible(allow_stop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.page_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.time_label)
        layout.addWidget(self.note)
        layout.addWidget(self.log)
        layout.addWidget(self.stop_button)

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

    def set_batch_page(
        self,
        position: int,
        total: int,
        completed: int,
        source_name: str,
    ) -> None:
        self.page_label.show()
        self.page_label.setText(
            f"Страница {position} из {total} · {source_name}"
        )
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Обработано: %v из %m")
        self.note.setText(
            "Страницы обрабатываются последовательно, чтобы бережно использовать "
            "бесплатный лимит Gemini. Остановка выполняется после текущей страницы."
        )

    def mark_batch_completed(self, completed: int) -> None:
        self.progress.setValue(completed)

    def _request_stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Остановка после текущей страницы…")
        self.set_stage("Запрошена остановка. Текущая страница будет завершена.")
        self.stop_requested.emit()

    def finish(self) -> None:
        self._timer.stop()
        super().accept()
        self.deleteLater()

    def _update_time(self) -> None:
        elapsed = max(0, self._elapsed.elapsed() // 1000)
        if self.timeout_seconds is None:
            self.time_label.setText(f"Общее время: {_clock(elapsed)}")
            return
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
