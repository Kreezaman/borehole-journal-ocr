from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    finished = Signal()


class FunctionWorker(QRunnable):
    def __init__(self, fn: Callable[..., Any], with_progress: bool = False):
        super().__init__()
        self.fn = fn
        self.with_progress = with_progress
        self.signals = WorkerSignals()
        # Не даём QThreadPool удалять объект автоматически: иначе
        # объект может быть уничтожен до того, как GUI-поток
        # обработает отправленные сигналы.
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            if self.with_progress:
                result = self.fn(self.signals.progress.emit)
            else:
                result = self.fn()
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.signals.finished.emit()