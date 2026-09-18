from __future__ import annotations

from collections.abc import Callable
from threading import Event
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


class SequentialWorkerSignals(QObject):
    item_started = Signal(object, int, int)
    item_result = Signal(object, object, int, int)
    error = Signal(object, str, int, int)
    progress = Signal(str)
    finished = Signal(int, bool)


class SequentialWorker(QRunnable):
    """Run items one at a time and allow a safe stop between items."""

    def __init__(
        self,
        items: list[Any],
        fn: Callable[[Any, Callable[[str], None]], Any],
        cleanup: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.items = list(items)
        self.fn = fn
        self.cleanup = cleanup
        self.signals = SequentialWorkerSignals()
        self._stop_requested = Event()
        self.setAutoDelete(False)

    def request_stop(self) -> None:
        self._stop_requested.set()

    @Slot()
    def run(self) -> None:
        completed = 0
        total = len(self.items)
        try:
            for position, item in enumerate(self.items, start=1):
                if self._stop_requested.is_set():
                    break
                self.signals.item_started.emit(item, position, total)

                def emit_progress(message: str, *, _position: int = position) -> None:
                    self.signals.progress.emit(
                        f"Страница {_position} из {total}: {message}"
                    )

                try:
                    result = self.fn(item, emit_progress)
                except Exception as exc:
                    self.signals.error.emit(
                        item,
                        f"{type(exc).__name__}: {exc}",
                        position,
                        total,
                    )
                    break
                completed += 1
                self.signals.item_result.emit(item, result, position, total)
        finally:
            if self.cleanup is not None:
                try:
                    self.cleanup()
                except Exception:
                    pass
            self.signals.finished.emit(completed, self._stop_requested.is_set())
