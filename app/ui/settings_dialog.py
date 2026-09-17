from __future__ import annotations

import os

import keyring
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from app.ocr.cloud_gemini import DEFAULT_GEMINI_MODEL


KEYRING_SERVICE = "BoreholeJournalOCR"
KEYRING_USER = "gemini-api-key"


def get_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or ""
    except Exception:
        return ""


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки распознавания")
        self.setMinimumWidth(520)
        settings = QSettings()
        self.api_key = QLineEdit(get_api_key())
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Ключ из Google AI Studio")
        self.model = QLineEdit(settings.value("gemini/model", DEFAULT_GEMINI_MODEL))

        form = QFormLayout()
        form.addRow("Gemini API-ключ:", self.api_key)
        form.addRow("Модель Gemini:", self.model)
        note = QLabel(
            "Ключ сохраняется в системном хранилище учётных данных. "
            "Переменные GEMINI_API_KEY и GOOGLE_API_KEY имеют приоритет.<br>"
            '<a href="https://aistudio.google.com/app/apikey">Получить ключ в Google AI Studio</a>'
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def accept(self) -> None:
        QSettings().setValue("gemini/model", self.model.text().strip() or DEFAULT_GEMINI_MODEL)
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, self.api_key.text().strip())
        except Exception as exc:
            self.api_key.setToolTip(f"Не удалось сохранить ключ: {exc}")
        super().accept()
