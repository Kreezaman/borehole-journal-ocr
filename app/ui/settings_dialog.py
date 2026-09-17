from __future__ import annotations

import os

import keyring
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
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
LEGACY_GEMINI_MODELS = {"gemini-3.8-flash"}


def get_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or ""
    except Exception:
        return ""


def get_gemini_model() -> str:
    settings = QSettings()
    model = str(settings.value("gemini/model", DEFAULT_GEMINI_MODEL)).strip()
    if not model or model in LEGACY_GEMINI_MODELS:
        model = DEFAULT_GEMINI_MODEL
        settings.setValue("gemini/model", model)
    return model


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки распознавания")
        self.setMinimumWidth(520)
        self.api_key = QLineEdit(get_api_key())
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Ключ из Google AI Studio")
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItem("Gemini 3.1 Flash-Lite — бесплатно, рекомендуется", DEFAULT_GEMINI_MODEL)
        current_model = get_gemini_model()
        if current_model != DEFAULT_GEMINI_MODEL:
            self.model.addItem(current_model, current_model)
            self.model.setCurrentIndex(1)

        form = QFormLayout()
        form.addRow("Gemini API-ключ:", self.api_key)
        form.addRow("Модель Gemini:", self.model)
        note = QLabel(
            "Gemini 3.1 Flash-Lite выбран как основной бесплатный режим. "
            "Другое имя модели можно ввести вручную.<br>"
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
        model = self.model.currentData()
        if self.model.isEditable() and self.model.currentText() != self.model.itemText(
            self.model.currentIndex()
        ):
            model = self.model.currentText().strip()
        QSettings().setValue("gemini/model", model or DEFAULT_GEMINI_MODEL)
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, self.api_key.text().strip())
        except Exception as exc:
            self.api_key.setToolTip(f"Не удалось сохранить ключ: {exc}")
        super().accept()
