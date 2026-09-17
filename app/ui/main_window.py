from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, QThreadPool, Qt
from PySide6.QtGui import QAction, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.alignment import PageAligner
from app.core.document import DocumentLoader
from app.core.export_xlsx import export_project
from app.core.layout import FOOTER_RECTS, HEADER_RECTS, TABLE_X, TABLE_Y, nr, template_path
from app.core.project_io import append_corrections, load_project, save_project
from app.core.validation import ValidationIssue, validate_page
from app.models import FooterData, HeaderData, JournalProject, JournalRow, ProjectPage, RecognizedPage
from app.ocr.cloud_gemini import DEFAULT_GEMINI_MODEL, GEMINI_TIMEOUT_SECONDS, GeminiCloudOCR
from app.ocr.local_process import run_local_ocr_process
from app.ui.image_view import ImageView
from app.ui.ocr_progress_dialog import OCRProgressDialog
from app.ui.settings_dialog import SettingsDialog, get_api_key
from app.ui.workers import FunctionWorker


TABLE_HEADERS = [
    "№ слоя",
    "От, м",
    "До, м",
    "Мерзлота / мощность",
    "Описание грунта",
    "№ образца",
    "Глубина отбора",
    "Воды",
]
ROW_FIELDS = [
    "layer_no",
    "depth_from",
    "depth_to",
    "frozen_interval",
    "description",
    "sample_no",
    "sample_depth",
    "water",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Буровой журнал OCR")
        self.resize(1600, 950)
        self.setMinimumSize(1100, 700)
        self.thread_pool = QThreadPool.globalInstance()
        self.project = JournalProject()
        self.project_path: Path | None = None
        self.current_index = -1
        self.loaders: dict[str, DocumentLoader] = {}
        self.aligned_cache: dict[int, np.ndarray] = {}
        self.recognition_baselines: dict[int, dict] = {}
        self.ocr_progress_dialog: OCRProgressDialog | None = None
        self.active_workers: list = []
        template_rgb = np.asarray(Image.open(template_path()).convert("RGB"))
        self.aligner = PageAligner(template_rgb)
        self._build_ui()
        self._build_toolbar()
        self._apply_style()
        self.statusBar().showMessage("Откройте PDF или изображения буровых журналов")

    def _build_ui(self) -> None:
        self.image_view = ImageView()
        self.image_view.setMinimumWidth(560)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_header_tab(), "Шапка")
        self.tabs.addTab(self._build_table_tab(), "Слои и образцы")
        self.tabs.addTab(self._build_footer_tab(), "УГВ и бурение")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self.tabs)
        splitter.setSizes([850, 750])

        self.prev_button = QPushButton("◀ Назад")
        self.next_button = QPushButton("Вперёд ▶")
        self.page_label = QLabel("Страница не открыта")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_button.clicked.connect(lambda: self.show_page(self.current_index - 1))
        self.next_button.clicked.connect(lambda: self.show_page(self.current_index + 1))
        navigation = QHBoxLayout()
        navigation.addWidget(self.prev_button)
        navigation.addWidget(self.page_label, 1)
        navigation.addWidget(self.next_button)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(navigation)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_header_tab(self) -> QWidget:
        self.header_edits = {
            "borehole_no": QLineEdit(),
            "drilling_start": QLineEdit(),
            "drilling_end": QLineEdit(),
            "object_name": QLineEdit(),
            "coordinate_n": QLineEdit(),
            "coordinate_e": QLineEdit(),
        }
        labels = {
            "borehole_no": "№ скважины",
            "drilling_start": "Дата начала",
            "drilling_end": "Дата окончания",
            "object_name": "Объект",
            "coordinate_n": "Координата N",
            "coordinate_e": "Координата E",
        }
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for name, edit in self.header_edits.items():
            form.addRow(labels[name], edit)
        note = QLabel("Пустое поле означает, что значение отсутствует. Знак ? — OCR не смог уверенно прочитать запись.")
        note.setWordWrap(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch(1)
        return widget

    def _build_table_tab(self) -> QWidget:
        self.table = QTableWidget(19, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setColumnWidth(0, 66)
        self.table.setColumnWidth(1, 78)
        self.table.setColumnWidth(2, 78)
        self.table.setColumnWidth(3, 135)
        self.table.setColumnWidth(4, 440)
        self.table.setColumnWidth(5, 90)
        self.table.setColumnWidth(6, 125)
        self.table.setColumnWidth(7, 90)
        for row in range(19):
            self.table.setRowHeight(row, 64)
            for column in range(len(TABLE_HEADERS)):
                self.table.setItem(row, column, QTableWidgetItem(""))
        self.table.itemSelectionChanged.connect(self._update_overlays)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        return widget

    def _build_footer_tab(self) -> QWidget:
        self.footer_edits = {
            "groundwater_exposed": QLineEdit(),
            "groundwater_stabilized": QLineEdit(),
            "specific_odor": QLineEdit(),
            "drilling_rig": QLineEdit(),
            "drilling_diameter": QLineEdit(),
            "drilling_depth": QLineEdit(),
            "drill_master": QLineEdit(),
            "engineer_geologist": QLineEdit(),
        }
        labels = {
            "groundwater_exposed": "УГВ вскрытый",
            "groundwater_stabilized": "УГВ установившийся",
            "specific_odor": "Специфический запах",
            "drilling_rig": "Буровая установка",
            "drilling_diameter": "Диаметр бурения",
            "drilling_depth": "Глубина бурения",
            "drill_master": "Буровой мастер",
            "engineer_geologist": "Инженер-геолог",
        }
        form = QFormLayout()
        for name, edit in self.footer_edits.items():
            form.addRow(labels[name], edit)
        self.sketch_notes = QTextEdit()
        self.sketch_notes.setPlaceholderText("Читаемые подписи в поле «Абрис»")
        self.page_notes = QTextEdit()
        self.page_notes.setPlaceholderText("Замечания оператора по странице")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addWidget(QLabel("Абрис / подписи"))
        layout.addWidget(self.sketch_notes, 1)
        layout.addWidget(QLabel("Замечания оператора"))
        layout.addWidget(self.page_notes, 1)
        return widget

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Основные действия", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        actions = [
            ("Открыть сканы", self.open_documents),
            ("Открыть проект", self.open_saved_project),
            ("Сохранить проект", self.save_current_project),
            ("Локальное OCR", self.recognize_local),
            ("Gemini OCR", self.recognize_cloud),
            ("Проверить", self.run_validation),
            ("Экспорт в Excel", self.export_excel),
            ("Настройки", self.open_settings),
        ]
        self.busy_actions: list[QAction] = []
        for text, slot in actions:
            action = QAction(text, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            self.busy_actions.append(action)
            if text in {"Сохранить проект", "Gemini OCR", "Проверить"}:
                toolbar.addSeparator()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef3f7; }
            QToolBar { background: #183a55; spacing: 5px; padding: 6px; border: 0; }
            QToolButton { color: white; background: #275b7e; padding: 8px 11px; border-radius: 4px; }
            QToolButton:hover { background: #347aa6; }
            QTabWidget::pane { background: white; border: 1px solid #b9c7d3; }
            QTabBar::tab { padding: 9px 14px; background: #dbe6ee; }
            QTabBar::tab:selected { background: white; color: #183a55; font-weight: bold; }
            QLineEdit, QTextEdit, QTableWidget { background: white; border: 1px solid #aebdca; }
            QLineEdit { padding: 7px; }
            QPushButton { padding: 7px 16px; }
            QHeaderView::section { background: #d7e6f0; padding: 6px; border: 1px solid #aebdca; font-weight: bold; }
            """
        )

    def open_documents(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Открыть буровые журналы",
            "",
            "Документы (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if paths:
            self._load_document_paths(paths)

    def _load_document_paths(self, paths: list[str]) -> None:
        self._close_loaders()
        self.project = JournalProject(source_path=";".join(paths))
        global_index = 0
        for raw_path in paths:
            path = str(Path(raw_path).resolve())
            loader = DocumentLoader(path)
            self.loaders[path] = loader
            for source_page in range(loader.page_count):
                self.project.pages.append(
                    ProjectPage(
                        page_index=global_index,
                        source_name=f"{Path(path).name} — стр. {source_page + 1}",
                        source_path=path,
                        source_page=source_page,
                    )
                )
                global_index += 1
        self.project_path = None
        self.aligned_cache.clear()
        self.recognition_baselines.clear()
        self.show_page(0)

    def open_saved_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект бурового журнала (*.bjproj)")
        if not path:
            return
        try:
            project = load_project(path)
            source_paths = sorted({page.source_path for page in project.pages if page.source_path})
            missing = [item for item in source_paths if not Path(item).exists()]
            if missing:
                raise FileNotFoundError("Не найдены исходные файлы:\n" + "\n".join(missing))
            self._close_loaders()
            for item in source_paths:
                self.loaders[item] = DocumentLoader(item)
            self.project = project
            self.project_path = Path(path)
            self.aligned_cache.clear()
            self.recognition_baselines = {
                i: page.data.model_dump() for i, page in enumerate(self.project.pages)
            }
            self.show_page(0)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось открыть проект", str(exc))

    def show_page(self, index: int) -> None:
        if not self.project.pages or not 0 <= index < len(self.project.pages):
            return
        self._commit_current_page()
        self.current_index = index
        project_page = self.project.pages[index]
        if index not in self.aligned_cache:
            loader = self.loaders[project_page.source_path]
            raw = loader.render_page(project_page.source_page)
            aligned = self.aligner.align(raw)
            self.aligned_cache[index] = aligned.image
            project_page.data.alignment_quality = aligned.quality
        self.image_view.set_image(self.aligned_cache[index])
        self._populate_form(project_page.data)
        self.page_label.setText(
            f"{index + 1} из {len(self.project.pages)} · {project_page.source_name} · "
            f"выравнивание {project_page.data.alignment_quality:.0%}"
        )
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.project.pages) - 1)
        self._update_overlays()

    def _populate_form(self, data: RecognizedPage) -> None:
        for name, edit in self.header_edits.items():
            edit.setText(getattr(data.header, name))
            edit.setStyleSheet("")
        rows = data.normalized_rows()
        for row_index, row in enumerate(rows):
            for column, field in enumerate(ROW_FIELDS):
                item = self.table.item(row_index, column)
                item.setText(getattr(row, field))
                item.setBackground(QColor("white") if row_index % 2 == 0 else QColor("#f7fafc"))
        for name, edit in self.footer_edits.items():
            edit.setText(getattr(data.footer, name))
            edit.setStyleSheet("")
        self.sketch_notes.setPlainText(data.footer.sketch_notes)
        self.page_notes.setPlainText(data.page_notes)

    def _data_from_form(self) -> RecognizedPage:
        current = self.project.pages[self.current_index].data if self.current_index >= 0 else RecognizedPage()
        header = HeaderData(**{name: edit.text().strip() for name, edit in self.header_edits.items()})
        rows = []
        for row_index in range(19):
            values = {
                field: self.table.item(row_index, column).text().strip()
                for column, field in enumerate(ROW_FIELDS)
            }
            rows.append(JournalRow(**values))
        footer_values = {name: edit.text().strip() for name, edit in self.footer_edits.items()}
        footer_values["sketch_notes"] = self.sketch_notes.toPlainText().strip()
        return RecognizedPage(
            header=header,
            rows=rows,
            footer=FooterData(**footer_values),
            page_notes=self.page_notes.toPlainText().strip(),
            recognition_mode=current.recognition_mode,
            alignment_quality=current.alignment_quality,
        )

    def _commit_current_page(self) -> None:
        if 0 <= self.current_index < len(self.project.pages):
            self.project.pages[self.current_index].data = self._data_from_form()

    def recognize_local(self) -> None:
        self._start_recognition(
            "Локальное распознавание в изолированном процессе…",
            lambda progress: run_local_ocr_process(self._current_image(), progress_callback=progress),
            with_progress=True,
        )

    def recognize_cloud(self) -> None:
        key = get_api_key()
        if not key:
            QMessageBox.information(
                self,
                "Нужен API-ключ",
                "Откройте «Настройки» и сохраните Gemini API-ключ.",
            )
            return
        model = str(QSettings().value("gemini/model", DEFAULT_GEMINI_MODEL))
        self._start_recognition(
            "Gemini OCR: подготовка изображения…",
            lambda progress: GeminiCloudOCR(key, model).recognize(
                self._current_image(), progress_callback=progress
            ),
            with_progress=True,
            show_progress_dialog=True,
        )

    def _start_recognition(
        self,
        message: str,
        fn,
        with_progress: bool = False,
        show_progress_dialog: bool = False,
    ) -> None:
        if self.current_index < 0:
            QMessageBox.information(self, "Нет страницы", "Сначала откройте PDF или изображение.")
            return
        index = self.current_index
        self._set_busy(True, message)
        if show_progress_dialog:
            self.ocr_progress_dialog = OCRProgressDialog(
                self, timeout_seconds=GEMINI_TIMEOUT_SECONDS
            )
            self.ocr_progress_dialog.start(message)
        worker = FunctionWorker(fn, with_progress=with_progress)
        self.active_workers.append(worker)

        def accept_result(result: RecognizedPage) -> None:
            quality = self.project.pages[index].data.alignment_quality
            result.alignment_quality = quality
            result.rows = result.normalized_rows()
            self.project.pages[index].data = result
            self.recognition_baselines[index] = result.model_dump()
            if self.current_index == index:
                self._populate_form(result)
                self.run_validation(show_dialog=False)

        worker.signals.result.connect(accept_result)
        def show_progress(text: str) -> None:
            self.statusBar().showMessage(text)
            if self.ocr_progress_dialog is not None:
                self.ocr_progress_dialog.set_stage(text)

        def show_error(text: str) -> None:
            self._finish_progress_dialog()
            QMessageBox.critical(self, "Ошибка OCR", text)

        def finish_recognition() -> None:
            self._finish_progress_dialog()
            self._set_busy(False, "Распознавание завершено")
            if worker in self.active_workers:
                self.active_workers.remove(worker)

        worker.signals.progress.connect(show_progress)
        worker.signals.error.connect(show_error)
        worker.signals.finished.connect(finish_recognition)
        self.thread_pool.start(worker)

    def _finish_progress_dialog(self) -> None:
        if self.ocr_progress_dialog is not None:
            self.ocr_progress_dialog.finish()
            self.ocr_progress_dialog = None

    def _current_image(self) -> np.ndarray:
        return self.aligned_cache[self.current_index].copy()

    def run_validation(self, show_dialog: bool = True) -> list[ValidationIssue]:
        if self.current_index < 0:
            return []
        data = self._data_from_form()
        self.project.pages[self.current_index].data = data
        issues = validate_page(data)
        self._paint_issues(issues)
        if show_dialog:
            if not issues:
                QMessageBox.information(self, "Проверка", "Критичных несоответствий не обнаружено.")
            else:
                lines = [f"• {issue.message}" for issue in issues[:18]]
                if len(issues) > 18:
                    lines.append(f"• …и ещё {len(issues) - 18}")
                QMessageBox.warning(self, "Результат проверки", "\n".join(lines))
        return issues

    def _paint_issues(self, issues: list[ValidationIssue]) -> None:
        self._populate_form(self._data_from_form())
        for issue in issues:
            color = "#ffd7d7" if issue.severity == "error" else "#fff1bf"
            if issue.field.startswith("header."):
                name = issue.field.split(".", 1)[1]
                if name in self.header_edits:
                    self.header_edits[name].setStyleSheet(f"background: {color};")
            elif issue.field.startswith("rows["):
                try:
                    row_index = int(issue.field.split("[", 1)[1].split("]", 1)[0])
                    field = issue.field.rsplit(".", 1)[1]
                    column = ROW_FIELDS.index(field)
                    self.table.item(row_index, column).setBackground(QColor(color))
                except (ValueError, IndexError):
                    pass

    def save_current_project(self) -> None:
        if not self.project.pages:
            return
        self._commit_current_page()
        path = self.project_path
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "буровой_журнал.bjproj", "Проект (*.bjproj)")
            if not selected:
                return
            path = Path(selected)
            if path.suffix.lower() != ".bjproj":
                path = path.with_suffix(".bjproj")
        self._record_corrections(path.parent / "corrections.jsonl")
        save_project(self.project, path)
        self.project_path = path
        self.statusBar().showMessage(f"Проект сохранён: {path.name}", 5000)

    def _record_corrections(self, path: Path) -> None:
        for index, baseline in list(self.recognition_baselines.items()):
            after = self.project.pages[index].data.model_dump()
            append_corrections(baseline, after, self.project.pages[index].source_name, index, path)
            self.recognition_baselines[index] = after

    def export_excel(self) -> None:
        if not self.project.pages:
            return
        self._commit_current_page()
        selected, _ = QFileDialog.getSaveFileName(self, "Экспорт в Excel", "буровые_журналы.xlsx", "Excel (*.xlsx)")
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        try:
            export_project(self.project, path)
            QMessageBox.information(self, "Экспорт завершён", f"Создан файл:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def open_settings(self) -> None:
        SettingsDialog(self).exec()

    def _update_overlays(self) -> None:
        if self.current_index < 0 or self.current_index not in self.aligned_cache:
            return
        height, width = self.aligned_cache[self.current_index].shape[:2]
        overlays = [
            (nr(14, 96, 1273, 689).pixels(width, height), "blue", 2),
            (nr(14, 690, 1273, 868).pixels(width, height), "blue", 2),
        ]
        selected = self.table.currentRow()
        if 0 <= selected < 19:
            rect = nr(TABLE_X[0], TABLE_Y[selected], TABLE_X[-1], TABLE_Y[selected + 1])
            overlays.append((rect.pixels(width, height), "orange", 4))
        self.image_view.set_overlays(overlays)

    def _set_busy(self, busy: bool, message: str) -> None:
        for action in self.busy_actions:
            action.setEnabled(not busy)
        self.prev_button.setEnabled(not busy and self.current_index > 0)
        self.next_button.setEnabled(not busy and self.current_index < len(self.project.pages) - 1)
        self.statusBar().showMessage(message)

    def _close_loaders(self) -> None:
        for loader in self.loaders.values():
            loader.close()
        self.loaders.clear()

    def closeEvent(self, event):  # noqa: N802
        self._close_loaders()
        event.accept()


def run_app() -> int:
    QApplication.setOrganizationName("GeoJournal")
    QApplication.setApplicationName("BoreholeJournalOCR")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())
    window = MainWindow()
    window.show()
    return app.exec()
