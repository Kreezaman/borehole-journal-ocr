from __future__ import annotations

import re
import time
from collections.abc import Callable

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.core.layout import FOOTER_RECTS, HEADER_RECTS, TABLE_FIELDS, table_cell_rect
from app.core.validation import validate_page
from app.models import FooterData, HeaderData, JournalRow, OCRWarning, RecognizedPage
from app.ocr.base import OCRProvider


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_TIMEOUT_SECONDS = 120
GEMINI_RETRY_DELAYS = (15, 30)
GEMINI_MAX_ATTEMPTS = len(GEMINI_RETRY_DELAYS) + 1
GEMINI_PROGRESS_TIMEOUT_SECONDS = (
    GEMINI_TIMEOUT_SECONDS * GEMINI_MAX_ATTEMPTS + sum(GEMINI_RETRY_DELAYS)
)
MAX_REVIEW_FIELDS = 6

OCR_PROMPT = """Ты выполняешь точный OCR русского рукописного бурового журнала инженерно-геологических изысканий.
Одни и те же данные показаны на нескольких изображениях: сначала весь лист для понимания структуры, затем увеличенные области. Сопоставь их и верни один результат строго по заданной JSON-схеме.

Правила:
- Возвращай только сведения, которые действительно видны на странице.
- Не исправляй и не додумывай сомнительные буквы и числа. Для полностью неразборчивого значения ставь "?", для незаполненного поля возвращай пустую строку.
- Сохраняй русские геологические термины, сокращения, десятичные запятые, интервалы глубин и обозначения в исходном виде.
- В rows возвращай логические интервалы грунта сверху вниз, не более 19. Если описание одного интервала занимает несколько печатных строк, объедини его в одну запись.
- Поля depth_from и depth_to бери из колонок глубины слоя «от» и «до».
- Поле frozen_interval содержит значение из колонки мощности/мерзлоты, как написано на бланке.
- Номера, глубины образцов и сведения о воде привязывай к соответствующему интервалу.
- Не превращай печатные заголовки бланка в распознанные значения.
- В sketch_notes перечисляй только читаемые текстовые подписи в поле «Абрис», не описывай линии как точный чертёж.
- В warnings перечисляй только действительно сомнительные поля. field записывай как точный путь JSON, например rows[5].depth_to или header.coordinate_n. Не выдумывай проценты уверенности.
- Все пояснения пиши по-русски, кроме исходных обозначений и сокращений.
"""

REVIEW_PROMPT = """Повторно прочитай только перечисленные фрагменты бурового журнала.
Каждому изображению предшествует точный путь поля. Верни по одной записи для каждого пути.
Не используй контекст для угадывания значения: перепиши только видимые символы. Сохраняй десятичную запятую.
Если надёжно прочитать поле нельзя, поставь uncertain=true, оставь наиболее вероятное чтение в value и кратко объясни сомнение в reason.
"""


class GeminiHeader(BaseModel):
    borehole_no: str = Field(description="Рукописный номер скважины без печатной подписи поля.")
    drilling_start: str = Field(description="Дата начала бурения в исходной записи.")
    drilling_end: str = Field(description="Дата окончания бурения в исходной записи.")
    object_name: str = Field(description="Рукописное название объекта без слова «Объект».")
    coordinate_n: str = Field(description="Значение северной координаты, как написано на бланке.")
    coordinate_e: str = Field(description="Значение восточной координаты, как написано на бланке.")


class GeminiRow(BaseModel):
    layer_no: str = Field(description="Номер инженерно-геологического слоя из первой колонки.")
    depth_from: str = Field(description="Глубина слоя «от», например 2,3. Сохраняй десятичную запятую.")
    depth_to: str = Field(description="Глубина слоя «до», например 4,8. Сохраняй десятичную запятую.")
    frozen_interval: str = Field(description="Запись из колонки мощности или интервала мерзлоты.")
    description: str = Field(description="Полное рукописное описание грунта интервала без печатных заголовков.")
    sample_no: str = Field(description="Номер образца, привязанный к этому интервалу.")
    sample_depth: str = Field(description="Глубина отбора образца в исходной записи.")
    water: str = Field(description="Запись из колонки воды для этого интервала.")


class GeminiFooter(BaseModel):
    groundwater_exposed: str = Field(description="Глубина вскрытого уровня подземных вод.")
    groundwater_stabilized: str = Field(description="Глубина установившегося уровня подземных вод.")
    specific_odor: str = Field(description="Рукописная запись о специфическом запахе.")
    drilling_rig: str = Field(description="Название или обозначение буровой установки.")
    drilling_diameter: str = Field(description="Диаметр бурения в исходной записи.")
    drilling_depth: str = Field(description="Итоговая глубина бурения в исходной записи.")
    drill_master: str = Field(description="Фамилия или подпись бурового мастера, только читаемый текст.")
    engineer_geologist: str = Field(description="Фамилия или подпись инженера-геолога, только читаемый текст.")
    sketch_notes: str = Field(description="Только читаемые текстовые подписи из поля «Абрис».")


class GeminiWarning(BaseModel):
    field: str = Field(description="Точный путь сомнительного поля в JSON.")
    reason: str = Field(description="Какие символы или варианты чтения вызывают сомнение.")
    alternative: str = Field(description="Второй возможный вариант или пустая строка.")


class GeminiExtraction(BaseModel):
    header: GeminiHeader
    rows: list[GeminiRow] = Field(min_length=0, max_length=19)
    footer: GeminiFooter
    page_notes: str = Field(description="Только общие замечания о читаемости страницы или пустая строка.")
    warnings: list[GeminiWarning] = Field(min_length=0, max_length=20)


class GeminiFieldReview(BaseModel):
    field: str = Field(description="Точный путь поля, указанный перед изображением.")
    value: str = Field(description="Повторно прочитанное значение в исходной записи.")
    uncertain: bool = Field(description="True, если хотя бы один важный символ нельзя прочитать надёжно.")
    reason: str = Field(description="Краткая причина сомнения или пустая строка.")


class GeminiReviewExtraction(BaseModel):
    results: list[GeminiFieldReview] = Field(min_length=0, max_length=MAX_REVIEW_FIELDS)


class GeminiCloudOCR(OCRProvider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        client=None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not api_key.strip():
            raise ValueError("API-ключ Gemini не задан")
        self.model = model.strip() or DEFAULT_GEMINI_MODEL
        self._sleep = sleep_fn
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Не установлен модуль google-genai. Повторно запустите install.bat."
                ) from exc

            # Retries are controlled here so the user sees the wait and countdown.
            retry_options = types.HttpRetryOptions(attempts=1)
            http_options = types.HttpOptions(
                timeout=GEMINI_TIMEOUT_SECONDS * 1000,
                retry_options=retry_options,
            )
            client = genai.Client(
                api_key=api_key.strip(),
                http_options=http_options,
            )
        self.client = client

    def recognize(
        self,
        aligned_rgb: np.ndarray,
        progress_callback=None,
    ) -> RecognizedPage:
        _progress(progress_callback, "Этап 1 из 5: подготовка полного листа и увеличенных областей…")
        primary_input = _primary_input(aligned_rgb)
        _progress(progress_callback, "Этап 2 из 5: Gemini распознаёт страницу…")
        response = self._create(
            input_parts=primary_input,
            schema=GeminiExtraction.model_json_schema(),
            progress_callback=progress_callback,
            max_attempts=GEMINI_MAX_ATTEMPTS,
        )

        _progress(progress_callback, "Этап 3 из 5: проверка структурированного ответа…")
        parsed = _parse_response(response, GeminiExtraction)
        page = RecognizedPage(
            header=HeaderData(**parsed.header.model_dump()),
            rows=[JournalRow(**row.model_dump()) for row in parsed.rows],
            footer=FooterData(**parsed.footer.model_dump()),
            page_notes=parsed.page_notes,
            ocr_warnings=[
                OCRWarning(**warning.model_dump())
                for warning in parsed.warnings
                if _field_rect(warning.field) is not None
            ],
            recognition_mode="gemini",
        )
        page.rows = page.normalized_rows()

        review_fields = _review_fields(page)
        if review_fields:
            _progress(
                progress_callback,
                f"Этап 4 из 5: повторная проверка сомнительных полей ({len(review_fields)})…",
            )
            try:
                self._review_fields(aligned_rgb, page, review_fields, progress_callback)
            except Exception as exc:
                # A selective recheck is an optional quality pass. Preserve the
                # complete primary OCR result when the free quota cannot afford it.
                _progress(
                    progress_callback,
                    f"Повторная проверка пропущена: {_friendly_error(exc)}",
                )
        else:
            _progress(progress_callback, "Этап 4 из 5: сомнительных полей не найдено.")

        _progress(progress_callback, "Этап 5 из 5: результат готов, проверьте жёлтые поля.")
        return page

    def _review_fields(
        self,
        aligned_rgb: np.ndarray,
        page: RecognizedPage,
        fields: list[str],
        progress_callback,
    ) -> None:
        input_parts: list[dict] = [{"type": "text", "text": REVIEW_PROMPT}]
        for index, field in enumerate(fields, start=1):
            rect = _field_rect(field)
            if rect is None:
                continue
            crop = _crop_rect(aligned_rgb, rect, padding=12)
            crop = _enlarge_small_crop(crop)
            input_parts.extend(
                [
                    {"type": "text", "text": f"Фрагмент {index}: поле {field}"},
                    _image_input(crop, resolution="high"),
                ]
            )

        response = self._create(
            input_parts=input_parts,
            schema=GeminiReviewExtraction.model_json_schema(),
            progress_callback=progress_callback,
            max_attempts=2,
        )
        parsed = _parse_response(response, GeminiReviewExtraction)
        allowed = set(fields)
        remaining_warnings = {warning.field: warning for warning in page.ocr_warnings}
        for result in parsed.results:
            if result.field not in allowed or _field_rect(result.field) is None:
                continue
            if result.uncertain:
                remaining_warnings[result.field] = OCRWarning(
                    field=result.field,
                    reason=result.reason or "значение осталось неразборчивым после повторной проверки",
                    alternative=result.value,
                )
                continue
            _set_field_value(page, result.field, result.value)
            remaining_warnings.pop(result.field, None)
        page.ocr_warnings = list(remaining_warnings.values())

    def _create(
        self,
        *,
        input_parts: list[dict],
        schema: dict,
        progress_callback,
        max_attempts: int,
    ):
        contents = _generate_content_parts(input_parts)
        for attempt in range(1, max_attempts + 1):
            try:
                # Gemini 3.1 Flash-Lite documents multimodal structured output
                # through generateContent.  Interactions accepted the SDK
                # objects locally but rejected inline image data with a generic
                # 400 INVALID_ARGUMENT response.
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "max_output_tokens": 8192,
                        "thinking_config": {"thinking_level": "minimal"},
                        "response_mime_type": "application/json",
                        "response_json_schema": schema,
                    },
                )
            except Exception as exc:
                if _is_rate_limit(exc) and not _is_daily_quota(exc) and attempt < max_attempts:
                    fallback = GEMINI_RETRY_DELAYS[min(attempt - 1, len(GEMINI_RETRY_DELAYS) - 1)]
                    delay = _retry_after(exc, fallback)
                    for seconds in range(delay, 0, -1):
                        _progress(
                            progress_callback,
                            f"⏳ Временный лимит Gemini. Повтор через {seconds} сек "
                            f"(попытка {attempt + 1} из {max_attempts})…",
                        )
                        self._sleep(1)
                    continue
                raise _mapped_error(exc) from exc
        raise RuntimeError("Gemini не выполнил запрос после повторных попыток.")


def _primary_input(image: np.ndarray) -> list[dict]:
    height, _width = image.shape[:2]
    regions = [
        ("Изображение 1 — полный лист для понимания структуры.", image, "medium"),
        (
            "Изображение 2 — увеличенная шапка журнала.",
            image[0 : round(height * 0.115), :],
            "high",
        ),
        (
            "Изображение 3 — увеличенная основная таблица, печатные строки 1–10.",
            image[round(height * 0.205) : round(height * 0.515), :],
            "high",
        ),
        (
            "Изображение 4 — увеличенная основная таблица, печатные строки 10–19.",
            image[round(height * 0.495) : round(height * 0.780), :],
            "high",
        ),
        (
            "Изображение 5 — увеличенная нижняя часть журнала и поле «Абрис».",
            image[round(height * 0.760) : height, :],
            "high",
        ),
    ]
    parts: list[dict] = [{"type": "text", "text": OCR_PROMPT}]
    for label, crop, resolution in regions:
        parts.append({"type": "text", "text": label})
        parts.append(_image_input(crop, resolution=resolution))
    return parts


def _image_input(image: np.ndarray, *, resolution: str) -> dict:
    return {
        "type": "image",
        "data": _encode_jpeg(image),
        "mime_type": "image/jpeg",
        "resolution": resolution,
    }


def _generate_content_parts(input_parts: list[dict]) -> list:
    """Convert internal OCR parts to supported generateContent SDK parts."""
    try:
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен модуль google-genai. Повторно запустите install.bat."
        ) from exc

    resolution_levels = {
        "medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
        "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    }
    contents = []
    for part in input_parts:
        if part["type"] == "text":
            contents.append(part["text"])
            continue
        contents.append(
            types.Part.from_bytes(
                data=part["data"],
                mime_type=part["mime_type"],
                media_resolution=resolution_levels[part["resolution"]],
            )
        )
    return contents


def _parse_response(response, model_type):
    output_text = (
        getattr(response, "text", "")
        or getattr(response, "output_text", "")
        or ""
    )
    if not output_text.strip():
        raise RuntimeError("Gemini не вернул результат распознавания.")
    try:
        return model_type.model_validate_json(output_text)
    except ValidationError as exc:
        raise RuntimeError("Gemini вернул результат неправильного формата.") from exc


def _review_fields(page: RecognizedPage) -> list[str]:
    fields: list[str] = []

    def add(field: str) -> None:
        if field not in fields and _field_rect(field) is not None:
            fields.append(field)

    for warning in page.ocr_warnings:
        add(warning.field)
    issues = validate_page(page)
    for issue in sorted(issues, key=lambda item: item.severity != "error"):
        add(issue.field)
    return fields[:MAX_REVIEW_FIELDS]


_ROW_FIELD = re.compile(r"^rows\[(\d+)]\.([a-z_]+)$")


def _field_rect(field: str):
    if field.startswith("header."):
        name = field.split(".", 1)[1]
        if name in {"drilling_start", "drilling_end"}:
            name = "drilling_dates"
        return HEADER_RECTS.get(name)
    if field.startswith("footer."):
        return FOOTER_RECTS.get(field.split(".", 1)[1])
    match = _ROW_FIELD.match(field)
    if not match:
        return None
    row_index = int(match.group(1))
    field_name = match.group(2)
    if not 0 <= row_index < 19 or field_name not in TABLE_FIELDS:
        return None
    return table_cell_rect(row_index, TABLE_FIELDS.index(field_name))


def _crop_rect(image: np.ndarray, rect, padding: int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = rect.pixels(width, height)
    return image[
        max(0, y1 - padding) : min(height, y2 + padding),
        max(0, x1 - padding) : min(width, x2 + padding),
    ]


def _enlarge_small_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if not height or not width:
        return image
    scale = min(4.0, 2048.0 / width, max(1.0, 220.0 / height))
    if scale <= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _set_field_value(page: RecognizedPage, field: str, value: str) -> None:
    if field.startswith("header."):
        name = field.split(".", 1)[1]
        if name in HeaderData.model_fields:
            setattr(page.header, name, value)
        return
    if field.startswith("footer."):
        name = field.split(".", 1)[1]
        if name in FooterData.model_fields:
            setattr(page.footer, name, value)
        return
    match = _ROW_FIELD.match(field)
    if match:
        row_index = int(match.group(1))
        name = match.group(2)
        if 0 <= row_index < len(page.rows) and name in JournalRow.model_fields:
            setattr(page.rows[row_index], name, value)


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def _is_daily_quota(exc: Exception) -> bool:
    text = f"{getattr(exc, 'details', '')} {exc}".lower()
    markers = ("per day", "requests per day", "daily", "rpd", "per_day", "perday")
    return any(marker in text for marker in markers)


def _retry_after(exc: Exception, fallback: int) -> int:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        if headers.get("retry-after-ms"):
            return max(1, min(60, round(float(headers["retry-after-ms"]) / 1000)))
        if headers.get("retry-after"):
            return max(1, min(60, round(float(headers["retry-after"]))))
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:retry(?:Delay|[- ]after)?[^0-9]{0,12})(\d+)(?:s| sec)?", str(exc), re.I)
    if match:
        return max(1, min(60, int(match.group(1))))
    return fallback


def _mapped_error(exc: Exception) -> Exception:
    message = str(exc).strip()
    if "timeout" in message.lower() or "timed out" in message.lower():
        return TimeoutError(
            f"Gemini не ответил за {GEMINI_TIMEOUT_SECONDS} секунд. "
            "Запрос остановлен; повторите попытку позже."
        )
    if _is_rate_limit(exc):
        if _is_daily_quota(exc):
            return RuntimeError(
                "Исчерпана суточная бесплатная квота Gemini. Дождитесь её сброса в Google AI Studio."
            )
        return RuntimeError(
            "Gemini временно ограничил частоту запросов. Автоматические повторы закончились."
        )
    if "401" in message or "403" in message or "API_KEY" in message.upper():
        return RuntimeError("Gemini отклонил API-ключ. Проверьте ключ в настройках программы.")
    if getattr(exc, "code", None) == 400 or "INVALID_ARGUMENT" in message.upper():
        return RuntimeError(
            "Gemini отклонил параметры запроса (400 INVALID_ARGUMENT). "
            "Обновите программу до последней версии; если ошибка повторится, "
            "проверьте выбранную модель в настройках."
        )
    return RuntimeError(f"Ошибка Gemini OCR: {message or type(exc).__name__}")


def _friendly_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text.startswith(("RuntimeError:", "TimeoutError:")):
        text = text.split(":", 1)[1].strip()
    return text or type(exc).__name__


def _encode_jpeg(image: np.ndarray, limit_bytes: int = 18 * 1024 * 1024) -> bytes:
    """Encode one RGB image below Gemini's inline-image limit."""
    current = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    quality = 94
    for _ in range(5):
        success, encoded = cv2.imencode(
            ".jpg",
            current,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not success:
            break
        payload = encoded.tobytes()
        if len(payload) <= limit_bytes:
            return payload
        current = cv2.resize(
            current,
            None,
            fx=0.8,
            fy=0.8,
            interpolation=cv2.INTER_AREA,
        )
        quality = max(82, quality - 3)
    raise RuntimeError("Не удалось подготовить скан для отправки в Gemini.")


def _progress(callback, message: str) -> None:
    if callback is not None:
        callback(message)
