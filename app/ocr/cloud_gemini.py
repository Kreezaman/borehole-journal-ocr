from __future__ import annotations

import base64

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.models import FooterData, HeaderData, JournalRow, RecognizedPage
from app.ocr.base import OCRProvider

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_TIMEOUT_SECONDS = 120

OCR_PROMPT = """Ты выполняешь OCR русского рукописного бурового журнала инженерно-геологических изысканий.
Распознай приложенную страницу и верни данные строго по заданной JSON-схеме.

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
- Все пояснения пиши по-русски, кроме исходных обозначений и сокращений.
"""


class GeminiExtraction(BaseModel):
    header: HeaderData
    rows: list[JournalRow] = Field(max_length=19)
    footer: FooterData
    page_notes: str


class GeminiCloudOCR(OCRProvider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        client=None,
    ):
        if not api_key.strip():
            raise ValueError("API-ключ Gemini не задан")
        self.model = model.strip() or DEFAULT_GEMINI_MODEL
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Не установлен модуль google-genai. Повторно запустите install.bat."
                ) from exc

            retry_options = types.HttpRetryOptions(attempts=0)
            http_options = types.HttpOptions(
                timeout=GEMINI_TIMEOUT_SECONDS * 1000,
                retry_options=retry_options,
            )
            client = genai.Client(
                api_key=api_key.strip(),
                http_options=http_options,
            )
            # google-genai normalizes 0 to one attempt for its REST client,
            # while the Interactions transport reads 0 as zero retries.
            # Restore the documented no-retry value after client setup.
            retry_options.attempts = 0
        self.client = client

    def recognize(
        self,
        aligned_rgb: np.ndarray,
        progress_callback=None,
    ) -> RecognizedPage:
        _progress(progress_callback, "Этап 1 из 3: подготовка изображения…")
        image64 = base64.b64encode(_encode_jpeg(aligned_rgb)).decode("ascii")
        _progress(progress_callback, "Этап 2 из 3: запрос отправлен, Gemini распознаёт страницу…")
        try:
            response = self.client.interactions.create(
                model=self.model,
                input=[
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image",
                        "data": image64,
                        "mime_type": "image/jpeg",
                    },
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": GeminiExtraction.model_json_schema(),
                },
            )
        except Exception as exc:
            message = str(exc).strip()
            if "timeout" in message.lower() or "timed out" in message.lower():
                raise TimeoutError(
                    f"Gemini не ответил за {GEMINI_TIMEOUT_SECONDS} секунд. "
                    "Запрос остановлен; повторите попытку позже."
                ) from exc
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                raise RuntimeError(
                    "Gemini отклонил запрос: исчерпана бесплатная квота или превышен лимит запросов."
                ) from exc
            if "401" in message or "403" in message or "API_KEY" in message.upper():
                raise RuntimeError(
                    "Gemini отклонил API-ключ. Проверьте ключ в настройках программы."
                ) from exc
            raise RuntimeError(f"Ошибка Gemini OCR: {message or type(exc).__name__}") from exc

        output_text = getattr(response, "output_text", "") or ""
        _progress(progress_callback, "Этап 3 из 3: ответ получен, проверка результата…")
        if not output_text.strip():
            raise RuntimeError("Gemini не вернул результат распознавания.")
        try:
            parsed = GeminiExtraction.model_validate_json(output_text)
        except ValidationError as exc:
            raise RuntimeError("Gemini вернул результат неправильного формата.") from exc

        page = RecognizedPage(
            header=parsed.header,
            rows=parsed.rows,
            footer=parsed.footer,
            page_notes=parsed.page_notes,
            recognition_mode="gemini",
        )
        page.rows = page.normalized_rows()
        return page


def _encode_jpeg(image: np.ndarray, limit_bytes: int = 18 * 1024 * 1024) -> bytes:
    """Encode an RGB page below Gemini's 20 MB inline-request limit."""
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