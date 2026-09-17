from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import RecognizedPage


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: str = "warning"


_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def parse_number(value: str) -> float | None:
    text = value.strip().replace(" ", "").replace(",", ".")
    if not text or not _NUMBER.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate_page(page: RecognizedPage) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not page.header.borehole_no.strip():
        issues.append(ValidationIssue("header.borehole_no", "Не указан номер скважины"))
    previous_to: float | None = None
    has_data = False
    for index, row in enumerate(page.normalized_rows()):
        values = [getattr(row, field) for field in row.model_fields]
        if not any(value.strip() for value in values):
            continue
        has_data = True
        start = parse_number(row.depth_from)
        end = parse_number(row.depth_to)
        prefix = f"rows[{index}]"
        if row.depth_from and start is None:
            issues.append(ValidationIssue(f"{prefix}.depth_from", "Глубина «от» не является числом"))
        if row.depth_to and end is None:
            issues.append(ValidationIssue(f"{prefix}.depth_to", "Глубина «до» не является числом"))
        if start is not None and end is not None and start > end:
            issues.append(ValidationIssue(f"{prefix}.depth_to", "Конечная глубина меньше начальной", "error"))
        if previous_to is not None and start is not None and abs(previous_to - start) > 0.11:
            issues.append(ValidationIssue(f"{prefix}.depth_from", "Возможен разрыв между слоями"))
        if end is not None:
            previous_to = end
        if (start is not None or end is not None) and not row.description.strip():
            issues.append(ValidationIssue(f"{prefix}.description", "Для интервала отсутствует описание грунта"))
    if not has_data:
        issues.append(ValidationIssue("rows", "Основная таблица не заполнена", "error"))
    return issues
