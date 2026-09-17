from __future__ import annotations

import json
from pathlib import Path

from app.models import JournalProject


def save_project(project: JournalProject, path: str | Path) -> None:
    project.touch()
    target = Path(path)
    target.write_text(project.model_dump_json(indent=2), encoding="utf-8")


def load_project(path: str | Path) -> JournalProject:
    return JournalProject.model_validate_json(Path(path).read_text(encoding="utf-8"))


def append_corrections(
    before: dict,
    after: dict,
    source_name: str,
    page_index: int,
    output_path: str | Path,
) -> int:
    """Append only changed scalar values as future training examples."""
    changes: list[dict] = []

    def walk(old: object, new: object, field_path: str) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                if not field_path and key == "ocr_warnings":
                    continue
                walk(old.get(key), new.get(key), f"{field_path}.{key}" if field_path else key)
        elif isinstance(old, list) and isinstance(new, list):
            for index in range(max(len(old), len(new))):
                old_item = old[index] if index < len(old) else None
                new_item = new[index] if index < len(new) else None
                walk(old_item, new_item, f"{field_path}[{index}]")
        elif old != new:
            changes.append(
                {
                    "source": source_name,
                    "page": page_index + 1,
                    "field": field_path,
                    "recognized": old,
                    "corrected": new,
                }
            )

    walk(before, after, "")
    if changes:
        with Path(output_path).open("a", encoding="utf-8") as stream:
            for item in changes:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(changes)
