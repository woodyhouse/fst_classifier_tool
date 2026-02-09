"""JSON Schema 校验工具."""
from __future__ import annotations

import json
from pathlib import Path
from jsonschema import Draft7Validator


def load_schema(schema_path: str | Path) -> dict:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def validate_instance(instance: dict, schema: dict) -> None:
    v = Draft7Validator(schema)
    errors = sorted(v.iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        msgs = []
        for e in errors[:20]:
            loc = "/".join([str(x) for x in e.path]) or "<root>"
            msgs.append(f"  {loc}: {e.message}")
        raise ValueError("Schema validation failed:\n" + "\n".join(msgs))
