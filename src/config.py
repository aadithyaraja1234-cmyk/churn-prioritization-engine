from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    return _load_simple_yaml(config_path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith(" "):
            section_name = line[:-1].strip()
            current_section = {}
            config[section_name] = current_section
            continue
        if ":" not in line:
            continue
        if current_section is None:
            key, value = [part.strip() for part in line.split(":", 1)]
            config[key] = _parse_scalar(value)
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        current_section[key] = _parse_scalar(value)
    return config


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("\"") and value.endswith("\""):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value
