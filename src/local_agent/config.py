from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from LOCAL_AGENT_* variables."""

    model_config = SettingsConfigDict(env_prefix="LOCAL_AGENT_", env_file=".env", extra="ignore")
    model: str = "Qwen/Qwen3-14B-AWQ"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "local"
    temperature: float = 0.2
    max_tokens: int = 2048
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    knowledge_dir: Path = Path("data/knowledge")
    index_dir: Path = Path("data/indexes")
    checkpoint_db: Path = Path("data/checkpoints/langgraph.sqlite")
    checkpoint_enabled: bool = True
    thread_id: str = "default"


def load_agent_yaml(path: Path = Path("config/agent.yaml")) -> dict[str, Any]:
    """Load the project's small YAML config without adding a runtime dependency."""
    data: dict[str, Any] = {}
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            data[key] = {}
            section = key
            continue
        if section is None or ":" not in line:
            continue

        key, raw_value = line.strip().split(":", 1)
        data[section][key] = _parse_scalar(raw_value.strip())

    return data


def _parse_scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
