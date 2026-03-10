"""utils/config.py — YAML config loader with environment variable overrides."""
from __future__ import annotations
import copy, os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """Dot-notation config wrapper."""

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, Config(val))
            elif isinstance(val, list):
                setattr(self, key, [Config(v) if isinstance(v, dict) else v for v in val])
            else:
                setattr(self, key, val)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [i.to_dict() if isinstance(i, Config) else i for i in v]
            else:
                result[k] = v
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> Config:
    """
    Load YAML config and apply environment variable overrides:
      PPVP_DB_PASSWORD  → database.password
      PPVP_DB_HOST      → database.host
      PPVP_DB_USER      → database.user
      PPVP_DB_NAME      → database.name
      PPVP_HMAC_KEY     → crypto.hmac.secret_key
      PPVP_LOG_LEVEL    → logging.level
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Env var overrides (never hard-code secrets in YAML)
    env_map = {
        "PPVP_DB_PASSWORD": ["database", "password"],
        "PPVP_DB_HOST":     ["database", "host"],
        "PPVP_DB_USER":     ["database", "user"],
        "PPVP_DB_NAME":     ["database", "name"],
        "PPVP_HMAC_KEY":    ["crypto", "hmac", "secret_key"],
        "PPVP_LOG_LEVEL":   ["logging", "level"],
    }
    for env_var, key_path in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            node = raw
            for k in key_path[:-1]:
                node = node.setdefault(k, {})
            node[key_path[-1]] = val
            logger.debug("Config override from env: %s", env_var)

    return Config(raw)
