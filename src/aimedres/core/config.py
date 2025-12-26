"""
Enhanced Configuration Management for AiMedRes

Key Features:
- Layered configuration (defaults < file(s) < environment < overrides)
- Advanced validation (optional Pydantic if installed)
- Rich environment variable parsing with automatic mapping
- Hierarchical get/set with dot-notation
- Secret management abstraction (Env, Vault, AWS Secrets Manager) + caching
- Optional dynamic reloading (if watchdog installed)
- Provenance tracking (where each value came from)
- Schema + safe export (masking sensitive values)
- Pluggable sections and secret providers

NOTE:
This module is self-contained and keeps optional dependencies lazy-loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_type_hints

import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Dataclass Section Definitions
# ------------------------------------------------------------------------------


@dataclass
class SecurityConfig:
    auth_enabled: bool = True
    api_key_required: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 100
    session_timeout_minutes: int = 30
    encryption_key_length: int = 32
    password_min_length: int = 12
    enable_2fa: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "aimedres.db"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_connections: int = 20
    connection_timeout: int = 30


@dataclass
class NeuralNetworkConfig:
    input_size: int = 32
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    output_size: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.3
    activation_function: str = "relu"
    optimizer: str = "adam"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_enabled: bool = True
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    max_request_size: int = 16 * 1024 * 1024  # 16MB


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

SENSITIVE_KEY_PATTERN = re.compile(r"(pass(word)?|secret|token|key|credential)", re.IGNORECASE)


def deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def mask_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if SENSITIVE_KEY_PATTERN.search(key):
        return "***MASKED***"
    return value


def load_file_any(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        suffix = path.suffix.lower()
        if suffix in (".yml", ".yaml"):
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f) or {}
        raise ValueError(f"Unsupported config extension: {suffix}")


# ------------------------------------------------------------------------------
# Secret Providers
# ------------------------------------------------------------------------------


class SecretProvider:
    name: str = "base"

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError


class EnvSecretProvider(SecretProvider):
    name = "env"

    def get(self, key: str) -> Optional[str]:
        return os.getenv(key)


class VaultSecretProvider(SecretProvider):
    name = "vault"

    def __init__(self):
        self._loaded = False
        self._client = None

    def _ensure(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import hvac  # type: ignore

            vault_url = os.getenv("VAULT_URL")
            vault_token = os.getenv("VAULT_TOKEN")
            if not (vault_url and vault_token):
                return
            client = hvac.Client(url=vault_url, token=vault_token)
            if client.is_authenticated():
                self._client = client
            else:
                logger.warning("Vault authentication failed")
        except ImportError:
            logger.debug("hvac not installed; Vault provider inactive")
        except Exception as e:
            logger.warning(f"Vault provider init failed: {e}")

    def get(self, key: str) -> Optional[str]:
        self._ensure()
        if not self._client:
            return None
        try:
            vault_path = os.getenv("VAULT_SECRET_PATH", "aimedres")
            response = self._client.secrets.kv.v2.read_secret_version(path=vault_path)
            data = response["data"]["data"]
            return data.get(key)
        except Exception as e:
            logger.debug(f"Vault get error for {key}: {e}")
            return None


class AWSSecretProvider(SecretProvider):
    name = "aws"

    def __init__(self):
        self._loaded = False
        self._client = None
        self._cached: Dict[str, Any] = {}

    def _ensure(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import boto3  # type: ignore
            from botocore.exceptions import ClientError  # type: ignore

            secret_name = os.getenv("AWS_SECRET_NAME", "aimedres/secrets")
            region_name = os.getenv("AWS_REGION", "us-east-1")
            session = boto3.session.Session()
            client = session.client("secretsmanager", region_name=region_name)
            response = client.get_secret_value(SecretId=secret_name)
            payload = response.get("SecretString")
            if payload:
                self._cached = json.loads(payload)
        except ImportError:
            logger.debug("boto3 not installed; AWS provider inactive")
        except Exception as e:
            logger.debug(f"AWS secrets load failed: {e}")

    def get(self, key: str) -> Optional[str]:
        self._ensure()
        return self._cached.get(key)


class SecretManager:
    def __init__(self, providers: Optional[List[SecretProvider]] = None, cache_ttl: int = 300):
        self.providers = providers or [
            EnvSecretProvider(),
            VaultSecretProvider(),
            AWSSecretProvider(),
        ]
        self.cache: Dict[str, Tuple[float, Optional[str]]] = {}
        self.cache_ttl = cache_ttl

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        if key in self.cache:
            ts, val = self.cache[key]
            if now - ts < self.cache_ttl:
                return val
        for provider in self.providers:
            try:
                val = provider.get(key)
                if val is not None:
                    self.cache[key] = (now, val)
                    return val
            except Exception as e:
                logger.debug(f"Secret provider {provider.name} error: {e}")
        self.cache[key] = (now, None)
        return None


# ------------------------------------------------------------------------------
# Validation Framework (with optional Pydantic)
# ------------------------------------------------------------------------------


class ValidationErrorReport(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def validate_builtin(sections: Dict[str, Any]) -> None:
    errors: List[str] = []
    sec: SecurityConfig = sections["security"]
    if sec.max_requests_per_minute < 1:
        errors.append("security.max_requests_per_minute must be > 0")
    if sec.session_timeout_minutes < 1:
        errors.append("security.session_timeout_minutes must be > 0")
    if sec.password_min_length < 8:
        errors.append("security.password_min_length must be >= 8")

    db: DatabaseConfig = sections["database"]
    if not (1 <= db.port <= 65535):
        errors.append("database.port must be between 1 and 65535")

    api: APIConfig = sections["api"]
    if not (1 <= api.port <= 65535):
        errors.append("api.port must be between 1 and 65535")
    if api.ssl_enabled and not (api.ssl_cert_path and api.ssl_key_path):
        errors.append("api requires ssl_cert_path and ssl_key_path when ssl_enabled = True")

    nn: NeuralNetworkConfig = sections["neural_network"]
    if not (0 < nn.learning_rate <= 1):
        errors.append("neural_network.learning_rate must be between 0 and 1")
    if nn.batch_size < 1:
        errors.append("neural_network.batch_size must be > 0")
    if nn.epochs < 1:
        errors.append("neural_network.epochs must be > 0")

    if errors:
        raise ValidationErrorReport(errors)


def try_pydantic_validate(sections: Dict[str, Any]) -> Optional[Exception]:
    try:
        import pydantic  # type: ignore

        class SecurityModel(pydantic.BaseModel):
            auth_enabled: bool
            api_key_required: bool
            rate_limit_enabled: bool
            max_requests_per_minute: int
            session_timeout_minutes: int
            encryption_key_length: int
            password_min_length: int
            enable_2fa: bool
            allowed_origins: List[str]

        class DatabaseModel(pydantic.BaseModel):
            type: str
            host: str
            port: int
            name: str
            username: str
            password: str
            pool_size: int
            max_connections: int
            connection_timeout: int

        class NeuralNetworkModel(pydantic.BaseModel):
            input_size: int
            hidden_layers: List[int]
            output_size: int
            learning_rate: float
            batch_size: int
            epochs: int
            dropout_rate: float
            activation_function: str
            optimizer: str

        class APIModel(pydantic.BaseModel):
            host: str
            port: int
            debug: bool
            cors_enabled: bool
            ssl_enabled: bool
            ssl_cert_path: str
            ssl_key_path: str
            max_request_size: int

        # Validate each
        SecurityModel(**sections["security"].__dict__)
        DatabaseModel(**sections["database"].__dict__)
        NeuralNetworkModel(**sections["neural_network"].__dict__)
        APIModel(**sections["api"].__dict__)

    except ImportError:
        return None
    except Exception as e:
        return e
    return None


# ------------------------------------------------------------------------------
# Configuration Core
# ------------------------------------------------------------------------------


class DuetMindConfig:
    """
    Centralized configuration manager with layering, validation, secrets & reload.

    Precedence:
        Defaults < File(s) < Environment < Overrides

    Methods:
        load_files([...])
        load_environment()
        load_overrides({...})
        validate()
        get(path), set(path, value)
        explain()
        to_dict(), to_safe_dict()
        save_to_file(path, format='yaml')
        enable_auto_reload(paths, debounce=1.0)

    Provenance:
        Tracked in self._provenance[(section, key)] = source_label
    """

    ENV_PREFIX = "AIMEDRES_"

    def __init__(
        self,
        config_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        auto_env: bool = True,
        secret_manager: Optional[SecretManager] = None,
    ):
        # Sections
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.neural_network = NeuralNetworkConfig()
        self.api = APIConfig()

        self._section_names = ["security", "database", "neural_network", "api"]
        self._provenance: Dict[Tuple[str, str], str] = {}
        self._overrides: Dict[str, Any] = overrides or {}
        self._secret_manager = secret_manager or SecretManager()
        self._watch_threads: List[threading.Thread] = []
        self._watch_stop_event = threading.Event()
        self._reload_queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()

        if config_files:
            self.load_files(config_files)

        if auto_env:
            self.load_environment()

        if overrides:
            self.load_overrides(overrides)

        self.validate()

    # ------------------------------ Loading ---------------------------------

    def load_files(self, files: Union[str, Path, List[Union[str, Path]]]) -> None:
        if isinstance(files, (str, Path)):
            file_list = [files]
        else:
            file_list = files
        for f in file_list:
            path = Path(f).expanduser().resolve()
            if not path.exists():
                logger.warning(f"Config file does not exist: {path}")
                continue
            try:
                data = load_file_any(path)
                if not isinstance(data, dict):
                    logger.warning(f"Config file {path} did not yield a dict root.")
                    continue
                self._apply_dict(data, provenance=f"file:{path.name}")
                logger.info(f"Loaded config file: {path}")
            except Exception as e:
                logger.error(f"Error loading config file {path}: {e}")

    def load_environment(self) -> None:
        """
        Auto-maps environment variables using pattern:
        PREFIX + SECTION + '_' + FIELD
        e.g. AIMEDRES_SECURITY_MAX_REQUESTS_PER_MINUTE=250
        Supports JSON for lists/dicts (attempt parse).
        """
        for section_name in self._section_names:
            section_obj = getattr(self, section_name)
            for f in fields(section_obj):
                env_name = f"{self.ENV_PREFIX}{section_name.upper()}_{f.name.upper()}"
                val = os.getenv(env_name)
                if val is None:
                    continue
                try:
                    parsed = self._coerce_env_value(val, f.type)
                    setattr(section_obj, f.name, parsed)
                    self._provenance[(section_name, f.name)] = f"env:{env_name}"
                    logger.debug(f"Loaded env {env_name} -> {section_name}.{f.name}={parsed}")
                except Exception as e:
                    logger.error(f"Failed parsing env {env_name}={val}: {e}")

    def load_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply programmatic overrides (highest precedence).
        Accepts nested dict with keys matching section names.
        """
        self._apply_dict(overrides, provenance="override")
        deep_merge(self._overrides, overrides)

    # ------------------------------ Internal Helpers ---------------------------------

    def _apply_dict(self, data: Dict[str, Any], provenance: str) -> None:
        for section_name, section_data in data.items():
            if section_name not in self._section_names:
                continue
            if not isinstance(section_data, dict):
                logger.warning(f"Ignoring non-dict section data: {section_name}")
                continue
            section_obj = getattr(self, section_name)
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    self._provenance[(section_name, key)] = provenance
                    logger.debug(f"Set {section_name}.{key}={value} ({provenance})")

    def _coerce_env_value(self, raw: str, target_type: Any) -> Any:
        raw_stripped = raw.strip()
        # Try JSON for complex types
        if raw_stripped.startswith("[") or raw_stripped.startswith("{"):
            try:
                return json.loads(raw_stripped)
            except Exception:
                pass
        # Basic bool heuristics
        if target_type == bool:
            return raw_stripped.lower() in ("1", "true", "yes", "on", "y")
        # Int / Float
        if target_type == int:
            return int(raw_stripped)
        if target_type == float:
            return float(raw_stripped)
        # Fallback
        return raw

    # ------------------------------ Validation ---------------------------------

    def validate(self) -> None:
        sections = {name: getattr(self, name) for name in self._section_names}
        # Try optional Pydantic first
        pydantic_err = try_pydantic_validate(sections)
        if pydantic_err:
            raise ValidationErrorReport([f"Pydantic validation failed: {pydantic_err}"])
        # Built-in validation
        validate_builtin(sections)
        logger.info("Configuration validation passed")

    # ------------------------------ Access & Introspection ---------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        """
        Retrieve value using dot notation: e.g., "security.max_requests_per_minute".
        """
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError("Path must be in 'section.key' format")
        section, key = parts
        if section not in self._section_names:
            raise KeyError(f"Unknown section: {section}")
        section_obj = getattr(self, section)
        return getattr(section_obj, key, default)

    def set(self, path: str, value: Any, provenance: str = "set") -> None:
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError("Path must be in 'section.key' format")
        section, key = parts
        if section not in self._section_names:
            raise KeyError(f"Unknown section: {section}")
        section_obj = getattr(self, section)
        if not hasattr(section_obj, key):
            raise AttributeError(f"No such key {key} in section {section}")
        setattr(section_obj, key, value)
        self._provenance[(section, key)] = provenance

    def explain(self) -> List[Dict[str, str]]:
        """
        Returns provenance info for each field.
        """
        report = []
        for section_name in self._section_names:
            section_obj = getattr(self, section_name)
            for f in fields(section_obj):
                prov = self._provenance.get((section_name, f.name), "default")
                report.append(
                    {
                        "path": f"{section_name}.{f.name}",
                        "value": repr(getattr(section_obj, f.name)),
                        "provenance": prov,
                    }
                )
        return report

    # ------------------------------ Export ---------------------------------

    def to_dict(self, mask_sensitive: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for section_name in self._section_names:
            section_obj = getattr(self, section_name)
            section_dict = {}
            for f in fields(section_obj):
                val = getattr(section_obj, f.name)
                if mask_sensitive:
                    section_dict[f.name] = mask_value(f.name, val)
                else:
                    section_dict[f.name] = val
            out[section_name] = section_dict
        return out

    def to_safe_dict(self) -> Dict[str, Any]:
        return self.to_dict(mask_sensitive=True)

    def to_yaml(self, mask_sensitive: bool = False) -> str:
        return yaml.safe_dump(self.to_dict(mask_sensitive=mask_sensitive), sort_keys=False)

    def to_json(self, mask_sensitive: bool = False) -> str:
        return json.dumps(self.to_dict(mask_sensitive=mask_sensitive), indent=2)

    def config_fingerprint(self) -> str:
        digest = hashlib.sha256(self.to_json(mask_sensitive=False).encode("utf-8")).hexdigest()
        return digest

    def save_to_file(
        self, path: Union[str, Path], format: str = "yaml", mask_sensitive: bool = False
    ):
        path = Path(path).expanduser()
        data = self.to_dict(mask_sensitive=mask_sensitive)
        try:
            with open(path, "w", encoding="utf-8") as f:
                if format.lower() in ("yaml", "yml"):
                    yaml.safe_dump(data, f, sort_keys=False)
                elif format.lower() == "json":
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            logger.info(f"Configuration saved to: {path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            raise

    # ------------------------------ Secrets ---------------------------------

    def get_secret(self, key: str) -> Optional[str]:
        val = self._secret_manager.get(key)
        if val is None:
            logger.warning(f"Secret not found: {key}")
        return val

    # ------------------------------ Dynamic Reload (Optional) ---------------------------------

    def enable_auto_reload(
        self,
        files: Union[str, Path, List[Union[str, Path]]],
        interval: float = 1.5,
        debounce: float = 0.5,
        on_reload: Optional[Callable[["DuetMindConfig"], None]] = None,
    ):
        """
        Simple polling-based reloader (avoids mandatory watchdog dependency).
        If watchdog is installed, you could extend this to a FS event-based approach.
        """
        if isinstance(files, (str, Path)):
            watch_files = [Path(files).resolve()]
        else:
            watch_files = [Path(f).resolve() for f in files]

        mtimes: Dict[Path, float] = {}
        for p in watch_files:
            if p.exists():
                mtimes[p] = p.stat().st_mtime

        def watcher():
            last_reload = 0.0
            while not self._watch_stop_event.is_set():
                for p in watch_files:
                    if not p.exists():
                        continue
                    try:
                        current_mtime = p.stat().st_mtime
                        if p not in mtimes:
                            mtimes[p] = current_mtime
                        elif current_mtime > mtimes[p]:
                            mtimes[p] = current_mtime
                            now = time.time()
                            if now - last_reload > debounce:
                                logger.info(f"Detected config file change: {p}")
                                self._reload_files(watch_files)
                                last_reload = now
                                if on_reload:
                                    try:
                                        on_reload(self)
                                    except Exception as e:
                                        logger.error(f"on_reload callback error: {e}")
                    except Exception as e:
                        logger.debug(f"Watcher error on {p}: {e}")
                time.sleep(interval)

        t = threading.Thread(target=watcher, daemon=True, name="ConfigReloader")
        t.start()
        self._watch_threads.append(t)
        logger.info("Auto-reload enabled for config files")

    def _reload_files(self, files: List[Path]) -> None:
        # Reload logic: apply files again then env then overrides, then validate
        for f in files:
            try:
                if f.exists():
                    data = load_file_any(f)
                    self._apply_dict(data, provenance=f"reload:{f.name}")
            except Exception as e:
                logger.error(f"Error reloading {f}: {e}")
        # Re-apply environment & overrides to maintain precedence
        self.load_environment()
        if self._overrides:
            self.load_overrides(self._overrides)
        try:
            self.validate()
        except Exception as e:
            logger.error(f"Validation failed after reload: {e}")

    def stop_auto_reload(self):
        self._watch_stop_event.set()
        for t in self._watch_threads:
            if t.is_alive():
                t.join(timeout=2)

    # ------------------------------ Schema Generation ---------------------------------

    def schema(self) -> Dict[str, Any]:
        """
        Generate a simple schema representation for documentation or UI forms.
        """
        schema_out: Dict[str, Any] = {}
        for section_name in self._section_names:
            section_obj = getattr(self, section_name)
            section_schema = {}
            hints = get_type_hints(type(section_obj))
            for f in fields(section_obj):
                f_type = hints.get(f.name, str)
                section_schema[f.name] = {
                    "type": getattr(f_type, "__name__", str(f_type)),
                    "default": f.default if f.default is not MISSING else None,
                    "current": getattr(section_obj, f.name),
                    "sensitive": bool(SENSITIVE_KEY_PATTERN.search(f.name)),
                    "provenance": self._provenance.get((section_name, f.name), "default"),
                }
            schema_out[section_name] = section_schema
        return schema_out

    # ------------------------------ Representation ---------------------------------

    def __repr__(self):
        return f"<DuetMindConfig fingerprint={self.config_fingerprint()[:12]}>"


# End of module
