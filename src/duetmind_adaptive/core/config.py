"""
Configuration Management for DuetMind Adaptive

Provides secure, centralized configuration management with environment
variable support and validation.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    auth_enabled: bool = True
    api_key_required: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 100
    session_timeout_minutes: int = 30
    encryption_key_length: int = 32
    password_min_length: int = 12
    enable_2fa: bool = False
    allowed_origins: list = field(default_factory=lambda: ["localhost", "127.0.0.1"])

@dataclass  
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "duetmind.db"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_connections: int = 20
    connection_timeout: int = 30

@dataclass
class NeuralNetworkConfig:
    """Neural network configuration settings"""
    input_size: int = 32
    hidden_layers: list = field(default_factory=lambda: [64, 32, 16])
    output_size: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.3
    activation_function: str = "relu"
    optimizer: str = "adam"
    
@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_enabled: bool = True
    ssl_enabled: bool = False  # Default to False for development
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    max_request_size: int = 16 * 1024 * 1024  # 16MB

class DuetMindConfig:
    """
    Centralized configuration manager with security and validation.
    
    Supports loading from:
    - Environment variables
    - YAML/JSON configuration files
    - Programmatic overrides
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.neural_network = NeuralNetworkConfig()
        self.api = APIConfig()
        
        # Load configuration hierarchy
        self._load_defaults()
        if config_path:
            self._load_file_config(config_path)
        self._load_environment_config()
        self._validate_config()
    
    def _load_defaults(self):
        """Load default configuration values"""
        logger.info("Loading default configuration")
        
    def _load_file_config(self, config_path: Union[str, Path]):
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {config_path.suffix}")
                    return
                    
            self._apply_config_data(config_data)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
    
    def _load_environment_config(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Security
            'DUETMIND_AUTH_ENABLED': ('security', 'auth_enabled', bool),
            'DUETMIND_API_KEY_REQUIRED': ('security', 'api_key_required', bool),
            'DUETMIND_RATE_LIMIT_ENABLED': ('security', 'rate_limit_enabled', bool),
            'DUETMIND_MAX_REQUESTS_PER_MINUTE': ('security', 'max_requests_per_minute', int),
            'DUETMIND_SESSION_TIMEOUT': ('security', 'session_timeout_minutes', int),
            
            # Database  
            'DUETMIND_DB_TYPE': ('database', 'type', str),
            'DUETMIND_DB_HOST': ('database', 'host', str),
            'DUETMIND_DB_PORT': ('database', 'port', int),
            'DUETMIND_DB_NAME': ('database', 'name', str),
            'DUETMIND_DB_USER': ('database', 'username', str),
            'DUETMIND_DB_PASSWORD': ('database', 'password', str),
            
            # API
            'DUETMIND_API_HOST': ('api', 'host', str),
            'DUETMIND_API_PORT': ('api', 'port', int),
            'DUETMIND_API_DEBUG': ('api', 'debug', bool),
            'DUETMIND_SSL_ENABLED': ('api', 'ssl_enabled', bool),
            'DUETMIND_SSL_CERT_PATH': ('api', 'ssl_cert_path', str),
            'DUETMIND_SSL_KEY_PATH': ('api', 'ssl_key_path', str),
            
            # Neural Network
            'DUETMIND_NN_INPUT_SIZE': ('neural_network', 'input_size', int),
            'DUETMIND_NN_LEARNING_RATE': ('neural_network', 'learning_rate', float),
            'DUETMIND_NN_BATCH_SIZE': ('neural_network', 'batch_size', int),
            'DUETMIND_NN_EPOCHS': ('neural_network', 'epochs', int),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_func == bool:
                        parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        parsed_value = type_func(value)
                    
                    section_obj = getattr(self, section)
                    setattr(section_obj, key, parsed_value)
                    logger.debug(f"Set {section}.{key} = {parsed_value} from environment")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid environment variable {env_var}={value}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to config objects"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        logger.debug(f"Set {section_name}.{key} = {value}")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate security settings
        if self.security.max_requests_per_minute < 1:
            errors.append("max_requests_per_minute must be > 0")
        
        if self.security.session_timeout_minutes < 1:
            errors.append("session_timeout_minutes must be > 0")
            
        if self.security.password_min_length < 8:
            errors.append("password_min_length should be at least 8")
        
        # Validate database settings
        if self.database.port < 1 or self.database.port > 65535:
            errors.append("database port must be between 1-65535")
        
        # Validate API settings  
        if self.api.port < 1 or self.api.port > 65535:
            errors.append("API port must be between 1-65535")
        
        if self.api.ssl_enabled and not (self.api.ssl_cert_path and self.api.ssl_key_path):
            errors.append("SSL cert and key paths required when SSL enabled")
        
        # Validate neural network settings
        if self.neural_network.learning_rate <= 0 or self.neural_network.learning_rate > 1:
            errors.append("learning_rate must be between 0 and 1")
        
        if self.neural_network.batch_size < 1:
            errors.append("batch_size must be > 0")
            
        if self.neural_network.epochs < 1:
            errors.append("epochs must be > 0")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'security': self.security.__dict__,
            'database': self.database.__dict__,
            'neural_network': self.neural_network.__dict__,
            'api': self.api.__dict__,
        }
    
    def save_to_file(self, path: Union[str, Path], format: str = 'yaml'):
        """Save configuration to file"""
        path = Path(path)
        config_dict = self.to_dict()
        
        try:
            with open(path, 'w') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            raise
            
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value from environment or secure storage"""
        # First try environment variable
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # TODO: Add support for HashiCorp Vault, AWS Secrets Manager, etc.
        logger.warning(f"Secret {key} not found in environment")
        return None