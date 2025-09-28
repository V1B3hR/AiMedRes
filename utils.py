"""
AiMedRes - Utility Functions

This module contains common utility functions extracted from larger modules
to improve code reusability and maintainability.
"""

import hashlib
import uuid
from typing import Dict, Any, List
from datetime import datetime


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())


def generate_error_id() -> str:
    """Generate a unique error ID for error tracking"""
    return str(uuid.uuid4())


def hash_data(data: str) -> str:
    """Generate a hash for data integrity checking"""
    return hashlib.sha256(data.encode()).hexdigest()


def create_audit_event(event_type: str, user_id: str = None, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a standardized audit event structure"""
    return {
        'event_type': event_type,
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id or 'system',
        'event_id': generate_request_id(),
        'details': details or {}
    }


def format_response_time(response_time: float) -> str:
    """Format response time for display"""
    if response_time < 1.0:
        return f"{response_time * 1000:.0f}ms"
    else:
        return f"{response_time:.3f}s"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:252] + '...'
    
    return filename


def calculate_percentage(part: float, total: float) -> float:
    """Safely calculate percentage avoiding division by zero"""
    if total == 0:
        return 0.0
    return (part / total) * 100


def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_config_keys(config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Validate that required configuration keys are present"""
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    return {
        'valid': len(missing_keys) == 0,
        'missing_keys': missing_keys
    }


def safe_json_extract(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely extract nested JSON values using dot notation"""
    try:
        keys = path.split('.')
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default