"""Custom exceptions and error handling utilities."""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PipelineError:
    """Structured pipeline error information."""
    component: str
    message: str
    severity: ErrorSeverity
    details: str | None = None
    recoverable: bool = True


class SemanticLayerSeparationError(Exception):
    """Base exception for semantic layer separation."""
    pass


class ConfigurationError(SemanticLayerSeparationError):
    """Configuration-related errors."""
    pass


class ProcessingError(SemanticLayerSeparationError):
    """Image processing errors."""
    pass


class ModelError(SemanticLayerSeparationError):
    """Model loading/inference errors."""
    pass


T = TypeVar("T")


def safe_execute(func: Callable[..., T], *args: Any, component: str = "unknown", severity: ErrorSeverity = ErrorSeverity.MEDIUM, recoverable: bool = True, default: T | None = None, **kwargs: Any) -> T | None:
    """Execute a function safely with structured error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        component: Component name for error tracking
        severity: Error severity level
        recoverable: Whether error is recoverable
        default: Default value if execution fails
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    logger = logging.getLogger(__name__)
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error = PipelineError(
            component=component,
            message=str(e),
            severity=severity,
            details=traceback.format_exc(),
            recoverable=recoverable,
        )
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[severity]
        
        logger.log(
            log_level,
            f"[{error.component}] {error.message} (recoverable={error.recoverable})",
            exc_info=True if severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL) else False,
        )
        
        if not recoverable:
            raise
        return default
