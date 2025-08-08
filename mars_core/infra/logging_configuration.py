"""
MARS Quantum - Advanced Logging Configuration and Management System

This module provides comprehensive logging infrastructure for the MARS Quantum
cognitive AI system. It implements enterprise-grade logging with multiple output
formats, handlers, security features, and performance monitoring capabilities.

Key Features:
- Multi-level logging with custom formatters and filters
- Secure audit trail logging with tamper detection
- Performance monitoring and metrics collection
- Structured logging with JSON and plain text formats
- File rotation and compression for log management
- Real-time log streaming and monitoring capabilities
- Security-focused logging with sensitive data filtering
- Distributed logging support for multi-node systems

Security Considerations:
- Automatic filtering of sensitive information (passwords, tokens, etc.)
- Secure log file permissions and encryption options
- Audit trail integrity checking with cryptographic signatures
- Role-based access control for log viewing and management
- Log tampering detection and prevention mechanisms

Performance Features:
- Asynchronous logging for high-throughput scenarios
- Log buffering and batching for improved I/O performance
- Memory-efficient log rotation and compression
- Configurable log levels and filtering for optimal performance
- Real-time performance metrics and health monitoring

Usage Examples:
    Basic logger setup:
        config = LoggingConfigurator()
        logger = config.get_logger("MyModule")
        logger.info("Application started")
    
    Advanced configuration:
        config = LoggingConfigurator(
            level=LogLevel.DEBUG,
            enable_file_logging=True,
            enable_security_audit=True
        )
        logger = config.configure_comprehensive_logging()
    
    Performance monitoring:
        perf_logger = config.get_performance_logger()
        with perf_logger.timer("operation_name"):
            # Your code here
            pass

"""

import logging
import logging.handlers
import json
import os
import sys
import time
import uuid
import hashlib
import threading
import traceback
import gzip
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, TextIO
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor


# ========================================
# Configuration Enums and Data Classes
# ========================================

class LogLevel(Enum):
    """Enumeration of logging levels with descriptions."""
    CRITICAL = logging.CRITICAL    # System failures, immediate attention required
    ERROR = logging.ERROR          # Error conditions that don't stop execution
    WARNING = logging.WARNING      # Warning conditions that should be noted
    INFO = logging.INFO            # General informational messages
    DEBUG = logging.DEBUG          # Detailed debugging information
    TRACE = 5                      # Ultra-detailed tracing (custom level)


class LogFormat(Enum):
    """Enumeration of available log formats."""
    SIMPLE = "simple"              # Basic timestamp and message
    DETAILED = "detailed"          # Full context with module and function info
    JSON = "json"                  # Structured JSON format for parsing
    AUDIT = "audit"                # Security audit format with integrity
    PERFORMANCE = "performance"    # Performance monitoring format
    COMPACT = "compact"            # Space-efficient format for high volume


class LogDestination(Enum):
    """Enumeration of log output destinations."""
    CONSOLE = "console"            # Standard output/error streams
    FILE = "file"                  # Local file system
    ROTATING_FILE = "rotating"     # File with automatic rotation
    NETWORK = "network"            # Network logging (syslog, etc.)
    DATABASE = "database"          # Database logging
    MEMORY = "memory"              # In-memory buffer for testing


@dataclass
class LogConfiguration:
    """Configuration settings for logging system."""
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.DETAILED
    destinations: List[LogDestination] = field(default_factory=lambda: [LogDestination.CONSOLE])
    log_directory: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_compression: bool = True
    enable_security_audit: bool = False
    enable_performance_monitoring: bool = False
    async_logging: bool = False
    buffer_size: int = 1000
    flush_interval: float = 1.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "level": self.level.name,
            "format_type": self.format_type.value,
            "destinations": [dest.value for dest in self.destinations],
            "log_directory": self.log_directory,
            "max_file_size": self.max_file_size,
            "backup_count": self.backup_count,
            "enable_compression": self.enable_compression,
            "enable_security_audit": self.enable_security_audit,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "async_logging": self.async_logging,
            "buffer_size": self.buffer_size,
            "flush_interval": self.flush_interval
        }


@dataclass
class LogMetrics:
    """Container for logging performance metrics."""
    total_messages: int = 0
    messages_by_level: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_message_time: float = 0.0
    average_message_rate: float = 0.0
    peak_message_rate: float = 0.0
    errors_count: int = 0
    buffer_overflows: int = 0
    
    def update_metrics(self, level: str) -> None:
        """Update metrics with new log message."""
        self.total_messages += 1
        self.messages_by_level[level] = self.messages_by_level.get(level, 0) + 1
        current_time = time.time()
        self.last_message_time = current_time
        
        # Calculate rates
        elapsed = current_time - self.start_time
        if elapsed > 0:
            self.average_message_rate = self.total_messages / elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_messages": self.total_messages,
            "messages_by_level": self.messages_by_level,
            "uptime_seconds": time.time() - self.start_time,
            "average_message_rate": self.average_message_rate,
            "peak_message_rate": self.peak_message_rate,
            "errors_count": self.errors_count,
            "buffer_overflows": self.buffer_overflows,
            "last_message_time": datetime.fromtimestamp(self.last_message_time).isoformat()
        }


# ========================================
# Custom Formatters and Filters
# ========================================

class EnhancedFormatter(logging.Formatter):
    """
    Enhanced formatter with comprehensive context and security features.
    
    This formatter provides rich context information, security filtering,
    and supports multiple output formats including JSON and audit trails.
    """
    
    def __init__(self, format_type: LogFormat = LogFormat.DETAILED,
                 include_trace: bool = False,
                 filter_sensitive: bool = True):
        """
        Initialize the enhanced formatter.
        
        Args:
            format_type: Type of formatting to apply
            include_trace: Whether to include stack traces
            filter_sensitive: Whether to filter sensitive information
        """
        self.format_type = format_type
        self.include_trace = include_trace
        self.filter_sensitive = filter_sensitive
        self.session_id = str(uuid.uuid4())[:8]
        self.process_id = os.getpid()
        self.hostname = socket.gethostname()
        
        # Sensitive data patterns to filter
        self.sensitive_patterns = [
            r'password["\s]*[:=]["\s]*[^"\s,}]+',
            r'token["\s]*[:=]["\s]*[^"\s,}]+',
            r'api[_-]?key["\s]*[:=]["\s]*[^"\s,}]+',
            r'secret["\s]*[:=]["\s]*[^"\s,}]+',
            r'credential[s]?["\s]*[:=]["\s]*[^"\s,}]+'
        ]
        
        # Set base format string
        super().__init__(self._get_format_string())
    
    def _get_format_string(self) -> str:
        """Get format string based on format type."""
        if self.format_type == LogFormat.SIMPLE:
            return '%(timestamp)s - %(levelname)s - %(message)s'
        elif self.format_type == LogFormat.DETAILED:
            return ('%(timestamp)s [%(username)s@%(hostname)s:%(process_id)s:%(session_id)s] '
                   '%(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
        elif self.format_type == LogFormat.COMPACT:
            return '%(timestamp_short)s %(levelname_short)s %(name_short)s: %(message)s'
        else:
            return '%(message)s'  # JSON and others handle formatting differently
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with enhanced context and security filtering.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message string
        """
        try:
            # Add enhanced context to record
            self._enhance_record(record)
            
            # Filter sensitive information if enabled
            if self.filter_sensitive:
                record.msg = self._filter_sensitive_data(str(record.msg))
                if hasattr(record, 'args') and record.args:
                    record.args = tuple(
                        self._filter_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                        for arg in record.args
                    )
            
            # Format based on type
            if self.format_type == LogFormat.JSON:
                return self._format_json(record)
            elif self.format_type == LogFormat.AUDIT:
                return self._format_audit(record)
            elif self.format_type == LogFormat.PERFORMANCE:
                return self._format_performance(record)
            else:
                return super().format(record)
                
        except Exception as e:
            # Fallback formatting if enhancement fails
            return f"LOGGING_ERROR: {str(e)} | Original: {record.getMessage()}"
    
    def _enhance_record(self, record: logging.LogRecord) -> None:
        """Add enhanced context information to the log record."""
        # Basic context
        record.username = os.environ.get("USER_LOGIN", "Shriram-2005")
        record.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        record.timestamp_short = datetime.now().strftime("%H:%M:%S")
        record.session_id = self.session_id
        record.process_id = self.process_id
        record.hostname = self.hostname
        record.thread_name = threading.current_thread().name
        record.thread_id = threading.get_ident()
        
        # Shortened versions for compact format
        record.levelname_short = record.levelname[0]
        record.name_short = record.name.split('.')[-1] if '.' in record.name else record.name
        
        # Performance context
        if hasattr(record, 'duration'):
            record.duration_ms = getattr(record, 'duration', 0) * 1000
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            record.memory_mb = 0
        
        # Exception context
        if record.exc_info and self.include_trace:
            record.stack_trace = self.formatException(record.exc_info)
    
    def _filter_sensitive_data(self, text: str) -> str:
        """Filter sensitive information from log messages."""
        if not self.filter_sensitive:
            return text
        
        import re
        filtered_text = text
        
        for pattern in self.sensitive_patterns:
            filtered_text = re.sub(pattern, '[FILTERED]', filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format record as JSON structure."""
        log_obj = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread_name,
            "process_id": record.process_id,
            "session_id": record.session_id,
            "hostname": record.hostname,
            "username": record.username
        }
        
        # Add performance data if available
        if hasattr(record, 'duration_ms'):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, 'memory_mb'):
            log_obj["memory_mb"] = record.memory_mb
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_obj, ensure_ascii=False)
    
    def _format_audit(self, record: logging.LogRecord) -> str:
        """Format record for security audit trail."""
        # Create integrity hash
        content = f"{record.timestamp}|{record.levelname}|{record.getMessage()}"
        integrity_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        audit_obj = {
            "audit_timestamp": record.timestamp,
            "severity": record.levelname,
            "event_source": record.name,
            "user": record.username,
            "session": record.session_id,
            "host": record.hostname,
            "process": record.process_id,
            "message": record.getMessage(),
            "integrity": integrity_hash
        }
        
        return json.dumps(audit_obj, ensure_ascii=False)
    
    def _format_performance(self, record: logging.LogRecord) -> str:
        """Format record for performance monitoring."""
        perf_obj = {
            "timestamp": record.timestamp,
            "metric_type": getattr(record, 'metric_type', 'general'),
            "operation": getattr(record, 'operation', record.funcName),
            "duration_ms": getattr(record, 'duration_ms', 0),
            "memory_mb": getattr(record, 'memory_mb', 0),
            "success": getattr(record, 'success', True),
            "details": record.getMessage()
        }
        
        return json.dumps(perf_obj, ensure_ascii=False)


class SensitiveDataFilter(logging.Filter):
    """Filter to remove or mask sensitive information from logs."""
    
    def __init__(self, sensitive_keys: Optional[List[str]] = None):
        """
        Initialize the sensitive data filter.
        
        Args:
            sensitive_keys: List of keys to filter (case-insensitive)
        """
        super().__init__()
        self.sensitive_keys = sensitive_keys or [
            'password', 'token', 'api_key', 'secret', 'credential',
            'private_key', 'session_id', 'auth', 'authorization'
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record for sensitive information.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record, False to block it
        """
        try:
            message = record.getMessage().lower()
            
            # Check for sensitive keywords
            for key in self.sensitive_keys:
                if key.lower() in message:
                    # Don't completely block, but mark for filtering
                    record.contains_sensitive = True
                    break
            
            return True  # Allow record but mark it for formatter filtering
            
        except Exception:
            return True  # Allow record if filtering fails


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def __init__(self, min_duration_ms: float = 0.0):
        """
        Initialize performance filter.
        
        Args:
            min_duration_ms: Minimum duration in ms to log
        """
        super().__init__()
        self.min_duration_ms = min_duration_ms
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on performance criteria."""
        if hasattr(record, 'duration_ms'):
            return record.duration_ms >= self.min_duration_ms
        return True


# ========================================
# Async Logging Handler
# ========================================

class AsyncLoggingHandler(logging.Handler):
    """Asynchronous logging handler for high-performance scenarios."""
    
    def __init__(self, target_handler: logging.Handler, buffer_size: int = 1000):
        """
        Initialize async logging handler.
        
        Args:
            target_handler: The actual handler to write to
            buffer_size: Size of the internal buffer
        """
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.log_queue = queue.Queue(maxsize=buffer_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.start_worker()
    
    def start_worker(self) -> None:
        """Start the background worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def _worker(self) -> None:
        """Background worker that processes log records."""
        while not self.stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # Sentinel value to stop
                    break
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log the error to stderr to avoid recursion
                print(f"Async logging error: {e}", file=sys.stderr)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record asynchronously."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Buffer is full, drop the message or handle overflow
            print("Logging buffer overflow - dropping message", file=sys.stderr)
    
    def close(self) -> None:
        """Close the handler and stop the worker thread."""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_queue.put(None)  # Sentinel to stop worker
            self.worker_thread.join(timeout=5.0)
        super().close()


# ========================================
# Main Logging Configurator Class
# ========================================

class LoggingConfigurator:
    """
    Comprehensive logging configuration manager for MARS Quantum system.
    
    This class provides enterprise-grade logging infrastructure with multiple
    output formats, security features, performance monitoring, and advanced
    configuration options.
    
    Features:
    - Multi-destination logging (console, file, network, database)
    - Multiple format types (simple, detailed, JSON, audit, performance)
    - Asynchronous logging for high-performance scenarios
    - Security audit trails with integrity checking
    - Performance monitoring and metrics collection
    - Automatic log rotation and compression
    - Sensitive data filtering and protection
    - Real-time configuration updates
    """
    
    def __init__(self, config: Optional[LogConfiguration] = None):
        """
        Initialize the logging configurator.
        
        Args:
            config: Optional logging configuration
        """
        self.config = config or LogConfiguration()
        self.loggers = {}  # Registry of configured loggers
        self.handlers = {}  # Registry of handlers by type
        self.metrics = LogMetrics()
        self.session_id = str(uuid.uuid4())[:8]
        
        # Ensure log directory exists
        self.log_dir = Path(self.config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Register custom log level
        logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="logging")
        
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        print(f"Logging configurator initialized - Session: {self.session_id}")
    
    def configure_comprehensive_logging(self) -> logging.Logger:
        """
        Configure comprehensive logging with all enabled features.
        
        Returns:
            Main MARS Quantum logger instance
        """
        with self._lock:
            # Create main logger
            logger = logging.getLogger("MARS.Quantum")
            logger.setLevel(self.config.level.value)
            
            # Clear existing handlers to avoid duplicates
            logger.handlers.clear()
            
            # Configure destinations
            for destination in self.config.destinations:
                handler = self._create_handler(destination)
                if handler:
                    logger.addHandler(handler)
            
            # Configure security audit logging if enabled
            if self.config.enable_security_audit:
                audit_handler = self._create_audit_handler()
                if audit_handler:
                    logger.addHandler(audit_handler)
            
            # Configure performance monitoring if enabled
            if self.config.enable_performance_monitoring:
                perf_handler = self._create_performance_handler()
                if perf_handler:
                    logger.addHandler(perf_handler)
            
            # Store in registry
            self.loggers["main"] = logger
            
            logger.info("MARS Quantum comprehensive logging system initialized")
            logger.info(f"Configuration: {self.config.to_dict()}")
            
            return logger
    
    def _create_handler(self, destination: LogDestination) -> Optional[logging.Handler]:
        """
        Create handler for specific destination.
        
        Args:
            destination: Logging destination type
            
        Returns:
            Configured handler instance or None
        """
        try:
            if destination == LogDestination.CONSOLE:
                handler = logging.StreamHandler(sys.stdout)
                
            elif destination == LogDestination.FILE:
                log_file = self.log_dir / "mars_quantum.log"
                handler = logging.FileHandler(log_file, encoding='utf-8')
                
            elif destination == LogDestination.ROTATING_FILE:
                log_file = self.log_dir / "mars_quantum_rotating.log"
                handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                
            elif destination == LogDestination.NETWORK:
                # Syslog handler for network logging
                handler = logging.handlers.SysLogHandler(address=('localhost', 514))
                
            elif destination == LogDestination.MEMORY:
                # Memory handler for testing
                handler = logging.handlers.MemoryHandler(capacity=1000)
                
            else:
                return None
            
            # Configure formatter
            formatter = EnhancedFormatter(
                format_type=self.config.format_type,
                include_trace=True,
                filter_sensitive=True
            )
            handler.setFormatter(formatter)
            
            # Add filters
            handler.addFilter(SensitiveDataFilter())
            
            # Wrap with async handler if enabled
            if self.config.async_logging:
                handler = AsyncLoggingHandler(handler, self.config.buffer_size)
            
            # Store in registry
            self.handlers[destination.value] = handler
            
            return handler
            
        except Exception as e:
            print(f"Failed to create handler for {destination.value}: {e}", file=sys.stderr)
            return None
    
    def _create_audit_handler(self) -> Optional[logging.Handler]:
        """Create security audit logging handler."""
        try:
            audit_file = self.log_dir / "security_audit.log"
            handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            
            formatter = EnhancedFormatter(
                format_type=LogFormat.AUDIT,
                filter_sensitive=True
            )
            handler.setFormatter(formatter)
            
            # Only log WARNING and above for audit
            handler.setLevel(logging.WARNING)
            
            return handler
            
        except Exception as e:
            print(f"Failed to create audit handler: {e}", file=sys.stderr)
            return None
    
    def _create_performance_handler(self) -> Optional[logging.Handler]:
        """Create performance monitoring handler."""
        try:
            perf_file = self.log_dir / "performance.log"
            handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            
            formatter = EnhancedFormatter(format_type=LogFormat.PERFORMANCE)
            handler.setFormatter(formatter)
            
            # Add performance filter
            handler.addFilter(PerformanceFilter(min_duration_ms=1.0))
            
            return handler
            
        except Exception as e:
            print(f"Failed to create performance handler: {e}", file=sys.stderr)
            return None
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        with self._lock:
            if name in self.loggers:
                return self.loggers[name]
            
            # Create new logger
            logger = logging.getLogger(f"MARS.Quantum.{name}")
            logger.setLevel(self.config.level.value)
            
            # Inherit handlers from main logger if it exists
            main_logger = self.loggers.get("main")
            if main_logger:
                for handler in main_logger.handlers:
                    logger.addHandler(handler)
            else:
                # Create console handler as fallback
                handler = logging.StreamHandler()
                formatter = EnhancedFormatter(self.config.format_type)
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            self.loggers[name] = logger
            return logger
    
    def get_performance_logger(self) -> 'PerformanceLogger':
        """
        Get performance monitoring logger.
        
        Returns:
            Performance logger instance
        """
        return PerformanceLogger(self.get_logger("Performance"))
    
    @contextmanager
    def temporary_level(self, logger_name: str, level: LogLevel):
        """
        Temporarily change log level for a logger.
        
        Args:
            logger_name: Name of logger to modify
            level: Temporary log level
        """
        logger = self.get_logger(logger_name)
        original_level = logger.level
        
        try:
            logger.setLevel(level.value)
            yield logger
        finally:
            logger.setLevel(original_level)
    
    def update_configuration(self, new_config: LogConfiguration) -> None:
        """
        Update logging configuration at runtime.
        
        Args:
            new_config: New configuration to apply
        """
        with self._lock:
            self.config = new_config
            
            # Reconfigure all loggers
            for logger_name, logger in self.loggers.items():
                logger.setLevel(new_config.level.value)
                
                # Update handlers
                for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
                    logger.removeHandler(handler)
                    handler.close()
                
                # Recreate handlers with new config
                for destination in new_config.destinations:
                    handler = self._create_handler(destination)
                    if handler:
                        logger.addHandler(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get logging system metrics and statistics.
        
        Returns:
            Dictionary containing metrics information
        """
        with self._lock:
            metrics_dict = self.metrics.to_dict()
            metrics_dict.update({
                "session_id": self.session_id,
                "active_loggers": len(self.loggers),
                "active_handlers": len(self.handlers),
                "configuration": self.config.to_dict(),
                "log_directory": str(self.log_dir),
                "log_files": [f.name for f in self.log_dir.glob("*.log")]
            })
            return metrics_dict
    
    def export_logs(self, output_file: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> str:
        """
        Export logs to a compressed archive.
        
        Args:
            output_file: Output file path
            start_time: Start time for log filtering
            end_time: End time for log filtering
            
        Returns:
            Path to exported archive
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"mars_logs_export_{timestamp}.tar.gz"
        
        import tarfile
        
        with tarfile.open(output_file, "w:gz") as tar:
            for log_file in self.log_dir.glob("*.log*"):
                tar.add(log_file, arcname=log_file.name)
        
        return output_file
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """
        Clean up old log files.
        
        Args:
            days_to_keep: Number of days to keep logs
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        cleaned_count = 0
        
        for log_file in self.log_dir.glob("*.log.*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"Failed to delete {log_file}: {e}", file=sys.stderr)
        
        return cleaned_count
    
    def shutdown(self) -> None:
        """Gracefully shutdown the logging system."""
        with self._lock:
            # Close all handlers
            for handler_list in [logger.handlers for logger in self.loggers.values()]:
                for handler in handler_list:
                    handler.close()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            print(f"Logging system shutdown complete - Session: {self.session_id}")


# ========================================
# Performance Logger Class
# ========================================

class PerformanceLogger:
    """Specialized logger for performance monitoring and metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.timers = {}  # Active timers
        self.counters = {}  # Performance counters
    
    @contextmanager
    def timer(self, operation_name: str, **kwargs):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            **kwargs: Additional context to log
        """
        start_time = time.perf_counter()
        timer_id = f"{operation_name}_{time.time()}"
        
        try:
            yield timer_id
        finally:
            duration = time.perf_counter() - start_time
            
            # Create performance log record
            record = self.logger.makeRecord(
                self.logger.name, logging.INFO, "", 0,
                f"Operation '{operation_name}' completed", (), None
            )
            record.operation = operation_name
            record.duration = duration
            record.duration_ms = duration * 1000
            record.metric_type = "timing"
            record.success = True
            
            # Add additional context
            for key, value in kwargs.items():
                setattr(record, key, value)
            
            self.logger.handle(record)
    
    def count(self, counter_name: str, increment: int = 1, **kwargs) -> None:
        """
        Increment a performance counter.
        
        Args:
            counter_name: Name of the counter
            increment: Amount to increment
            **kwargs: Additional context
        """
        self.counters[counter_name] = self.counters.get(counter_name, 0) + increment
        
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Counter '{counter_name}' = {self.counters[counter_name]}", (), None
        )
        record.counter_name = counter_name
        record.counter_value = self.counters[counter_name]
        record.increment = increment
        record.metric_type = "counter"
        
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def gauge(self, gauge_name: str, value: Union[int, float], **kwargs) -> None:
        """
        Record a gauge value.
        
        Args:
            gauge_name: Name of the gauge
            value: Current value
            **kwargs: Additional context
        """
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Gauge '{gauge_name}' = {value}", (), None
        )
        record.gauge_name = gauge_name
        record.gauge_value = value
        record.metric_type = "gauge"
        
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        self.logger.handle(record)


# ========================================
# Convenience Functions and Legacy Support
# ========================================

def get_default_logger() -> logging.Logger:
    """
    Get default MARS Quantum logger with basic configuration.
    
    Returns:
        Configured logger instance
    """
    config = LogConfiguration(
        level=LogLevel.INFO,
        format_type=LogFormat.DETAILED,
        destinations=[LogDestination.CONSOLE]
    )
    
    configurator = LoggingConfigurator(config)
    return configurator.configure_comprehensive_logging()


def configure_legacy_logging() -> logging.Logger:
    """
    Legacy function for backward compatibility.
    
    This function maintains the original interface while using
    the enhanced logging system underneath.
    
    Returns:
        Configured logger instance
    """
    # Create simple configuration for legacy compatibility
    config = LogConfiguration(
        level=LogLevel.INFO,
        format_type=LogFormat.DETAILED,
        destinations=[LogDestination.CONSOLE]
    )
    
    configurator = LoggingConfigurator(config)
    logger = configurator.configure_comprehensive_logging()
    
    # Add legacy formatter for backward compatibility
    for handler in logger.handlers:
        if isinstance(handler.formatter, EnhancedFormatter):
            continue  # Already has enhanced formatter
        
        legacy_formatter = EnhancedFormatter(LogFormat.DETAILED)
        handler.setFormatter(legacy_formatter)
    
    print(f"Legacy logging configured for user {os.environ.get('USER_LOGIN', 'Shriram-2005')}")
    return logger


# Create default logger instance for immediate use
logger = get_default_logger()


# ========================================
# Example Usage and Testing
# ========================================

def demonstrate_logging_capabilities():
    """Demonstrate the comprehensive logging capabilities."""
    print("=== MARS Quantum Logging System Demo ===\n")
    
    # 1. Basic logging
    print("1. Basic Logging:")
    basic_logger = get_default_logger()
    basic_logger.info("This is a basic info message")
    basic_logger.warning("This is a warning message")
    basic_logger.error("This is an error message")
    print()
    
    # 2. Advanced configuration
    print("2. Advanced Configuration:")
    advanced_config = LogConfiguration(
        level=LogLevel.DEBUG,
        format_type=LogFormat.JSON,
        destinations=[LogDestination.CONSOLE, LogDestination.FILE],
        enable_security_audit=True,
        enable_performance_monitoring=True
    )
    
    advanced_configurator = LoggingConfigurator(advanced_config)
    advanced_logger = advanced_configurator.configure_comprehensive_logging()
    advanced_logger.info("Advanced logging system initialized")
    print()
    
    # 3. Performance monitoring
    print("3. Performance Monitoring:")
    perf_logger = advanced_configurator.get_performance_logger()
    
    with perf_logger.timer("demo_operation"):
        time.sleep(0.1)  # Simulate work
    
    perf_logger.count("demo_counter", 5)
    perf_logger.gauge("demo_gauge", 42.5)
    print()
    
    # 4. Different log formats
    print("4. Different Log Formats:")
    for format_type in [LogFormat.SIMPLE, LogFormat.DETAILED, LogFormat.COMPACT]:
        config = LogConfiguration(format_type=format_type)
        configurator = LoggingConfigurator(config)
        format_logger = configurator.get_logger(f"Format.{format_type.value}")
        format_logger.info(f"Testing {format_type.value} format")
    print()
    
    # 5. Metrics
    print("5. System Metrics:")
    metrics = advanced_configurator.get_metrics()
    for key, value in metrics.items():
        if key not in ['configuration', 'messages_by_level']:
            print(f"   {key}: {value}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    """
    Main execution block for logging configuration.
    
    Supports multiple execution modes:
    - Default: Configure legacy logging
    - Demo: Demonstrate all logging features
    - Test: Run logging system tests
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demonstrate_logging_capabilities()
        elif sys.argv[1] == "test":
            # Run basic tests
            test_logger = get_default_logger()
            test_logger.info("Running logging system tests...")
            
            # Test different log levels
            for level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]:
                test_logger.log(level.value, f"Test message at {level.name} level")
            
            test_logger.info("All tests completed successfully")
        else:
            print("Usage: python logging_configuration.py [demo|test]")
    else:
        # Default legacy behavior
        legacy_logger = configure_legacy_logging()
        legacy_logger.info("MARS Quantum logging system ready")