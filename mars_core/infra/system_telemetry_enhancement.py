
"""
Enterprise System Telemetry Enhancement Platform

This module provides comprehensive system telemetry tracking, monitoring, and analytics
capabilities for enterprise environments with advanced features including:

Features:
- Multi-dimensional telemetry data collection and aggregation
- Real-time performance monitoring and alerting
- Advanced analytics and trend analysis
- Distributed telemetry with clustering support
- High-performance storage with multiple backends
- Compliance and audit trail capabilities
- Machine learning-powered anomaly detection
- Extensible plugin architecture for custom metrics
- Enterprise security and data encryption
- Advanced reporting and visualization support

Architecture:
- TelemetryCollector: Core data collection engine
- TelemetryStore: High-performance storage layer
- TelemetryAnalyzer: Analytics and insights engine
- TelemetryReporter: Reporting and visualization
- TelemetryMonitor: Real-time monitoring and alerting

"""

import os
import sys
import json
import time
import uuid
import hashlib
import threading
import asyncio
import logging
import socket
import platform
import traceback
import subprocess
import statistics
import sqlite3
import queue
import csv
import concurrent.futures
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable, Generator
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from pathlib import Path
from collections import defaultdict, deque, Counter
from contextlib import contextmanager
import warnings

# Optional dependencies with graceful fallbacks
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server  # type: ignore
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    PrometheusCounter = Histogram = Gauge = start_http_server = None

try:
    import influxdb_client  # type: ignore
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    influxdb_client = None

# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryLevel(IntEnum):
    """
    Telemetry data priority levels for filtering and processing.
    
    Provides hierarchical telemetry levels for efficient data management
    and processing optimization in enterprise environments.
    """
    TRACE = 0      # Detailed debugging information
    DEBUG = 1      # Debug-level telemetry data
    INFO = 2       # Informational telemetry events
    METRIC = 3     # Performance metrics and measurements
    WARNING = 4    # Warning-level telemetry data
    ERROR = 5      # Error-level telemetry events
    CRITICAL = 6   # Critical system telemetry data
    
    @property
    def description(self) -> str:
        """Get telemetry level description."""
        descriptions = {
            TelemetryLevel.TRACE: "Detailed tracing information for debugging",
            TelemetryLevel.DEBUG: "Debug-level telemetry for development",
            TelemetryLevel.INFO: "General informational telemetry data",
            TelemetryLevel.METRIC: "Performance metrics and measurements",
            TelemetryLevel.WARNING: "Warning conditions requiring attention",
            TelemetryLevel.ERROR: "Error conditions affecting operations",
            TelemetryLevel.CRITICAL: "Critical conditions requiring immediate attention"
        }
        return descriptions.get(self, "Unknown telemetry level")
    
    @property
    def priority(self) -> int:
        """Get numerical priority for processing order."""
        return self.value


class TelemetryCategory(Enum):
    """
    Comprehensive telemetry data categorization system.
    
    Organizes telemetry data into logical categories for efficient
    processing, analysis, and reporting in enterprise environments.
    """
    # System Performance Categories
    SYSTEM_PERFORMANCE = ("system.performance", "System performance metrics")
    CPU_METRICS = ("system.cpu", "CPU utilization and performance data")
    MEMORY_METRICS = ("system.memory", "Memory usage and allocation metrics")
    DISK_METRICS = ("system.disk", "Disk I/O and storage metrics")
    NETWORK_METRICS = ("system.network", "Network traffic and performance")
    
    # Application Performance Categories
    APPLICATION_PERFORMANCE = ("app.performance", "Application performance metrics")
    REQUEST_METRICS = ("app.request", "HTTP request and response metrics")
    DATABASE_METRICS = ("app.database", "Database query and connection metrics")
    CACHE_METRICS = ("app.cache", "Cache hit/miss and performance metrics")
    API_METRICS = ("app.api", "API call and response time metrics")
    
    # Business Intelligence Categories
    BUSINESS_METRICS = ("business.metrics", "Business intelligence data")
    USER_BEHAVIOR = ("business.user", "User interaction and behavior analytics")
    TRANSACTION_METRICS = ("business.transaction", "Transaction and commerce metrics")
    ENGAGEMENT_METRICS = ("business.engagement", "User engagement and retention")
    
    # Security and Compliance Categories
    SECURITY_METRICS = ("security.metrics", "Security-related telemetry data")
    AUDIT_TRAIL = ("security.audit", "Audit trail and compliance data")
    ACCESS_METRICS = ("security.access", "Access control and authentication metrics")
    THREAT_METRICS = ("security.threat", "Threat detection and response metrics")
    
    # Infrastructure Categories
    INFRASTRUCTURE_METRICS = ("infra.metrics", "Infrastructure monitoring data")
    CONTAINER_METRICS = ("infra.container", "Container and orchestration metrics")
    CLOUD_METRICS = ("infra.cloud", "Cloud service and resource metrics")
    DEPLOYMENT_METRICS = ("infra.deployment", "Deployment and release metrics")
    
    # Quality and Reliability Categories
    QUALITY_METRICS = ("quality.metrics", "Quality assurance metrics")
    ERROR_METRICS = ("quality.error", "Error tracking and analysis")
    AVAILABILITY_METRICS = ("quality.availability", "Service availability metrics")
    RELIABILITY_METRICS = ("quality.reliability", "System reliability metrics")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    @classmethod
    def from_code(cls, code: str) -> 'TelemetryCategory':
        """Convert code string to TelemetryCategory."""
        for category in cls:
            if category.code == code:
                return category
        return cls.SYSTEM_PERFORMANCE  # Default fallback


class MetricType(Enum):
    """Metric data type classification."""
    COUNTER = ("counter", "Monotonically increasing counter")
    GAUGE = ("gauge", "Point-in-time measurement")
    HISTOGRAM = ("histogram", "Distribution of values over time")
    SUMMARY = ("summary", "Statistical summary of observations")
    TIMER = ("timer", "Duration measurements")
    RATE = ("rate", "Rate of change over time")
    EVENT = ("event", "Discrete event or occurrence")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    @classmethod
    def from_code(cls, code: str) -> 'MetricType':
        """Convert code string to MetricType."""
        for metric_type in cls:
            if metric_type.code == code:
                return metric_type
        return cls.GAUGE  # Default fallback


class TelemetryStatus(Enum):
    """Telemetry data processing status."""
    PENDING = ("pending", "Awaiting processing")
    PROCESSING = ("processing", "Currently being processed")
    PROCESSED = ("processed", "Successfully processed")
    FAILED = ("failed", "Processing failed")
    ARCHIVED = ("archived", "Archived for long-term storage")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class SystemContext:
    """
    Comprehensive system context for telemetry data.
    
    Captures detailed system information for correlation
    and analysis of telemetry data across different environments.
    """
    # System identification
    hostname: Optional[str] = None
    system_id: Optional[str] = None
    environment: str = "development"
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    
    # Operating system context
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    os_architecture: Optional[str] = None
    kernel_version: Optional[str] = None
    
    # Hardware context
    cpu_count: Optional[int] = None
    cpu_model: Optional[str] = None
    total_memory: Optional[int] = None
    total_disk: Optional[int] = None
    
    # Process context
    process_id: Optional[int] = None
    process_name: Optional[str] = None
    parent_process_id: Optional[int] = None
    process_start_time: Optional[float] = None
    
    # Application context
    application_name: Optional[str] = None
    application_version: Optional[str] = None
    service_name: Optional[str] = None
    instance_id: Optional[str] = None
    
    # Container context (if applicable)
    container_id: Optional[str] = None
    container_image: Optional[str] = None
    container_name: Optional[str] = None
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    
    # Cloud context (if applicable)
    cloud_provider: Optional[str] = None
    cloud_region: Optional[str] = None
    instance_type: Optional[str] = None
    cluster_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if not self.hostname:
            self.hostname = socket.gethostname()
        
        if not self.os_name:
            self.os_name = platform.system()
        
        if not self.os_version:
            self.os_version = platform.release()
        
        if not self.os_architecture:
            self.os_architecture = platform.machine()
        
        if not self.process_id:
            self.process_id = os.getpid()
        
        if not self.system_id:
            self.system_id = hashlib.md5(f"{self.hostname}:{self.process_id}".encode()).hexdigest()[:12]
        
        # Populate hardware information if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                if not self.cpu_count:
                    self.cpu_count = psutil.cpu_count()
                if not self.total_memory:
                    self.total_memory = psutil.virtual_memory().total
                if not self.total_disk:
                    self.total_disk = psutil.disk_usage('/').total
            except Exception:
                pass  # Graceful degradation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class TelemetryMetric:
    """
    Comprehensive telemetry metric representation.
    
    Captures detailed metric information with metadata,
    context, and analytics support for enterprise telemetry systems.
    """
    # Core metric identification
    metric_id: str = field(default_factory=lambda: f"metric_{uuid.uuid4().hex[:12]}")
    name: str = "unknown_metric"
    category: TelemetryCategory = TelemetryCategory.SYSTEM_PERFORMANCE
    metric_type: MetricType = MetricType.GAUGE
    
    # Temporal properties
    timestamp: float = field(default_factory=time.time)
    collection_interval: Optional[float] = None
    retention_period: Optional[int] = None  # days
    
    # Metric value and metadata
    value: Union[int, float, str, Dict, List] = 0
    unit: Optional[str] = None
    precision: Optional[int] = None
    description: Optional[str] = None
    
    # Telemetry classification
    level: TelemetryLevel = TelemetryLevel.INFO
    status: TelemetryStatus = TelemetryStatus.PENDING
    priority: int = 5  # 1=highest, 10=lowest
    
    # Context and correlation
    system_context: Optional[SystemContext] = None
    user_context: Optional[Dict[str, Any]] = None
    request_context: Optional[Dict[str, Any]] = None
    
    # Metadata and tags
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    # Quality and validation
    quality_score: float = 1.0  # 0.0-1.0
    validation_rules: List[str] = field(default_factory=list)
    data_source: Optional[str] = None
    collection_method: Optional[str] = None
    
    # Processing and analytics
    aggregation_window: Optional[int] = None  # seconds
    aggregation_method: str = "last"  # last, avg, sum, min, max
    derived_metrics: List[str] = field(default_factory=list)
    
    # Correlation and relationships
    parent_metric_id: Optional[str] = None
    child_metric_ids: List[str] = field(default_factory=list)
    related_metric_ids: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Additional metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if self.system_context is None:
            self.system_context = SystemContext()
        
        # Add automatic tags
        self._add_automatic_tags()
        
        # Validate metric data
        self._validate_metric()
    
    def _add_automatic_tags(self) -> None:
        """Add automatic tags based on metric properties."""
        self.tags.update({
            "category": self.category.code,
            "type": self.metric_type.code,
            "level": self.level.name.lower(),
            "hostname": self.system_context.hostname if self.system_context else "unknown"
        })
        
        if self.system_context:
            if self.system_context.environment:
                self.tags["environment"] = self.system_context.environment
            if self.system_context.service_name:
                self.tags["service"] = self.system_context.service_name
            if self.system_context.application_name:
                self.tags["application"] = self.system_context.application_name
    
    def _validate_metric(self) -> None:
        """Validate metric data integrity."""
        try:
            # Basic validation
            if not self.name or not isinstance(self.name, str):
                self.quality_score *= 0.8
                logger.warning(f"Invalid metric name: {self.name}")
            
            # Value validation based on metric type
            if self.metric_type == MetricType.COUNTER and isinstance(self.value, (int, float)):
                if self.value < 0:
                    self.quality_score *= 0.5
                    logger.warning(f"Counter metric {self.name} has negative value: {self.value}")
            
            # Timestamp validation
            current_time = time.time()
            if abs(self.timestamp - current_time) > 86400:  # More than 1 day difference
                self.quality_score *= 0.7
                logger.warning(f"Metric {self.name} has suspicious timestamp: {self.timestamp}")
                
        except Exception as e:
            self.quality_score *= 0.3
            logger.error(f"Metric validation failed for {self.name}: {e}")
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the metric."""
        self.tags[key] = value
    
    def add_annotation(self, key: str, value: Any) -> None:
        """Add an annotation to the metric."""
        self.annotations[key] = value
    
    def get_age_seconds(self) -> float:
        """Get the age of the metric in seconds."""
        return time.time() - self.timestamp
    
    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if the metric has expired."""
        return self.get_age_seconds() > max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            # Core identification
            "metric_id": self.metric_id,
            "name": self.name,
            "category": self.category.code,
            "category_description": self.category.description,
            "metric_type": self.metric_type.code,
            "metric_type_description": self.metric_type.description,
            
            # Temporal information
            "timestamp": self.timestamp,
            "formatted_timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "collection_interval": self.collection_interval,
            "retention_period": self.retention_period,
            "age_seconds": self.get_age_seconds(),
            
            # Metric value and metadata
            "value": self.value,
            "unit": self.unit,
            "precision": self.precision,
            "description": self.description,
            
            # Classification
            "level": self.level.name,
            "level_description": self.level.description,
            "status": self.status.code,
            "priority": self.priority,
            
            # Context
            "system_context": self.system_context.to_dict() if self.system_context else None,
            "user_context": self.user_context,
            "request_context": self.request_context,
            
            # Metadata
            "tags": self.tags,
            "labels": self.labels,
            "annotations": self.annotations,
            
            # Quality and validation
            "quality_score": self.quality_score,
            "validation_rules": self.validation_rules,
            "data_source": self.data_source,
            "collection_method": self.collection_method,
            
            # Processing
            "aggregation_window": self.aggregation_window,
            "aggregation_method": self.aggregation_method,
            "derived_metrics": self.derived_metrics,
            
            # Relationships
            "parent_metric_id": self.parent_metric_id,
            "child_metric_ids": self.child_metric_ids,
            "related_metric_ids": self.related_metric_ids,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            
            # Additional fields
            "custom_fields": self.custom_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryMetric':
        """Create TelemetryMetric from dictionary."""
        # Extract system context if present
        system_context = None
        if data.get("system_context"):
            system_context = SystemContext(**data["system_context"])
        
        return cls(
            metric_id=data.get("metric_id", f"metric_{uuid.uuid4().hex[:12]}"),
            name=data.get("name", "unknown_metric"),
            category=TelemetryCategory.from_code(data.get("category", "system.performance")),
            value=data.get("value", 0),
            timestamp=data.get("timestamp", time.time()),
            system_context=system_context,
            tags=data.get("tags", {}),
            # Add other fields as needed
        )


class TelemetryStore:
    """
    High-performance telemetry data storage and retrieval system.
    
    Provides scalable storage with multiple backends, advanced indexing,
    and efficient querying capabilities for enterprise telemetry data.
    """
    
    def __init__(self,
                 max_metrics: int = 1000000,
                 enable_redis: bool = True,
                 redis_config: Optional[Dict[str, Any]] = None,
                 enable_influxdb: bool = False,
                 influxdb_config: Optional[Dict[str, Any]] = None,
                 database_path: str = "telemetry_metrics.db"):
        """
        Initialize the telemetry store.
        
        Args:
            max_metrics: Maximum number of metrics to store in memory
            enable_redis: Whether to use Redis for distributed storage
            redis_config: Redis configuration parameters
            enable_influxdb: Whether to use InfluxDB for time series storage
            influxdb_config: InfluxDB configuration parameters
            database_path: Path to SQLite database for persistence
        """
        self.max_metrics = max_metrics
        self.database_path = database_path
        
        # In-memory storage with indexing
        self.metrics: Dict[str, TelemetryMetric] = {}
        self.metrics_by_category: Dict[TelemetryCategory, List[str]] = defaultdict(list)
        self.metrics_by_level: Dict[TelemetryLevel, List[str]] = defaultdict(list)
        self.metrics_by_name: Dict[str, List[str]] = defaultdict(list)
        self.metrics_by_time: List[Tuple[float, str]] = []  # (timestamp, metric_id)
        self.metrics_by_tags: Dict[str, List[str]] = defaultdict(list)
        self.metrics_by_hostname: Dict[str, List[str]] = defaultdict(list)
        
        # Performance optimization
        self.metric_cache: Dict[str, TelemetryMetric] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_max_size = 10000
        
        # Statistics
        self.stats = {
            "total_metrics": 0,
            "metrics_by_category": defaultdict(int),
            "metrics_by_level": defaultdict(int),
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_operations": 0,
            "query_operations": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        
        # Redis setup
        self.redis_client = None
        if enable_redis and REDIS_AVAILABLE:
            self._setup_redis(redis_config)
        
        # InfluxDB setup
        self.influxdb_client = None
        if enable_influxdb and INFLUXDB_AVAILABLE:
            self._setup_influxdb(influxdb_config)
        
        # Database setup
        self._setup_database()
        
        # Background maintenance
        self._maintenance_thread = None
        self._shutdown_event = threading.Event()
        self._start_maintenance_thread()
    
    def store_metric(self, metric: TelemetryMetric) -> str:
        """Store a telemetry metric with full indexing."""
        with self._write_lock:
            metric_id = metric.metric_id
            
            # Store in memory
            self.metrics[metric_id] = metric
            
            # Update indexes
            self._update_indexes(metric, metric_id)
            
            # Cache the metric
            self._cache_metric(metric_id, metric)
            
            # Store in Redis if enabled
            if self.redis_client:
                try:
                    metric_data = json.dumps(metric.to_dict(), default=str)
                    self.redis_client.hset("telemetry_metrics", metric_id, metric_data)
                    self.redis_client.expire(f"telemetry_metrics:{metric_id}", 604800)  # 7 days
                except Exception as e:
                    logger.error(f"Redis storage failed for metric {metric_id}: {e}")
            
            # Store in InfluxDB if enabled
            if self.influxdb_client:
                try:
                    self._store_metric_influxdb(metric)
                except Exception as e:
                    logger.error(f"InfluxDB storage failed for metric {metric_id}: {e}")
            
            # Store in database
            self._store_metric_db(metric)
            
            # Update statistics
            self.stats["total_metrics"] += 1
            self.stats["metrics_by_category"][metric.category] += 1
            self.stats["metrics_by_level"][metric.level] += 1
            self.stats["storage_operations"] += 1
            
            # Enforce storage limits
            self._enforce_limits()
            
            return metric_id
    
    def get_metric(self, metric_id: str) -> Optional[TelemetryMetric]:
        """Retrieve a telemetry metric by ID."""
        with self._lock:
            # Check cache first
            if metric_id in self.metric_cache:
                self.cache_access_times[metric_id] = time.time()
                self.stats["cache_hits"] += 1
                return self.metric_cache[metric_id]
            
            # Check memory storage
            if metric_id in self.metrics:
                metric = self.metrics[metric_id]
                self._cache_metric(metric_id, metric)
                return metric
            
            # Check Redis
            if self.redis_client:
                try:
                    metric_data = self.redis_client.hget("telemetry_metrics", metric_id)
                    if metric_data:
                        metric_dict = json.loads(metric_data)
                        metric = TelemetryMetric.from_dict(metric_dict)
                        self._cache_metric(metric_id, metric)
                        return metric
                except Exception as e:
                    logger.error(f"Redis retrieval failed for metric {metric_id}: {e}")
            
            # Check database
            metric = self._get_metric_db(metric_id)
            if metric:
                self._cache_metric(metric_id, metric)
                return metric
            
            self.stats["cache_misses"] += 1
            return None
    
    def query_metrics(self, criteria: Dict[str, Any], limit: int = 100) -> List[TelemetryMetric]:
        """Advanced metric search with multiple criteria."""
        with self._lock:
            self.stats["query_operations"] += 1
            
            # Start with appropriate index
            candidate_metric_ids = self._get_candidate_metrics(criteria)
            
            # Apply filters
            results = []
            for metric_id in candidate_metric_ids:
                metric = self.get_metric(metric_id)
                if metric and self._matches_criteria(metric, criteria):
                    results.append(metric)
                    if len(results) >= limit:
                        break
            
            return results
    
    def get_metrics_by_category(self, category: TelemetryCategory, limit: int = 100) -> List[TelemetryMetric]:
        """Get metrics by category."""
        metric_ids = self.metrics_by_category[category][-limit:]
        return [metric for metric in [self.get_metric(mid) for mid in metric_ids] if metric]
    
    def get_metrics_by_name(self, name: str, limit: int = 100) -> List[TelemetryMetric]:
        """Get metrics by name pattern."""
        metric_ids = self.metrics_by_name[name][-limit:]
        return [metric for metric in [self.get_metric(mid) for mid in metric_ids] if metric]
    
    def get_recent_metrics(self, minutes: int = 60, limit: int = 100) -> List[TelemetryMetric]:
        """Get recent metrics within specified time window."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metric_ids = []
        
        for timestamp, metric_id in reversed(self.metrics_by_time):
            if timestamp < cutoff_time:
                break
            recent_metric_ids.append(metric_id)
            if len(recent_metric_ids) >= limit:
                break
        
        return [metric for metric in [self.get_metric(mid) for mid in recent_metric_ids] if metric]
    
    def get_metrics_by_tag(self, tag_key: str, tag_value: Optional[str] = None) -> List[TelemetryMetric]:
        """Get metrics by tag key and optionally tag value."""
        if tag_value:
            tag_filter = f"{tag_key}:{tag_value}"
        else:
            tag_filter = tag_key
        
        metric_ids = self.metrics_by_tags[tag_filter]
        return [metric for metric in [self.get_metric(mid) for mid in metric_ids] if metric]
    
    def _update_indexes(self, metric: TelemetryMetric, metric_id: str) -> None:
        """Update all index structures for a metric."""
        # Category index
        self.metrics_by_category[metric.category].append(metric_id)
        
        # Level index
        self.metrics_by_level[metric.level].append(metric_id)
        
        # Name index
        self.metrics_by_name[metric.name].append(metric_id)
        
        # Time index
        self.metrics_by_time.append((metric.timestamp, metric_id))
        
        # Tags index
        for key, value in metric.tags.items():
            self.metrics_by_tags[f"{key}:{value}"].append(metric_id)
            self.metrics_by_tags[key].append(metric_id)
        
        # Hostname index
        if metric.system_context and metric.system_context.hostname:
            self.metrics_by_hostname[metric.system_context.hostname].append(metric_id)
    
    def _cache_metric(self, metric_id: str, metric: TelemetryMetric) -> None:
        """Cache metric with LRU eviction."""
        self.metric_cache[metric_id] = metric
        self.cache_access_times[metric_id] = time.time()
        
        # Enforce cache size limit
        if len(self.metric_cache) > self.cache_max_size:
            lru_id = min(self.cache_access_times.keys(), 
                        key=lambda k: self.cache_access_times[k])
            del self.metric_cache[lru_id]
            del self.cache_access_times[lru_id]
    
    def _get_candidate_metrics(self, criteria: Dict[str, Any]) -> List[str]:
        """Get candidate metric IDs based on primary search criteria."""
        if "category" in criteria:
            category = TelemetryCategory.from_code(criteria["category"])
            return self.metrics_by_category[category]
        elif "level" in criteria:
            level = TelemetryLevel[criteria["level"].upper()]
            return self.metrics_by_level[level]
        elif "name" in criteria:
            return self.metrics_by_name[criteria["name"]]
        elif "hostname" in criteria:
            return self.metrics_by_hostname[criteria["hostname"]]
        elif "tag" in criteria:
            return self.metrics_by_tags[criteria["tag"]]
        else:
            # Return all metrics if no specific index criteria
            return list(self.metrics.keys())
    
    def _matches_criteria(self, metric: TelemetryMetric, criteria: Dict[str, Any]) -> bool:
        """Check if metric matches search criteria."""
        for key, value in criteria.items():
            if key == "start_time" and metric.timestamp < value:
                return False
            elif key == "end_time" and metric.timestamp > value:
                return False
            elif key == "min_value" and isinstance(metric.value, (int, float)) and metric.value < value:
                return False
            elif key == "max_value" and isinstance(metric.value, (int, float)) and metric.value > value:
                return False
            elif key == "quality_threshold" and metric.quality_score < value:
                return False
        
        return True
    
    def _setup_redis(self, redis_config: Optional[Dict[str, Any]]) -> None:
        """Setup Redis connection."""
        try:
            config = redis_config or {"host": "localhost", "port": 6379, "db": 3}
            self.redis_client = redis.Redis(**config, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for TelemetryStore")
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}")
            self.redis_client = None
    
    def _setup_influxdb(self, influxdb_config: Optional[Dict[str, Any]]) -> None:
        """Setup InfluxDB connection."""
        try:
            config = influxdb_config or {
                "url": "http://localhost:8086",
                "token": "your-token",
                "org": "your-org",
                "bucket": "telemetry"
            }
            self.influxdb_client = influxdb_client.InfluxDBClient(**config)
            logger.info("InfluxDB connection established for TelemetryStore")
        except Exception as e:
            logger.warning(f"InfluxDB setup failed: {e}")
            self.influxdb_client = None
    
    def _setup_database(self) -> None:
        """Setup SQLite database for persistence."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_metrics (
                    metric_id TEXT PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    level TEXT,
                    metric_type TEXT,
                    timestamp REAL,
                    value TEXT,
                    unit TEXT,
                    hostname TEXT,
                    metric_data TEXT,
                    created_at REAL DEFAULT (datetime('now'))
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON telemetry_metrics (name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON telemetry_metrics (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON telemetry_metrics (level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry_metrics (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hostname ON telemetry_metrics (hostname)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def _store_metric_db(self, metric: TelemetryMetric) -> None:
        """Store metric in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            metric_data = json.dumps(metric.to_dict(), default=str)
            
            cursor.execute("""
                INSERT OR REPLACE INTO telemetry_metrics 
                (metric_id, name, category, level, metric_type, timestamp, 
                 value, unit, hostname, metric_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id,
                metric.name,
                metric.category.code,
                metric.level.name,
                metric.metric_type.code,
                metric.timestamp,
                json.dumps(metric.value, default=str),
                metric.unit,
                metric.system_context.hostname if metric.system_context else None,
                metric_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database storage failed for metric {metric.metric_id}: {e}")
    
    def _get_metric_db(self, metric_id: str) -> Optional[TelemetryMetric]:
        """Retrieve metric from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT metric_data FROM telemetry_metrics WHERE metric_id = ?", (metric_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                metric_data = result[0]
                metric_dict = json.loads(metric_data)
                return TelemetryMetric.from_dict(metric_dict)
                
        except Exception as e:
            logger.error(f"Database retrieval failed for metric {metric_id}: {e}")
        
        return None
    
    def _store_metric_influxdb(self, metric: TelemetryMetric) -> None:
        """Store metric in InfluxDB."""
        try:
            if INFLUXDB_AVAILABLE and influxdb_client:
                from influxdb_client import Point  # type: ignore
                from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore
                
                write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
                
                point = Point(metric.name) \
                    .tag("category", metric.category.code) \
                    .tag("level", metric.level.name) \
                    .tag("hostname", metric.system_context.hostname if metric.system_context else "unknown") \
                    .field("value", metric.value) \
                    .time(int(metric.timestamp * 1000000000))  # nanoseconds
                
                # Add custom tags
                for key, value in metric.tags.items():
                    point.tag(key, value)
                
                write_api.write(bucket="telemetry", record=point)
            
        except Exception as e:
            logger.error(f"InfluxDB storage failed for metric {metric.metric_id}: {e}")
    
    def _enforce_limits(self) -> None:
        """Enforce storage limits by removing old metrics."""
        if len(self.metrics) <= self.max_metrics:
            return
        
        # Remove oldest metrics
        metrics_to_remove = len(self.metrics) - self.max_metrics
        oldest_metrics = sorted(self.metrics_by_time)[:metrics_to_remove]
        
        for timestamp, metric_id in oldest_metrics:
            self._remove_metric(metric_id)
    
    def _remove_metric(self, metric_id: str) -> None:
        """Remove metric from all storage structures."""
        if metric_id not in self.metrics:
            return
        
        metric = self.metrics[metric_id]
        
        # Remove from memory
        del self.metrics[metric_id]
        
        # Remove from cache
        if metric_id in self.metric_cache:
            del self.metric_cache[metric_id]
        if metric_id in self.cache_access_times:
            del self.cache_access_times[metric_id]
        
        # Remove from indexes
        self._remove_from_indexes(metric, metric_id)
    
    def _remove_from_indexes(self, metric: TelemetryMetric, metric_id: str) -> None:
        """Remove metric from all index structures."""
        # Helper function to safely remove from index
        def safe_remove(index_list, metric_id):
            if metric_id in index_list:
                index_list.remove(metric_id)
        
        # Remove from all indexes
        safe_remove(self.metrics_by_category[metric.category], metric_id)
        safe_remove(self.metrics_by_level[metric.level], metric_id)
        safe_remove(self.metrics_by_name[metric.name], metric_id)
        
        for key, value in metric.tags.items():
            safe_remove(self.metrics_by_tags[f"{key}:{value}"], metric_id)
            safe_remove(self.metrics_by_tags[key], metric_id)
        
        if metric.system_context and metric.system_context.hostname:
            safe_remove(self.metrics_by_hostname[metric.system_context.hostname], metric_id)
        
        # Remove from time index
        self.metrics_by_time = [(t, mid) for t, mid in self.metrics_by_time if mid != metric_id]
    
    def _start_maintenance_thread(self) -> None:
        """Start background maintenance thread."""
        def maintenance_worker():
            while not self._shutdown_event.wait(3600):  # Run every hour
                try:
                    # Cleanup old metrics
                    self._cleanup_old_metrics()
                    
                    # Optimize indexes
                    self._optimize_indexes()
                    
                except Exception as e:
                    logger.error(f"Maintenance thread error: {e}")
        
        self._maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_thread.start()
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        retention_days = 30  # Default retention period
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        old_metrics = []
        for timestamp, metric_id in self.metrics_by_time:
            if timestamp < cutoff_time:
                old_metrics.append(metric_id)
            else:
                break  # Metrics are sorted by time
        
        for metric_id in old_metrics:
            self._remove_metric(metric_id)
        
        if old_metrics:
            logger.info(f"Cleaned up {len(old_metrics)} old telemetry metrics")
    
    def _optimize_indexes(self) -> None:
        """Optimize index structures for performance."""
        # Remove empty index entries
        for index_dict in [self.metrics_by_category, self.metrics_by_level, 
                          self.metrics_by_name, self.metrics_by_tags, self.metrics_by_hostname]:
            empty_keys = [k for k, v in index_dict.items() if not v]
            for k in empty_keys:
                del index_dict[k]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the telemetry store."""
        with self._lock:
            return {
                "storage_stats": {
                    "total_metrics": len(self.metrics),
                    "max_metrics": self.max_metrics,
                    "cache_size": len(self.metric_cache),
                    "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                },
                "category_distribution": dict(self.stats["metrics_by_category"]),
                "level_distribution": dict(self.stats["metrics_by_level"]),
                "performance_stats": self.stats,
                "configuration": {
                    "redis_enabled": self.redis_client is not None,
                    "influxdb_enabled": self.influxdb_client is not None,
                    "database_path": self.database_path
                }
            }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the telemetry store."""
        self._shutdown_event.set()
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass
        
        if self.influxdb_client:
            try:
                self.influxdb_client.close()
            except Exception:
                pass


class TelemetryAnalyzer:
    """
    Advanced telemetry data analysis and insights generation system.
    
    Provides comprehensive analytics including trend analysis, anomaly detection,
    performance insights, capacity planning, and predictive analytics.
    """
    
    def __init__(self, store: TelemetryStore):
        """
        Initialize the telemetry analyzer.
        
        Args:
            store: TelemetryStore instance for data access
        """
        self.store = store
        
        # Analysis configuration
        self.analysis_config = {
            "anomaly_threshold": 2.5,  # Standard deviations
            "trend_window_hours": 24,
            "correlation_threshold": 0.7,
            "performance_baseline_days": 7,
            "capacity_planning_days": 90
        }
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Anomaly detection models
        self.anomaly_models: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "analyses_performed": 0,
            "anomalies_detected": 0,
            "trends_identified": 0,
            "correlations_found": 0
        }
    
    def analyze_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends for a specific metric over time."""
        cache_key = f"trends_{metric_name}_{hours}"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get metrics for analysis
            end_time = time.time()
            start_time = end_time - (hours * 3600)
            
            criteria = {
                "name": metric_name,
                "start_time": start_time,
                "end_time": end_time
            }
            
            metrics = self.store.query_metrics(criteria, limit=10000)
            
            if len(metrics) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Extract values and timestamps
            data_points = []
            for metric in sorted(metrics, key=lambda m: m.timestamp):
                if isinstance(metric.value, (int, float)):
                    data_points.append((metric.timestamp, metric.value))
            
            if len(data_points) < 2:
                return {"error": "Insufficient numeric data for trend analysis"}
            
            # Perform trend analysis
            result = self._calculate_trend_analysis(data_points, metric_name)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            if result.get("trend_strength", 0) > 0.5:
                self.stats["trends_identified"] += 1
            
            return result
    
    def detect_anomalies(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Detect anomalies in metric data using statistical analysis."""
        cache_key = f"anomalies_{metric_name}_{hours}"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get historical baseline data
            baseline_end = time.time() - (hours * 3600)
            baseline_start = baseline_end - (7 * 24 * 3600)  # 7 days baseline
            
            baseline_criteria = {
                "name": metric_name,
                "start_time": baseline_start,
                "end_time": baseline_end
            }
            
            baseline_metrics = self.store.query_metrics(baseline_criteria, limit=10000)
            
            # Get recent data for anomaly detection
            recent_end = time.time()
            recent_start = recent_end - (hours * 3600)
            
            recent_criteria = {
                "name": metric_name,
                "start_time": recent_start,
                "end_time": recent_end
            }
            
            recent_metrics = self.store.query_metrics(recent_criteria, limit=1000)
            
            if len(baseline_metrics) < 10:
                return {"error": "Insufficient baseline data for anomaly detection"}
            
            # Perform anomaly detection
            result = self._detect_statistical_anomalies(baseline_metrics, recent_metrics, metric_name)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            if result.get("anomalies"):
                self.stats["anomalies_detected"] += len(result["anomalies"])
            
            return result
    
    def analyze_correlations(self, metrics: List[str], hours: int = 24) -> Dict[str, Any]:
        """Analyze correlations between multiple metrics."""
        cache_key = f"correlations_{'_'.join(sorted(metrics))}_{hours}"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get data for all metrics
            end_time = time.time()
            start_time = end_time - (hours * 3600)
            
            metric_data = {}
            for metric_name in metrics:
                criteria = {
                    "name": metric_name,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                metric_list = self.store.query_metrics(criteria, limit=5000)
                metric_data[metric_name] = [(m.timestamp, m.value) for m in metric_list 
                                          if isinstance(m.value, (int, float))]
            
            # Calculate correlations
            result = self._calculate_correlations(metric_data)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            strong_correlations = [c for c in result.get("correlations", []) 
                                 if abs(c.get("coefficient", 0)) > self.analysis_config["correlation_threshold"]]
            if strong_correlations:
                self.stats["correlations_found"] += len(strong_correlations)
            
            return result
    
    def performance_insights(self, category: TelemetryCategory, hours: int = 24) -> Dict[str, Any]:
        """Generate performance insights for a specific category."""
        cache_key = f"performance_{category.code}_{hours}"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get metrics for category
            metrics = self.store.get_metrics_by_category(category, limit=5000)
            
            if not metrics:
                return {"error": f"No metrics found for category {category.name}"}
            
            # Generate insights
            result = self._generate_performance_insights(metrics, category)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
    
    def capacity_planning(self, metric_name: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate capacity planning forecasts."""
        cache_key = f"capacity_{metric_name}_{forecast_days}"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get historical data for forecasting
            end_time = time.time()
            start_time = end_time - (self.analysis_config["capacity_planning_days"] * 24 * 3600)
            
            criteria = {
                "name": metric_name,
                "start_time": start_time,
                "end_time": end_time
            }
            
            metrics = self.store.query_metrics(criteria, limit=10000)
            
            if len(metrics) < 50:
                return {"error": "Insufficient historical data for capacity planning"}
            
            # Generate forecast
            result = self._generate_capacity_forecast(metrics, metric_name, forecast_days)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
    
    def system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        cache_key = "system_health"
        
        # Check cache
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        with self._lock:
            self.stats["analyses_performed"] += 1
            
            # Get recent metrics for health calculation
            recent_metrics = self.store.get_recent_metrics(minutes=60, limit=1000)
            
            if not recent_metrics:
                return {"error": "No recent metrics available for health assessment"}
            
            # Calculate health score
            result = self._calculate_system_health(recent_metrics)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
    
    def _calculate_trend_analysis(self, data_points: List[Tuple[float, float]], metric_name: str) -> Dict[str, Any]:
        """Calculate comprehensive trend analysis."""
        try:
            # Extract timestamps and values
            timestamps = [dp[0] for dp in data_points]
            values = [dp[1] for dp in data_points]
            
            if not values:
                return {"error": "No valid data points"}
            
            # Basic statistics
            mean_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)
            
            # Calculate variance and standard deviation
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            std_dev = variance ** 0.5
            
            # Linear regression for trend
            n = len(data_points)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in data_points)
            sum_x2 = sum(t * t for t in timestamps)
            
            # Calculate slope and intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                slope = 0
                intercept = mean_value
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n
            
            # Calculate correlation coefficient (trend strength)
            if std_dev == 0:
                correlation = 0
            else:
                time_mean = sum_x / n
                time_std = ((sum_x2 / n) - (time_mean ** 2)) ** 0.5
                if time_std == 0:
                    correlation = 0
                else:
                    correlation = sum((t - time_mean) * (v - mean_value) for t, v in data_points) / (n * time_std * std_dev)
            
            # Determine trend direction and strength
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            trend_strength = abs(correlation)
            
            # Calculate rate of change
            time_span = max(timestamps) - min(timestamps)
            rate_of_change = slope * time_span if time_span > 0 else 0
            
            return {
                "metric_name": metric_name,
                "data_points": len(data_points),
                "time_span_hours": time_span / 3600,
                "statistics": {
                    "mean": mean_value,
                    "min": min_value,
                    "max": max_value,
                    "std_dev": std_dev,
                    "variance": variance
                },
                "trend": {
                    "direction": trend_direction,
                    "strength": trend_strength,
                    "slope": slope,
                    "intercept": intercept,
                    "correlation": correlation,
                    "rate_of_change": rate_of_change
                },
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {metric_name}: {e}")
            return {"error": str(e)}
    
    def _detect_statistical_anomalies(self, baseline_metrics: List[TelemetryMetric], 
                                    recent_metrics: List[TelemetryMetric], 
                                    metric_name: str) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        try:
            # Extract baseline values
            baseline_values = [m.value for m in baseline_metrics if isinstance(m.value, (int, float))]
            
            if len(baseline_values) < 10:
                return {"error": "Insufficient baseline data"}
            
            # Calculate baseline statistics
            baseline_mean = sum(baseline_values) / len(baseline_values)
            baseline_variance = sum((v - baseline_mean) ** 2 for v in baseline_values) / len(baseline_values)
            baseline_std = baseline_variance ** 0.5
            
            if baseline_std == 0:
                return {"error": "Zero variance in baseline data"}
            
            # Detect anomalies in recent data
            anomalies = []
            threshold = self.analysis_config["anomaly_threshold"]
            
            for metric in recent_metrics:
                if isinstance(metric.value, (int, float)):
                    z_score = abs(metric.value - baseline_mean) / baseline_std
                    
                    if z_score > threshold:
                        anomaly_severity = "high" if z_score > threshold * 2 else "medium"
                        anomalies.append({
                            "timestamp": metric.timestamp,
                            "value": metric.value,
                            "z_score": z_score,
                            "severity": anomaly_severity,
                            "deviation": metric.value - baseline_mean,
                            "metric_id": metric.metric_id
                        })
            
            # Calculate anomaly rate
            total_recent = len([m for m in recent_metrics if isinstance(m.value, (int, float))])
            anomaly_rate = len(anomalies) / max(1, total_recent)
            
            return {
                "metric_name": metric_name,
                "baseline_period": {
                    "samples": len(baseline_values),
                    "mean": baseline_mean,
                    "std_dev": baseline_std,
                    "min": min(baseline_values),
                    "max": max(baseline_values)
                },
                "analysis_period": {
                    "samples": total_recent,
                    "anomalies_detected": len(anomalies),
                    "anomaly_rate": anomaly_rate
                },
                "anomalies": anomalies,
                "threshold": threshold,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_correlations(self, metric_data: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Calculate correlations between multiple metrics."""
        try:
            correlations = []
            metric_names = list(metric_data.keys())
            
            # Calculate pairwise correlations
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    metric_a = metric_names[i]
                    metric_b = metric_names[j]
                    
                    # Align data points by timestamp
                    aligned_data = self._align_time_series(metric_data[metric_a], metric_data[metric_b])
                    
                    if len(aligned_data) < 3:
                        continue
                    
                    values_a = [point[0] for point in aligned_data]
                    values_b = [point[1] for point in aligned_data]
                    
                    # Calculate Pearson correlation
                    correlation = self._calculate_pearson_correlation(values_a, values_b)
                    
                    if correlation is not None:
                        strength = "strong" if abs(correlation) > 0.8 else "moderate" if abs(correlation) > 0.5 else "weak"
                        
                        correlations.append({
                            "metric_a": metric_a,
                            "metric_b": metric_b,
                            "coefficient": correlation,
                            "strength": strength,
                            "data_points": len(aligned_data)
                        })
            
            return {
                "correlations": correlations,
                "metrics_analyzed": len(metric_names),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_performance_insights(self, metrics: List[TelemetryMetric], 
                                     category: TelemetryCategory) -> Dict[str, Any]:
        """Generate comprehensive performance insights."""
        try:
            insights = []
            metric_groups = defaultdict(list)
            
            # Group metrics by name
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            # Analyze each metric group
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 2:
                    continue
                
                values = [m.value for m in metric_list if isinstance(m.value, (int, float))]
                if not values:
                    continue
                
                # Calculate performance statistics
                mean_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
                
                # Calculate percentiles
                sorted_values = sorted(values)
                p50 = sorted_values[len(sorted_values) // 2]
                p95 = sorted_values[int(len(sorted_values) * 0.95)]
                p99 = sorted_values[int(len(sorted_values) * 0.99)]
                
                # Detect performance issues
                issues = []
                if max_value > mean_value * 3:
                    issues.append("High variability detected")
                if p95 > mean_value * 2:
                    issues.append("95th percentile significantly elevated")
                
                insights.append({
                    "metric_name": metric_name,
                    "sample_count": len(values),
                    "statistics": {
                        "mean": mean_value,
                        "min": min_value,
                        "max": max_value,
                        "p50": p50,
                        "p95": p95,
                        "p99": p99
                    },
                    "issues": issues
                })
            
            # Generate category-level insights
            category_insights = {
                "total_metrics": len(metric_groups),
                "total_data_points": len(metrics),
                "time_span": {
                    "start": min(m.timestamp for m in metrics),
                    "end": max(m.timestamp for m in metrics)
                },
                "metric_insights": insights
            }
            
            return {
                "category": category.name,
                "insights": category_insights,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_capacity_forecast(self, metrics: List[TelemetryMetric], 
                                  metric_name: str, forecast_days: int) -> Dict[str, Any]:
        """Generate capacity planning forecast."""
        try:
            # Extract time series data
            data_points = [(m.timestamp, m.value) for m in metrics 
                          if isinstance(m.value, (int, float))]
            
            if len(data_points) < 10:
                return {"error": "Insufficient data for forecasting"}
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x[0])
            
            # Calculate trend
            trend_analysis = self._calculate_trend_analysis(data_points, metric_name)
            
            if "error" in trend_analysis:
                return trend_analysis
            
            # Generate forecast
            last_timestamp = data_points[-1][0]
            last_value = data_points[-1][1]
            slope = trend_analysis["trend"]["slope"]
            
            forecast_points = []
            for day in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + (day * 24 * 3600)
                forecast_value = last_value + (slope * day * 24 * 3600)
                forecast_points.append({
                    "timestamp": future_timestamp,
                    "forecast_value": forecast_value,
                    "day": day
                })
            
            # Calculate capacity thresholds
            current_max = max(point[1] for point in data_points)
            forecast_max = max(point["forecast_value"] for point in forecast_points)
            
            # Generate recommendations
            recommendations = []
            if forecast_max > current_max * 1.5:
                recommendations.append("Significant growth expected - consider capacity expansion")
            if slope > 0 and forecast_max > current_max * 2:
                recommendations.append("High growth rate detected - urgent capacity planning needed")
            
            return {
                "metric_name": metric_name,
                "forecast_period_days": forecast_days,
                "historical_data": {
                    "data_points": len(data_points),
                    "current_value": last_value,
                    "current_max": current_max
                },
                "trend_analysis": trend_analysis["trend"],
                "forecast": forecast_points,
                "capacity_analysis": {
                    "forecast_max": forecast_max,
                    "growth_rate": slope,
                    "capacity_factor": forecast_max / current_max if current_max > 0 else 0
                },
                "recommendations": recommendations,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Capacity forecasting failed for {metric_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_system_health(self, metrics: List[TelemetryMetric]) -> Dict[str, Any]:
        """Calculate overall system health score."""
        try:
            health_scores = {}
            category_metrics = defaultdict(list)
            
            # Group metrics by category
            for metric in metrics:
                category_metrics[metric.category].append(metric)
            
            # Calculate health score for each category
            for category, metric_list in category_metrics.items():
                # Count error/warning metrics
                error_count = sum(1 for m in metric_list if m.level in [TelemetryLevel.ERROR, TelemetryLevel.CRITICAL])
                warning_count = sum(1 for m in metric_list if m.level == TelemetryLevel.WARNING)
                total_count = len(metric_list)
                
                # Calculate health score (0-100)
                error_ratio = error_count / max(1, total_count)
                warning_ratio = warning_count / max(1, total_count)
                
                category_health = 100 - (error_ratio * 50 + warning_ratio * 25)
                health_scores[category.name] = max(0, category_health)
            
            # Calculate overall health score
            if health_scores:
                overall_health = sum(health_scores.values()) / len(health_scores)
            else:
                overall_health = 100  # No data means healthy
            
            # Determine health status
            if overall_health >= 90:
                health_status = "excellent"
            elif overall_health >= 75:
                health_status = "good"
            elif overall_health >= 50:
                health_status = "fair"
            elif overall_health >= 25:
                health_status = "poor"
            else:
                health_status = "critical"
            
            return {
                "overall_health_score": overall_health,
                "health_status": health_status,
                "category_scores": health_scores,
                "metrics_analyzed": len(metrics),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"System health calculation failed: {e}")
            return {"error": str(e)}
    
    def _align_time_series(self, series_a: List[Tuple[float, float]], 
                          series_b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Align two time series for correlation analysis."""
        # Simple alignment: find closest timestamps
        aligned = []
        tolerance = 300  # 5 minutes tolerance
        
        for timestamp_a, value_a in series_a:
            closest_b = None
            min_diff = float('inf')
            
            for timestamp_b, value_b in series_b:
                diff = abs(timestamp_a - timestamp_b)
                if diff < min_diff and diff <= tolerance:
                    min_diff = diff
                    closest_b = value_b
            
            if closest_b is not None:
                aligned.append((value_a, closest_b))
        
        return aligned
    
    def _calculate_pearson_correlation(self, x_values: List[float], y_values: List[float]) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        try:
            n = len(x_values)
            if n < 2 or len(y_values) != n:
                return None
            
            # Calculate means
            mean_x = sum(x_values) / n
            mean_y = sum(y_values) / n
            
            # Calculate correlation
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
            
            sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
            sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
            
            denominator = (sum_sq_x * sum_sq_y) ** 0.5
            
            if denominator == 0:
                return None
            
            return numerator / denominator
            
        except Exception:
            return None
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result if valid."""
        if cache_key in self.analysis_cache:
            if time.time() - self.cache_timestamps[cache_key] < self.cache_ttl:
                return self.analysis_cache[cache_key]
            else:
                # Remove expired cache entry
                del self.analysis_cache[cache_key]
                del self.cache_timestamps[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache analysis result."""
        self.analysis_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        # Enforce cache size limit
        if len(self.analysis_cache) > 100:
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.analysis_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "statistics": self.stats,
            "cache_info": {
                "cached_results": len(self.analysis_cache),
                "cache_ttl": self.cache_ttl
            },
            "configuration": self.analysis_config
        }


class TelemetryCollector:
    """
    Advanced telemetry data collection system with multi-source integration.
    
    Provides intelligent data collection from system metrics, application metrics,
    custom metrics, and external data sources with adaptive sampling and filtering.
    """
    
    def __init__(self, store: TelemetryStore):
        """
        Initialize the telemetry collector.
        
        Args:
            store: TelemetryStore instance for metric storage
        """
        self.store = store
        self.system_context = SystemContext()
        
        # Collection configuration
        self.config = {
            "collection_interval": 60,  # seconds
            "adaptive_sampling": True,
            "max_metrics_per_batch": 1000,
            "enable_system_metrics": True,
            "enable_process_metrics": True,
            "enable_network_metrics": True,
            "enable_disk_metrics": True,
            "metric_filters": [],
            "custom_collectors": {}
        }
        
        # Collection state
        self.collectors = {}
        self.collection_threads = {}
        self.collection_stats = defaultdict(int)
        self.last_collection_times = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Initialize built-in collectors
        self._initialize_system_collectors()
        
        # Start collection threads
        self._start_collection_threads()
    
    def collect_metric(self, name: str, value: Any, 
                      category: TelemetryCategory = TelemetryCategory.APPLICATION_PERFORMANCE,
                      level: TelemetryLevel = TelemetryLevel.INFO,
                      metric_type: MetricType = MetricType.GAUGE,
                      unit: str = "",
                      tags: Optional[Dict[str, str]] = None,
                      context: Optional[Dict[str, Any]] = None) -> str:
        """Collect a single metric."""
        
        metric = TelemetryMetric(
            name=name,
            value=value,
            category=category,
            level=level,
            metric_type=metric_type,
            unit=unit,
            tags=tags or {},
            system_context=self.system_context,
            custom_fields=context or {}
        )
        
        # Apply filters
        if self._passes_filters(metric):
            metric_id = self.store.store_metric(metric)
            self.collection_stats["metrics_collected"] += 1
            return metric_id
        else:
            self.collection_stats["metrics_filtered"] += 1
            return ""
    
    def collect_batch(self, metrics_data: List[Dict[str, Any]]) -> List[str]:
        """Collect multiple metrics in a batch."""
        metric_ids = []
        
        for metric_data in metrics_data:
            try:
                metric = TelemetryMetric(
                    name=metric_data["name"],
                    value=metric_data["value"],
                    category=TelemetryCategory.from_code(metric_data.get("category", "app_perf")),
                    level=TelemetryLevel[metric_data.get("level", "INFO").upper()],
                    metric_type=MetricType.from_code(metric_data.get("type", "gauge")),
                    unit=metric_data.get("unit", ""),
                    tags=metric_data.get("tags", {}),
                    system_context=self.system_context,
                    context=metric_data.get("context", {})
                )
                
                if self._passes_filters(metric):
                    metric_id = self.store.store_metric(metric)
                    metric_ids.append(metric_id)
                    self.collection_stats["metrics_collected"] += 1
                else:
                    self.collection_stats["metrics_filtered"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to collect metric from batch data: {e}")
                self.collection_stats["collection_errors"] += 1
        
        return metric_ids
    
    def start_system_monitoring(self, interval: int = 60) -> None:
        """Start automatic system metric collection."""
        self.config["collection_interval"] = interval
        
        if not self._shutdown_event.is_set():
            self._start_collection_threads()
    
    def stop_system_monitoring(self) -> None:
        """Stop automatic system metric collection."""
        self._shutdown_event.set()
        
        # Wait for collection threads to finish
        for thread in self.collection_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
    
    def add_custom_collector(self, name: str, collector_func: callable, interval: int = 60) -> None:
        """Add a custom metric collector function."""
        self.config["custom_collectors"][name] = {
            "function": collector_func,
            "interval": interval
        }
        
        # Start collection thread for custom collector
        if not self._shutdown_event.is_set():
            self._start_custom_collector_thread(name)
    
    def remove_custom_collector(self, name: str) -> None:
        """Remove a custom metric collector."""
        if name in self.config["custom_collectors"]:
            del self.config["custom_collectors"][name]
        
        if name in self.collection_threads:
            # Thread will stop on next iteration due to missing config
            pass
    
    def add_metric_filter(self, filter_func: callable) -> None:
        """Add a metric filter function."""
        self.config["metric_filters"].append(filter_func)
    
    def _initialize_system_collectors(self) -> None:
        """Initialize built-in system metric collectors."""
        self.collectors = {
            "system_metrics": self._collect_system_metrics,
            "process_metrics": self._collect_process_metrics,
            "network_metrics": self._collect_network_metrics,
            "disk_metrics": self._collect_disk_metrics,
            "memory_metrics": self._collect_memory_metrics,
            "cpu_metrics": self._collect_cpu_metrics
        }
    
    def _start_collection_threads(self) -> None:
        """Start all collection threads."""
        for collector_name, collector_func in self.collectors.items():
            if self._is_collector_enabled(collector_name):
                thread = threading.Thread(
                    target=self._collection_worker,
                    args=(collector_name, collector_func),
                    daemon=True
                )
                thread.start()
                self.collection_threads[collector_name] = thread
        
        # Start custom collector threads
        for collector_name in self.config["custom_collectors"]:
            self._start_custom_collector_thread(collector_name)
    
    def _start_custom_collector_thread(self, collector_name: str) -> None:
        """Start a custom collector thread."""
        if collector_name in self.config["custom_collectors"]:
            collector_config = self.config["custom_collectors"][collector_name]
            thread = threading.Thread(
                target=self._custom_collection_worker,
                args=(collector_name, collector_config),
                daemon=True
            )
            thread.start()
            self.collection_threads[f"custom_{collector_name}"] = thread
    
    def _collection_worker(self, collector_name: str, collector_func: callable) -> None:
        """Worker thread for metric collection."""
        while not self._shutdown_event.wait(self.config["collection_interval"]):
            try:
                if self._is_collector_enabled(collector_name):
                    start_time = time.time()
                    collector_func()
                    collection_time = time.time() - start_time
                    
                    self.last_collection_times[collector_name] = start_time
                    self.collection_stats[f"{collector_name}_collections"] += 1
                    self.collection_stats[f"{collector_name}_collection_time"] = collection_time
                    
            except Exception as e:
                logger.error(f"Collection error in {collector_name}: {e}")
                self.collection_stats[f"{collector_name}_errors"] += 1
    
    def _custom_collection_worker(self, collector_name: str, collector_config: Dict[str, Any]) -> None:
        """Worker thread for custom metric collection."""
        interval = collector_config["interval"]
        collector_func = collector_config["function"]
        
        while not self._shutdown_event.wait(interval):
            try:
                if collector_name in self.config["custom_collectors"]:
                    start_time = time.time()
                    
                    # Call custom collector function
                    result = collector_func()
                    
                    # Process result if it's a list of metrics
                    if isinstance(result, list):
                        self.collect_batch(result)
                    elif isinstance(result, dict):
                        self.collect_batch([result])
                    
                    collection_time = time.time() - start_time
                    
                    self.last_collection_times[f"custom_{collector_name}"] = start_time
                    self.collection_stats[f"custom_{collector_name}_collections"] += 1
                    self.collection_stats[f"custom_{collector_name}_collection_time"] = collection_time
                    
            except Exception as e:
                logger.error(f"Custom collection error in {collector_name}: {e}")
                self.collection_stats[f"custom_{collector_name}_errors"] += 1
    
    def _collect_system_metrics(self) -> None:
        """Collect general system metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # System uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            self.collect_metric(
                "system.uptime",
                uptime,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "seconds"
            )
            
            # System load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.collect_metric(
                    "system.load_avg_1m",
                    load_avg[0],
                    TelemetryCategory.SYSTEM_PERFORMANCE,
                    TelemetryLevel.INFO,
                    MetricType.GAUGE
                )
                self.collect_metric(
                    "system.load_avg_5m",
                    load_avg[1],
                    TelemetryCategory.SYSTEM_PERFORMANCE,
                    TelemetryLevel.INFO,
                    MetricType.GAUGE
                )
                self.collect_metric(
                    "system.load_avg_15m",
                    load_avg[2],
                    TelemetryCategory.SYSTEM_PERFORMANCE,
                    TelemetryLevel.INFO,
                    MetricType.GAUGE
                )
            except AttributeError:
                # getloadavg not available on Windows
                pass
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _collect_process_metrics(self) -> None:
        """Collect current process metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            process = psutil.Process()
            
            # Process CPU usage
            cpu_percent = process.cpu_percent()
            self.collect_metric(
                "process.cpu_percent",
                cpu_percent,
                TelemetryCategory.APPLICATION_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "percent"
            )
            
            # Process memory usage
            memory_info = process.memory_info()
            self.collect_metric(
                "process.memory_rss",
                memory_info.rss,
                TelemetryCategory.APPLICATION_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "process.memory_vms",
                memory_info.vms,
                TelemetryCategory.APPLICATION_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            
            # Process thread count
            num_threads = process.num_threads()
            self.collect_metric(
                "process.thread_count",
                num_threads,
                TelemetryCategory.APPLICATION_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE
            )
            
            # Process file descriptors (Unix-like systems)
            try:
                num_fds = process.num_fds()
                self.collect_metric(
                    "process.file_descriptors",
                    num_fds,
                    TelemetryCategory.APPLICATION_PERFORMANCE,
                    TelemetryLevel.INFO,
                    MetricType.GAUGE
                )
            except AttributeError:
                # num_fds not available on Windows
                pass
            
        except Exception as e:
            logger.error(f"Process metrics collection failed: {e}")
    
    def _collect_network_metrics(self) -> None:
        """Collect network metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters()
            
            self.collect_metric(
                "network.bytes_sent",
                net_io.bytes_sent,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.COUNTER,
                "bytes"
            )
            self.collect_metric(
                "network.bytes_recv",
                net_io.bytes_recv,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.COUNTER,
                "bytes"
            )
            self.collect_metric(
                "network.packets_sent",
                net_io.packets_sent,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.COUNTER
            )
            self.collect_metric(
                "network.packets_recv",
                net_io.packets_recv,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.COUNTER
            )
            
            # Network connection count
            connections = psutil.net_connections()
            connection_count = len(connections)
            
            self.collect_metric(
                "network.connection_count",
                connection_count,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.GAUGE
            )
            
        except Exception as e:
            logger.error(f"Network metrics collection failed: {e}")
    
    def _collect_disk_metrics(self) -> None:
        """Collect disk metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Disk usage for root filesystem
            disk_usage = psutil.disk_usage('/')
            
            self.collect_metric(
                "disk.total_bytes",
                disk_usage.total,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "disk.used_bytes",
                disk_usage.used,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "disk.free_bytes",
                disk_usage.free,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            
            # Disk usage percentage
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            self.collect_metric(
                "disk.usage_percent",
                usage_percent,
                TelemetryCategory.INFRASTRUCTURE,
                TelemetryLevel.WARNING if usage_percent > 80 else TelemetryLevel.INFO,
                MetricType.GAUGE,
                "percent"
            )
            
            # Disk I/O statistics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.collect_metric(
                    "disk.read_bytes",
                    disk_io.read_bytes,
                    TelemetryCategory.INFRASTRUCTURE,
                    TelemetryLevel.INFO,
                    MetricType.COUNTER,
                    "bytes"
                )
                self.collect_metric(
                    "disk.write_bytes",
                    disk_io.write_bytes,
                    TelemetryCategory.INFRASTRUCTURE,
                    TelemetryLevel.INFO,
                    MetricType.COUNTER,
                    "bytes"
                )
            
        except Exception as e:
            logger.error(f"Disk metrics collection failed: {e}")
    
    def _collect_memory_metrics(self) -> None:
        """Collect memory metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Virtual memory
            memory = psutil.virtual_memory()
            
            self.collect_metric(
                "memory.total_bytes",
                memory.total,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "memory.available_bytes",
                memory.available,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "memory.used_bytes",
                memory.used,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "memory.usage_percent",
                memory.percent,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.WARNING if memory.percent > 85 else TelemetryLevel.INFO,
                MetricType.GAUGE,
                "percent"
            )
            
            # Swap memory
            swap = psutil.swap_memory()
            
            self.collect_metric(
                "swap.total_bytes",
                swap.total,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "swap.used_bytes",
                swap.used,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE,
                "bytes"
            )
            self.collect_metric(
                "swap.usage_percent",
                swap.percent,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.WARNING if swap.percent > 50 else TelemetryLevel.INFO,
                MetricType.GAUGE,
                "percent"
            )
            
        except Exception as e:
            logger.error(f"Memory metrics collection failed: {e}")
    
    def _collect_cpu_metrics(self) -> None:
        """Collect CPU metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.collect_metric(
                "cpu.usage_percent",
                cpu_percent,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.WARNING if cpu_percent > 80 else TelemetryLevel.INFO,
                MetricType.GAUGE,
                "percent"
            )
            
            # CPU count
            cpu_count = psutil.cpu_count()
            self.collect_metric(
                "cpu.count",
                cpu_count,
                TelemetryCategory.SYSTEM_PERFORMANCE,
                TelemetryLevel.INFO,
                MetricType.GAUGE
            )
            
            # CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    self.collect_metric(
                        "cpu.frequency_mhz",
                        cpu_freq.current,
                        TelemetryCategory.SYSTEM_PERFORMANCE,
                        TelemetryLevel.INFO,
                        MetricType.GAUGE,
                        "MHz"
                    )
            except AttributeError:
                # cpu_freq not available on all platforms
                pass
            
        except Exception as e:
            logger.error(f"CPU metrics collection failed: {e}")
    
    def _is_collector_enabled(self, collector_name: str) -> bool:
        """Check if a collector is enabled."""
        config_key = f"enable_{collector_name}"
        return self.config.get(config_key, True)
    
    def _passes_filters(self, metric: TelemetryMetric) -> bool:
        """Check if metric passes all configured filters."""
        for filter_func in self.config["metric_filters"]:
            try:
                if not filter_func(metric):
                    return False
            except Exception as e:
                logger.error(f"Metric filter error: {e}")
        
        return True
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "collection_stats": dict(self.collection_stats),
            "active_collectors": list(self.collectors.keys()),
            "custom_collectors": list(self.config["custom_collectors"].keys()),
            "last_collection_times": self.last_collection_times,
            "configuration": {
                "collection_interval": self.config["collection_interval"],
                "adaptive_sampling": self.config["adaptive_sampling"],
                "max_metrics_per_batch": self.config["max_metrics_per_batch"]
            }
        }


class TelemetryReporter:
    """
    Advanced telemetry reporting and visualization system.
    
    Provides comprehensive reporting capabilities including dashboards,
    alerts, exports, and automated report generation.
    """
    
    def __init__(self, store: TelemetryStore, analyzer: TelemetryAnalyzer):
        """
        Initialize the telemetry reporter.
        
        Args:
            store: TelemetryStore instance for data access
            analyzer: TelemetryAnalyzer instance for analytics
        """
        self.store = store
        self.analyzer = analyzer
        
        # Reporting configuration
        self.config = {
            "report_formats": ["json", "html", "csv"],
            "dashboard_refresh_interval": 300,  # 5 minutes
            "alert_thresholds": {},
            "export_directory": "telemetry_reports",
            "automated_reports": {}
        }
        
        # Report templates
        self.report_templates = {
            "system_health": self._generate_system_health_report,
            "performance_summary": self._generate_performance_report,
            "capacity_planning": self._generate_capacity_report,
            "anomaly_detection": self._generate_anomaly_report,
            "trend_analysis": self._generate_trend_report
        }
        
        # Alert state
        self.alert_history = []
        self.active_alerts = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create reports directory
        os.makedirs(self.config["export_directory"], exist_ok=True)
    
    def generate_dashboard(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive telemetry dashboard."""
        dashboard_data = {
            "timestamp": time.time(),
            "timeframe_hours": timeframe_hours,
            "system_overview": {},
            "performance_metrics": {},
            "alerts": {},
            "capacity_status": {}
        }
        
        try:
            # System health overview
            health_score = self.analyzer.system_health_score()
            dashboard_data["system_overview"] = health_score
            
            # Recent metrics summary
            recent_metrics = self.store.get_recent_metrics(minutes=timeframe_hours * 60, limit=1000)
            
            # Group metrics by category
            category_stats = defaultdict(list)
            for metric in recent_metrics:
                category_stats[metric.category.name].append(metric)
            
            # Generate category summaries
            for category, metrics in category_stats.items():
                if metrics:
                    numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
                    if numeric_values:
                        dashboard_data["performance_metrics"][category] = {
                            "count": len(metrics),
                            "mean": sum(numeric_values) / len(numeric_values),
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "latest": numeric_values[-1] if numeric_values else None
                        }
            
            # Check for active alerts
            alert_summary = self._check_alert_conditions()
            dashboard_data["alerts"] = alert_summary
            
            # Capacity indicators
            capacity_metrics = ["memory.usage_percent", "cpu.usage_percent", "disk.usage_percent"]
            capacity_status = {}
            
            for metric_name in capacity_metrics:
                recent_metric = self._get_latest_metric(metric_name)
                if recent_metric:
                    capacity_status[metric_name] = {
                        "current_value": recent_metric.value,
                        "timestamp": recent_metric.timestamp,
                        "status": self._get_capacity_status(metric_name, recent_metric.value)
                    }
            
            dashboard_data["capacity_status"] = capacity_status
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            dashboard_data["error"] = str(e)
        
        return dashboard_data
    
    def generate_report(self, report_type: str, **kwargs) -> Dict[str, Any]:
        """Generate a specific type of report."""
        if report_type not in self.report_templates:
            return {"error": f"Unknown report type: {report_type}"}
        
        try:
            report_generator = self.report_templates[report_type]
            report = report_generator(**kwargs)
            
            # Add metadata
            report["report_metadata"] = {
                "report_type": report_type,
                "generated_at": time.time(),
                "generated_by": "TelemetryReporter",
                "parameters": kwargs
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed for {report_type}: {e}")
            return {"error": str(e)}
    
    def export_report(self, report: Dict[str, Any], filename: str, format: str = "json") -> str:
        """Export report to file."""
        try:
            filepath = os.path.join(self.config["export_directory"], filename)
            
            if format == "json":
                with open(f"{filepath}.json", 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            elif format == "csv":
                # Convert report to CSV format
                csv_data = self._convert_to_csv(report)
                with open(f"{filepath}.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_data)
                    
            elif format == "html":
                html_content = self._convert_to_html(report)
                with open(f"{filepath}.html", 'w') as f:
                    f.write(html_content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return f"{filepath}.{format}"
            
        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return ""
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions and return active alerts."""
        alerts = []
        
        try:
            alert_summary = self._check_alert_conditions()
            alerts = alert_summary.get("active_alerts", [])
            
            # Update alert history
            for alert in alerts:
                alert_id = alert.get("id")
                if alert_id and alert_id not in self.active_alerts:
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append({
                        "alert": alert,
                        "triggered_at": time.time(),
                        "status": "triggered"
                    })
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
        
        return alerts
    
    def _generate_system_health_report(self, **kwargs) -> Dict[str, Any]:
        """Generate system health report."""
        try:
            health_data = self.analyzer.system_health_score()
            
            # Add recent system metrics
            system_metrics = self.store.get_metrics_by_category(
                TelemetryCategory.SYSTEM_PERFORMANCE, 
                limit=100
            )
            
            # Calculate system trends
            trends = {}
            for metric_name in ["cpu.usage_percent", "memory.usage_percent", "disk.usage_percent"]:
                trend_analysis = self.analyzer.analyze_trends(metric_name, hours=24)
                if "error" not in trend_analysis:
                    trends[metric_name] = trend_analysis["trend"]
            
            return {
                "health_score": health_data,
                "system_metrics_summary": {
                    "total_metrics": len(system_metrics),
                    "time_range": {
                        "start": min(m.timestamp for m in system_metrics) if system_metrics else None,
                        "end": max(m.timestamp for m in system_metrics) if system_metrics else None
                    }
                },
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"System health report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_performance_report(self, category: str = "APPLICATION_PERFORMANCE", **kwargs) -> Dict[str, Any]:
        """Generate performance report."""
        try:
            category_enum = TelemetryCategory.from_code(category.lower())
            insights = self.analyzer.performance_insights(category_enum, hours=kwargs.get("hours", 24))
            
            return {
                "performance_insights": insights,
                "category": category
            }
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_capacity_report(self, metrics: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate capacity planning report."""
        try:
            if not metrics:
                metrics = ["memory.usage_percent", "cpu.usage_percent", "disk.usage_percent"]
            
            forecasts = {}
            for metric_name in metrics:
                forecast = self.analyzer.capacity_planning(
                    metric_name, 
                    forecast_days=kwargs.get("forecast_days", 30)
                )
                if "error" not in forecast:
                    forecasts[metric_name] = forecast
            
            return {
                "capacity_forecasts": forecasts,
                "forecast_parameters": {
                    "forecast_days": kwargs.get("forecast_days", 30),
                    "metrics_analyzed": len(forecasts)
                }
            }
            
        except Exception as e:
            logger.error(f"Capacity report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_anomaly_report(self, metrics: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate anomaly detection report."""
        try:
            if not metrics:
                # Get most active metrics
                recent_metrics = self.store.get_recent_metrics(minutes=1440, limit=1000)  # 24 hours
                metric_counts = defaultdict(int)
                for metric in recent_metrics:
                    metric_counts[metric.name] += 1
                
                # Select top metrics for anomaly detection
                metrics = [name for name, count in sorted(metric_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:10]]
            
            anomaly_results = {}
            for metric_name in metrics:
                anomalies = self.analyzer.detect_anomalies(
                    metric_name, 
                    hours=kwargs.get("hours", 24)
                )
                if "error" not in anomalies:
                    anomaly_results[metric_name] = anomalies
            
            return {
                "anomaly_detection": anomaly_results,
                "metrics_analyzed": len(anomaly_results)
            }
            
        except Exception as e:
            logger.error(f"Anomaly report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_trend_report(self, metrics: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate trend analysis report."""
        try:
            if not metrics:
                # Get most active metrics
                recent_metrics = self.store.get_recent_metrics(minutes=1440, limit=1000)  # 24 hours
                metric_counts = defaultdict(int)
                for metric in recent_metrics:
                    metric_counts[metric.name] += 1
                
                # Select top metrics for trend analysis
                metrics = [name for name, count in sorted(metric_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:10]]
            
            trend_results = {}
            for metric_name in metrics:
                trends = self.analyzer.analyze_trends(
                    metric_name, 
                    hours=kwargs.get("hours", 24)
                )
                if "error" not in trends:
                    trend_results[metric_name] = trends
            
            return {
                "trend_analysis": trend_results,
                "metrics_analyzed": len(trend_results)
            }
            
        except Exception as e:
            logger.error(f"Trend report generation failed: {e}")
            return {"error": str(e)}
    
    def _check_alert_conditions(self) -> Dict[str, Any]:
        """Check all configured alert conditions."""
        active_alerts = []
        
        try:
            # Check capacity thresholds
            capacity_alerts = self._check_capacity_alerts()
            active_alerts.extend(capacity_alerts)
            
            # Check performance thresholds
            performance_alerts = self._check_performance_alerts()
            active_alerts.extend(performance_alerts)
            
            # Check anomaly alerts
            anomaly_alerts = self._check_anomaly_alerts()
            active_alerts.extend(anomaly_alerts)
            
        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")
        
        return {
            "active_alerts": active_alerts,
            "total_alerts": len(active_alerts),
            "checked_at": time.time()
        }
    
    def _check_capacity_alerts(self) -> List[Dict[str, Any]]:
        """Check capacity-related alert conditions."""
        alerts = []
        capacity_metrics = ["memory.usage_percent", "cpu.usage_percent", "disk.usage_percent"]
        
        for metric_name in capacity_metrics:
            latest_metric = self._get_latest_metric(metric_name)
            if latest_metric and isinstance(latest_metric.value, (int, float)):
                threshold = self.config["alert_thresholds"].get(metric_name, 90)
                
                if latest_metric.value > threshold:
                    alerts.append({
                        "id": f"capacity_{metric_name}_{int(time.time())}",
                        "type": "capacity_threshold",
                        "metric": metric_name,
                        "current_value": latest_metric.value,
                        "threshold": threshold,
                        "severity": "high" if latest_metric.value > threshold * 1.1 else "medium",
                        "message": f"{metric_name} is at {latest_metric.value:.1f}% (threshold: {threshold}%)"
                    })
        
        return alerts
    
    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check performance-related alert conditions."""
        alerts = []
        
        # This is a placeholder for performance-specific alerts
        # You can implement custom performance thresholds here
        
        return alerts
    
    def _check_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Check anomaly-related alert conditions."""
        alerts = []
        
        # This is a placeholder for anomaly-based alerts
        # You can implement anomaly detection alerts here
        
        return alerts
    
    def _get_latest_metric(self, metric_name: str) -> Optional[TelemetryMetric]:
        """Get the latest metric by name."""
        metrics = self.store.get_metrics_by_name(metric_name, limit=1)
        return metrics[0] if metrics else None
    
    def _get_capacity_status(self, metric_name: str, value: float) -> str:
        """Get capacity status based on value."""
        if value > 90:
            return "critical"
        elif value > 80:
            return "warning"
        elif value > 70:
            return "caution"
        else:
            return "normal"
    
    def _convert_to_csv(self, report: Dict[str, Any]) -> List[List[str]]:
        """Convert report to CSV format."""
        csv_data = []
        
        # Add headers
        csv_data.append(["Metric", "Value", "Timestamp", "Category"])
        
        # Extract data from report
        # This is a simplified conversion - you can enhance based on report structure
        for key, value in report.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    csv_data.append([f"{key}.{subkey}", str(subvalue), "", key])
            else:
                csv_data.append([key, str(value), "", "general"])
        
        return csv_data
    
    def _convert_to_html(self, report: Dict[str, Any]) -> str:
        """Convert report to HTML format."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Telemetry Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .section {{ margin: 20px 0; }}
                .alert {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Telemetry Report</h1>
            <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Add report content
        for key, value in report.items():
            html += f"<div class='section'><h3>{key.replace('_', ' ').title()}</h3>"
            if isinstance(value, dict):
                html += "<table><tr><th>Property</th><th>Value</th></tr>"
                for subkey, subvalue in value.items():
                    html += f"<tr><td>{subkey}</td><td>{subvalue}</td></tr>"
                html += "</table>"
            else:
                html += f"<p>{value}</p>"
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def get_reporter_stats(self) -> Dict[str, Any]:
        """Get reporter statistics."""
        return {
            "report_templates": list(self.report_templates.keys()),
            "alert_history_count": len(self.alert_history),
            "active_alerts_count": len(self.active_alerts),
            "configuration": self.config
        }


class TelemetrySystem:
    """
    Main telemetry system orchestrator and interface.
    
    Provides a unified interface to all telemetry components including
    collection, storage, analysis, and reporting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the complete telemetry system.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or {}
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self.system_stats = {
            "start_time": time.time(),
            "metrics_processed": 0,
            "reports_generated": 0,
            "alerts_triggered": 0,
            "errors_encountered": 0
        }
        
        logger.info("TelemetrySystem initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all telemetry system components."""
        try:
            # Initialize storage
            store_config = self.config.get("store", {})
            self.store = TelemetryStore(
                max_metrics=store_config.get("max_metrics", 1000000),
                enable_redis=store_config.get("enable_redis", True),
                redis_config=store_config.get("redis_config"),
                enable_influxdb=store_config.get("enable_influxdb", False),
                influxdb_config=store_config.get("influxdb_config"),
                database_path=store_config.get("database_path", "telemetry_metrics.db")
            )
            
            # Initialize analyzer
            self.analyzer = TelemetryAnalyzer(self.store)
            
            # Initialize collector
            self.collector = TelemetryCollector(self.store)
            
            # Initialize reporter
            self.reporter = TelemetryReporter(self.store, self.analyzer)
            
            logger.info("All telemetry components initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def start(self) -> None:
        """Start the telemetry system."""
        if self.is_running:
            logger.warning("Telemetry system is already running")
            return
        
        try:
            # Start automatic metric collection
            collection_config = self.config.get("collection", {})
            interval = collection_config.get("interval", 60)
            
            self.collector.start_system_monitoring(interval)
            
            self.is_running = True
            self.system_stats["start_time"] = time.time()
            
            logger.info("Telemetry system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start telemetry system: {e}")
            self.system_stats["errors_encountered"] += 1
            raise
    
    def stop(self) -> None:
        """Stop the telemetry system."""
        if not self.is_running:
            logger.warning("Telemetry system is not running")
            return
        
        try:
            # Stop collector
            self.collector.stop_system_monitoring()
            
            # Shutdown store
            self.store.shutdown()
            
            self.is_running = False
            self._shutdown_event.set()
            
            logger.info("Telemetry system stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop telemetry system: {e}")
            self.system_stats["errors_encountered"] += 1
    
    def track_operation(self, operation_name: str, user_id: Optional[str] = None, 
                       **kwargs) -> str:
        """
        Track a single operation (backward compatibility).
        
        Args:
            operation_name: Name of the operation being tracked
            user_id: Optional user identifier
            **kwargs: Additional context data
        
        Returns:
            Metric ID for the tracked operation
        """
        try:
            # Prepare context
            context = kwargs.copy()
            if user_id:
                context["user_id"] = user_id
            
            # Determine metric properties based on operation
            category = TelemetryCategory.APPLICATION_PERFORMANCE
            level = TelemetryLevel.INFO
            metric_type = MetricType.EVENT
            
            # Track the operation
            metric_id = self.collector.collect_metric(
                name=f"operation.{operation_name}",
                value=1,  # Count
                category=category,
                level=level,
                metric_type=metric_type,
                unit="count",
                tags={"operation": operation_name},
                context=context
            )
            
            self.system_stats["metrics_processed"] += 1
            
            return metric_id
            
        except Exception as e:
            logger.error(f"Operation tracking failed for {operation_name}: {e}")
            self.system_stats["errors_encountered"] += 1
            return ""


# Global telemetry system instance
_telemetry_system: Optional[TelemetrySystem] = None


def get_telemetry_system(config: Optional[Dict[str, Any]] = None) -> TelemetrySystem:
    """
    Get or create the global telemetry system instance.
    
    Args:
        config: Configuration for the telemetry system
        
    Returns:
        TelemetrySystem instance
    """
    global _telemetry_system
    
    if _telemetry_system is None:
        _telemetry_system = TelemetrySystem(config)
    
    return _telemetry_system


def track_operation(operation_name: str, user_id: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function to track an operation using the global telemetry system.
    
    Args:
        operation_name: Name of the operation being tracked
        user_id: Optional user identifier
        **kwargs: Additional context data
    
    Returns:
        Metric ID for the tracked operation
    """
    telemetry = get_telemetry_system()
    return telemetry.track_operation(operation_name, user_id, **kwargs)


# Export main classes and functions for module usage
__all__ = [
    'TelemetrySystem',
    'TelemetryStore', 
    'TelemetryCollector',
    'TelemetryAnalyzer',
    'TelemetryReporter',
    'TelemetryMetric',
    'TelemetryLevel',
    'TelemetryCategory',
    'MetricType',
    'SystemContext',
    'get_telemetry_system',
    'track_operation'
]