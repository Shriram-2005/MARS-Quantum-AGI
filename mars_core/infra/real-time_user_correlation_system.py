"""
Enterprise Real-Time User Correlation System for MARS Quantum Platform

This module provides comprehensive real-time correlation tracking and analysis with:
- Advanced temporal quantum field correlation analysis
- Multi-dimensional user behavior correlation
- Real-time event stream processing and clustering
- Predictive pattern mining and behavior forecasting
- Quantum-inspired correlation algorithms
- Enterprise-grade performance optimization
- Advanced anomaly detection and alerting
- Comprehensive audit trails and compliance logging

The system implements sophisticated correlation techniques including:
1. Temporal correlation analysis with quantum field theory principles
2. Behavioral pattern recognition and clustering
3. Predictive analytics for user action forecasting
4. Real-time anomaly detection and alerting
5. Multi-dimensional correlation scoring
6. Advanced session and user tracking
7. Performance optimization with adaptive algorithms
8. Enterprise security and audit compliance

Architecture:
- Event-driven correlation processing
- Distributed correlation clustering
- Temporal pattern mining engine
- Real-time analytics pipeline
- Quantum-inspired correlation algorithms
- Advanced caching and optimization
- Comprehensive monitoring and alerting

"""

import os
import sys
import time
import uuid
import json
import logging
import hashlib
import threading
import asyncio
import random
import math
import statistics
import pickle
import gzip
import base64
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Dict, List, Set, Tuple, Any, Optional, Union, Callable, 
    NamedTuple, Protocol, TypeVar, Generic, Awaitable, Iterator
)
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum, auto
from datetime import datetime, timedelta, timezone

# Optional dependencies with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import scipy.stats as stats
    import scipy.cluster.hierarchy as hierarchy
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure enterprise logging
logger = logging.getLogger("MARS.Correlation")
logger.setLevel(logging.INFO)

# Create formatter if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CorrelationLevel(IntEnum):
    """
    Enumeration of correlation confidence levels.
    
    Higher numeric values indicate stronger correlation confidence.
    Used for prioritizing and filtering correlation results.
    """
    NONE = 0         # No meaningful correlation detected
    WEAK = 25        # Low confidence correlation (< 0.3 score)
    MODERATE = 50    # Medium confidence correlation (0.3-0.6 score)
    STRONG = 75      # High confidence correlation (0.6-0.8 score)
    CERTAIN = 100    # Guaranteed correlation (> 0.8 score)
    
    def __str__(self) -> str:
        """String representation for logging and display."""
        return self.name
    
    @classmethod
    def from_score(cls, score: float) -> 'CorrelationLevel':
        """Convert correlation score to correlation level."""
        if score >= 0.8:
            return cls.CERTAIN
        elif score >= 0.6:
            return cls.STRONG
        elif score >= 0.3:
            return cls.MODERATE
        elif score > 0.0:
            return cls.WEAK
        else:
            return cls.NONE
    
    def to_score_range(self) -> Tuple[float, float]:
        """Get the score range for this correlation level."""
        ranges = {
            self.NONE: (0.0, 0.0),
            self.WEAK: (0.0, 0.3),
            self.MODERATE: (0.3, 0.6),
            self.STRONG: (0.6, 0.8),
            self.CERTAIN: (0.8, 1.0)
        }
        return ranges.get(self, (0.0, 0.0))


class EventType(Enum):
    """
    Comprehensive event type classification for correlation analysis.
    
    Each event type has specific correlation algorithms and weighting factors
    optimized for that type of event.
    """
    # User-initiated events
    USER_ACTION = ("user_action", "Direct user-initiated action")
    USER_INPUT = ("user_input", "User input or interaction")
    USER_NAVIGATION = ("user_navigation", "User navigation between pages/views")
    USER_AUTHENTICATION = ("user_auth", "Authentication-related events")
    USER_PREFERENCE = ("user_pref", "User preference or configuration changes")
    
    # System-generated events
    SYSTEM_RESPONSE = ("system_response", "System response to user action")
    SYSTEM_NOTIFICATION = ("system_notification", "System-generated notification")
    SYSTEM_MAINTENANCE = ("system_maintenance", "System maintenance operations")
    SYSTEM_STARTUP = ("system_startup", "System initialization events")
    SYSTEM_SHUTDOWN = ("system_shutdown", "System shutdown events")
    
    # Error and exception events
    ERROR = ("error", "Error or exception event")
    WARNING = ("warning", "Warning or cautionary event")
    CRITICAL_ERROR = ("critical_error", "Critical system error")
    VALIDATION_ERROR = ("validation_error", "Data validation error")
    TIMEOUT_ERROR = ("timeout_error", "Operation timeout error")
    
    # Performance-related events
    PERFORMANCE = ("performance", "Performance monitoring event")
    LATENCY_MEASUREMENT = ("latency", "Response time measurement")
    THROUGHPUT_MEASUREMENT = ("throughput", "System throughput measurement")
    RESOURCE_USAGE = ("resource_usage", "Resource utilization measurement")
    BOTTLENECK_DETECTION = ("bottleneck", "Performance bottleneck detection")
    
    # Security events
    SECURITY = ("security", "Security-related event")
    SECURITY_VIOLATION = ("security_violation", "Security policy violation")
    ACCESS_GRANTED = ("access_granted", "Access authorization granted")
    ACCESS_DENIED = ("access_denied", "Access authorization denied")
    AUDIT_EVENT = ("audit_event", "Security audit trail event")
    
    # Data and state changes
    STATE_CHANGE = ("state_change", "System state modification")
    DATA_CHANGE = ("data_change", "Data modification or update")
    CONFIGURATION_CHANGE = ("config_change", "Configuration modification")
    SCHEMA_CHANGE = ("schema_change", "Database schema modification")
    DEPLOYMENT_EVENT = ("deployment", "Application deployment event")
    
    # Background and automated processes
    BACKGROUND = ("background", "Background process event")
    SCHEDULED_TASK = ("scheduled_task", "Scheduled task execution")
    BATCH_PROCESS = ("batch_process", "Batch processing event")
    CLEANUP_OPERATION = ("cleanup", "System cleanup operation")
    MONITORING_CHECK = ("monitoring_check", "Automated monitoring check")
    
    # Quantum and advanced events
    QUANTUM = ("quantum", "Quantum computation event")
    QUANTUM_MEASUREMENT = ("quantum_measurement", "Quantum state measurement")
    QUANTUM_ENTANGLEMENT = ("quantum_entanglement", "Quantum entanglement operation")
    QUANTUM_SUPERPOSITION = ("quantum_superposition", "Quantum superposition state")
    QUANTUM_DECOHERENCE = ("quantum_decoherence", "Quantum decoherence event")
    
    # Business logic events
    BUSINESS_RULE = ("business_rule", "Business rule execution")
    WORKFLOW_EVENT = ("workflow", "Workflow execution event")
    DECISION_POINT = ("decision_point", "Automated decision point")
    APPROVAL_EVENT = ("approval", "Approval process event")
    NOTIFICATION_SENT = ("notification_sent", "Notification delivery event")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    def __str__(self) -> str:
        return f"{self.name}({self.code})"
    
    @classmethod
    def from_string(cls, event_type_str: str) -> 'EventType':
        """Convert string representation to EventType."""
        for event_type in cls:
            if event_type.name == event_type_str or event_type.code == event_type_str:
                return event_type
        return cls.USER_ACTION  # Default fallback
    
    def get_correlation_weight(self) -> float:
        """Get correlation weight factor for this event type."""
        weights = {
            # High correlation weight for user events
            self.USER_ACTION: 1.0,
            self.USER_INPUT: 0.9,
            self.USER_NAVIGATION: 0.8,
            self.USER_AUTHENTICATION: 0.95,
            
            # Medium correlation weight for system events
            self.SYSTEM_RESPONSE: 0.7,
            self.SYSTEM_NOTIFICATION: 0.6,
            
            # High correlation weight for errors
            self.ERROR: 0.9,
            self.CRITICAL_ERROR: 1.0,
            self.SECURITY_VIOLATION: 1.0,
            
            # Medium correlation weight for background events
            self.BACKGROUND: 0.4,
            self.SCHEDULED_TASK: 0.3,
            
            # High correlation weight for quantum events
            self.QUANTUM: 0.8,
            self.QUANTUM_ENTANGLEMENT: 0.9
        }
        return weights.get(self, 0.5)  # Default weight


class CorrelationAlgorithm(Enum):
    """
    Available correlation algorithms with different characteristics.
    
    Each algorithm is optimized for specific types of correlation analysis
    and provides different performance vs. accuracy trade-offs.
    """
    TEMPORAL_PROXIMITY = ("temporal", "Time-based proximity correlation")
    CAUSAL_INFERENCE = ("causal", "Causal relationship inference")
    BEHAVIORAL_PATTERN = ("behavioral", "User behavior pattern matching")
    SEMANTIC_SIMILARITY = ("semantic", "Content semantic similarity")
    GRAPH_ANALYSIS = ("graph", "Network graph correlation analysis")
    QUANTUM_ENTANGLEMENT = ("quantum", "Quantum entanglement-inspired correlation")
    STATISTICAL_CORRELATION = ("statistical", "Statistical correlation analysis")
    MACHINE_LEARNING = ("ml", "Machine learning-based correlation")
    FUZZY_LOGIC = ("fuzzy", "Fuzzy logic correlation")
    NEURAL_NETWORK = ("neural", "Neural network correlation")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class ProcessingStatus(Enum):
    """Event processing status tracking."""
    CREATED = ("created", "Event created but not processed")
    QUEUED = ("queued", "Event queued for processing")
    PROCESSING = ("processing", "Event currently being processed")
    PROCESSED = ("processed", "Event successfully processed")
    CORRELATED = ("correlated", "Event correlated with others")
    CLUSTERED = ("clustered", "Event assigned to cluster")
    ARCHIVED = ("archived", "Event archived for long-term storage")
    ERROR = ("error", "Error occurred during processing")
    FAILED = ("failed", "Processing failed permanently")
    EXPIRED = ("expired", "Event expired before processing")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class QuantumState:
    """
    Quantum state representation for quantum-inspired correlation algorithms.
    
    Implements quantum mechanics concepts like superposition and entanglement
    for advanced correlation analysis.
    """
    amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))
    phase: float = 0.0
    entangled_events: Set[str] = field(default_factory=set)
    superposition_states: List[str] = field(default_factory=list)
    measurement_history: List[Tuple[float, str]] = field(default_factory=list)
    coherence_time: float = 300.0  # 5 minutes default coherence
    last_measurement: Optional[float] = None
    
    def __post_init__(self):
        """Initialize quantum state after creation."""
        if self.last_measurement is None:
            self.last_measurement = time.time()
    
    def is_coherent(self) -> bool:
        """Check if quantum state is still coherent."""
        if not self.last_measurement:
            return True
        return (time.time() - self.last_measurement) < self.coherence_time
    
    def measure(self) -> str:
        """Perform quantum measurement, collapsing superposition."""
        current_time = time.time()
        
        if self.superposition_states:
            # Collapse superposition to single state
            measured_state = random.choice(self.superposition_states)
            self.superposition_states = [measured_state]
        else:
            measured_state = "ground_state"
        
        # Record measurement
        self.measurement_history.append((current_time, measured_state))
        self.last_measurement = current_time
        
        # Limit measurement history
        if len(self.measurement_history) > 100:
            self.measurement_history = self.measurement_history[-100:]
        
        return measured_state
    
    def entangle_with(self, event_id: str) -> None:
        """Create entanglement with another event."""
        self.entangled_events.add(event_id)
    
    def decohere(self) -> None:
        """Force quantum decoherence."""
        self.superposition_states.clear()
        self.entangled_events.clear()
        self.phase = 0.0
        self.last_measurement = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "amplitude_real": self.amplitude.real,
            "amplitude_imag": self.amplitude.imag,
            "phase": self.phase,
            "entangled_events": list(self.entangled_events),
            "superposition_states": self.superposition_states,
            "coherence_time": self.coherence_time,
            "last_measurement": self.last_measurement,
            "is_coherent": self.is_coherent(),
            "measurement_count": len(self.measurement_history)
        }


@dataclass
class CorrelationEvent:
    """
    Comprehensive event representation for correlation analysis.
    
    Stores all necessary information for advanced correlation algorithms
    including temporal, spatial, semantic, and quantum properties.
    """
    # Core identification
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    correlation_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex[:8]}")
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    
    # User and session context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Temporal properties
    timestamp: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    timeout_ms: Optional[float] = None
    retry_count: int = 0
    
    # Event classification
    event_type: EventType = EventType.USER_ACTION
    event_category: str = "general"
    event_subcategory: str = "default"
    priority: int = 5  # 1=highest, 10=lowest
    severity: str = "info"  # debug, info, warning, error, critical
    
    # Source and target information
    source: str = "unknown"
    target: str = "unknown"
    source_component: Optional[str] = None
    target_component: Optional[str] = None
    request_path: Optional[str] = None
    
    # Event data and metadata
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Processing information
    processing_status: ProcessingStatus = ProcessingStatus.CREATED
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    retry_attempts: List[float] = field(default_factory=list)
    
    # Correlation properties
    correlation_score: float = 0.0
    correlation_algorithms: Set[CorrelationAlgorithm] = field(default_factory=set)
    correlation_features: Dict[str, float] = field(default_factory=dict)
    
    # Quantum state for quantum-inspired algorithms
    quantum_state: Optional[QuantumState] = None
    
    # Geolocation data
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    network_latency: Optional[float] = None
    disk_io: Optional[float] = None
    
    # Business context
    business_process: Optional[str] = None
    workflow_id: Optional[str] = None
    transaction_id: Optional[str] = None
    cost_center: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        # Ensure correlation_id is set
        if not self.correlation_id:
            self.correlation_id = f"corr_{uuid.uuid4().hex[:8]}"
        
        # Set root_id if not specified
        if not self.root_id:
            self.root_id = self.parent_id or self.event_id
        
        # Initialize quantum state if not provided
        if self.quantum_state is None and EventType.QUANTUM in [self.event_type]:
            self.quantum_state = QuantumState()
        
        # Add automatic tags based on event properties
        self._add_automatic_tags()
    
    def _add_automatic_tags(self) -> None:
        """Add automatic tags based on event properties."""
        # Add event type tag
        self.tags.add(f"type:{self.event_type.code}")
        
        # Add severity tag
        self.tags.add(f"severity:{self.severity}")
        
        # Add user-related tags
        if self.user_id:
            self.tags.add("user_event")
        
        # Add error-related tags
        if "error" in self.event_type.name.lower():
            self.tags.add("error_event")
        
        # Add performance tags
        if self.duration_ms and self.duration_ms > 1000:
            self.tags.add("slow_event")
        
        # Add retry tags
        if self.retry_count > 0:
            self.tags.add("retry_event")
    
    def add_correlation_feature(self, feature_name: str, value: float) -> None:
        """Add a correlation feature for analysis."""
        self.correlation_features[feature_name] = value
    
    def get_age_seconds(self) -> float:
        """Get age of event in seconds."""
        return time.time() - self.timestamp
    
    def is_expired(self, max_age_seconds: float = 3600) -> bool:
        """Check if event has expired."""
        return self.get_age_seconds() > max_age_seconds
    
    def mark_processed(self, processing_time_ms: float = None) -> None:
        """Mark event as successfully processed."""
        self.processing_status = ProcessingStatus.PROCESSED
        if processing_time_ms:
            self.processing_time_ms = processing_time_ms
    
    def mark_error(self, error_message: str) -> None:
        """Mark event as having processing error."""
        self.processing_status = ProcessingStatus.ERROR
        self.error_message = error_message
    
    def calculate_hash(self) -> str:
        """Calculate content hash for deduplication."""
        content = f"{self.event_type.code}:{self.source}:{self.target}:{self.user_id}:{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            # Core identification
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            
            # User and session context
            "user_id": self.user_id,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            
            # Temporal properties
            "timestamp": self.timestamp,
            "formatted_timestamp": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "duration_ms": self.duration_ms,
            "age_seconds": self.get_age_seconds(),
            
            # Event classification
            "event_type": self.event_type.name,
            "event_code": self.event_type.code,
            "event_category": self.event_category,
            "priority": self.priority,
            "severity": self.severity,
            
            # Source and target
            "source": self.source,
            "target": self.target,
            "source_component": self.source_component,
            "target_component": self.target_component,
            
            # Data and metadata
            "data": self.data,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "labels": self.labels,
            
            # Processing information
            "processing_status": self.processing_status.code,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            
            # Correlation properties
            "correlation_score": self.correlation_score,
            "correlation_algorithms": [alg.code for alg in self.correlation_algorithms],
            "correlation_features": self.correlation_features,
            
            # Quantum state
            "quantum_state": self.quantum_state.to_dict() if self.quantum_state else None,
            
            # Location data
            "geolocation": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "country": self.country,
                "region": self.region,
                "city": self.city
            } if any([self.latitude, self.longitude, self.country]) else None,
            
            # Performance metrics
            "performance": {
                "cpu_usage": self.cpu_usage,
                "memory_usage": self.memory_usage,
                "network_latency": self.network_latency,
                "disk_io": self.disk_io
            } if any([self.cpu_usage, self.memory_usage, self.network_latency, self.disk_io]) else None,
            
            # Business context
            "business_context": {
                "business_process": self.business_process,
                "workflow_id": self.workflow_id,
                "transaction_id": self.transaction_id,
                "cost_center": self.cost_center
            } if any([self.business_process, self.workflow_id, self.transaction_id]) else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationEvent':
        """Create CorrelationEvent from dictionary."""
        # Extract quantum state if present
        quantum_state = None
        if data.get("quantum_state"):
            qs_data = data["quantum_state"]
            quantum_state = QuantumState(
                amplitude=complex(qs_data.get("amplitude_real", 1.0), qs_data.get("amplitude_imag", 0.0)),
                phase=qs_data.get("phase", 0.0),
                entangled_events=set(qs_data.get("entangled_events", [])),
                superposition_states=qs_data.get("superposition_states", []),
                coherence_time=qs_data.get("coherence_time", 300.0),
                last_measurement=qs_data.get("last_measurement")
            )
        
        # Extract geolocation if present
        geo = data.get("geolocation", {}) or {}
        
        # Extract performance metrics if present
        perf = data.get("performance", {}) or {}
        
        # Extract business context if present
        business = data.get("business_context", {}) or {}
        
        return cls(
            event_id=data.get("event_id", f"evt_{uuid.uuid4().hex[:12]}"),
            correlation_id=data.get("correlation_id", f"corr_{uuid.uuid4().hex[:8]}"),
            parent_id=data.get("parent_id"),
            root_id=data.get("root_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            device_id=data.get("device_id"),
            ip_address=data.get("ip_address"),
            timestamp=data.get("timestamp", time.time()),
            duration_ms=data.get("duration_ms"),
            event_type=EventType.from_string(data.get("event_type", "USER_ACTION")),
            event_category=data.get("event_category", "general"),
            priority=data.get("priority", 5),
            severity=data.get("severity", "info"),
            source=data.get("source", "unknown"),
            target=data.get("target", "unknown"),
            source_component=data.get("source_component"),
            target_component=data.get("target_component"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
            labels=data.get("labels", {}),
            processing_status=ProcessingStatus(data.get("processing_status", "created")),
            correlation_score=data.get("correlation_score", 0.0),
            correlation_algorithms=set([CorrelationAlgorithm(alg) for alg in data.get("correlation_algorithms", [])]),
            correlation_features=data.get("correlation_features", {}),
            quantum_state=quantum_state,
            latitude=geo.get("latitude"),
            longitude=geo.get("longitude"),
            country=geo.get("country"),
            region=geo.get("region"),
            city=geo.get("city"),
            cpu_usage=perf.get("cpu_usage"),
            memory_usage=perf.get("memory_usage"),
            network_latency=perf.get("network_latency"),
            disk_io=perf.get("disk_io"),
            business_process=business.get("business_process"),
            workflow_id=business.get("workflow_id"),
            transaction_id=business.get("transaction_id"),
            cost_center=business.get("cost_center")
        )
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None
    quantum_state: Optional[Dict] = None
    duration_ms: Optional[float] = None
    priority: int = 0
    processing_status: str = "created"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "formatted_timestamp": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "event_type": self.event_type.name,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "tags": list(self.tags),
            "parent_id": self.parent_id,
            "quantum_state": self.quantum_state,
            "duration_ms": self.duration_ms,
            "priority": self.priority,
            "processing_status": self.processing_status
        }


@dataclass
class CorrelationCluster:
    """
    Advanced correlation cluster for grouping related events.
    
    Implements sophisticated clustering algorithms with quantum-inspired
    correlation analysis and dynamic cluster evolution.
    """
    # Core identification
    cluster_id: str = field(default_factory=lambda: f"cluster_{uuid.uuid4().hex[:12]}")
    cluster_type: str = "temporal"  # temporal, behavioral, semantic, hybrid
    
    # Temporal properties
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0  # 1 hour default TTL
    
    # Cluster composition
    events: List[str] = field(default_factory=list)  # Event IDs in cluster
    event_count: int = 0
    root_event_id: Optional[str] = None  # Primary/triggering event
    
    # Correlation metrics
    correlation_level: CorrelationLevel = CorrelationLevel.WEAK
    correlation_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    statistical_significance: float = 0.0
    
    # User and session tracking
    user_ids: Set[str] = field(default_factory=set)
    session_ids: Set[str] = field(default_factory=set)
    unique_users: int = 0
    unique_sessions: int = 0
    
    # Cluster characteristics
    event_types: Set[EventType] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    targets: Set[str] = field(default_factory=set)
    
    # Temporal analysis
    duration_ms: Optional[float] = None
    time_span_seconds: float = 0.0
    event_frequency: float = 0.0  # Events per second
    peak_activity_time: Optional[float] = None
    
    # Cluster state
    active: bool = True
    frozen: bool = False  # Prevent further modifications
    mergeable: bool = True
    splittable: bool = True
    
    # Quality metrics
    cohesion_score: float = 0.0  # How tightly related events are
    separation_score: float = 0.0  # How distinct from other clusters
    silhouette_score: float = 0.0  # Overall cluster quality
    
    # Metadata and analytics
    metadata: Dict[str, Any] = field(default_factory=dict)
    analytics: Dict[str, float] = field(default_factory=dict)
    pattern_signature: Optional[str] = None
    
    # Prediction and forecasting
    predicted_next_events: List[str] = field(default_factory=list)
    completion_probability: float = 0.0
    anomaly_score: float = 0.0
    
    # Geographic distribution
    geographic_spread: Dict[str, int] = field(default_factory=dict)  # country -> count
    location_entropy: float = 0.0
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        self.event_count = len(self.events)
        self.unique_users = len(self.user_ids)
        self.unique_sessions = len(self.session_ids)
        
        # Set initial pattern signature
        if not self.pattern_signature:
            self.pattern_signature = self._generate_pattern_signature()
    
    def add_event(self, event_id: str, event: CorrelationEvent) -> None:
        """Add an event to the cluster with comprehensive updating."""
        if event_id in self.events:
            return  # Event already in cluster
        
        # Add event to cluster
        self.events.append(event_id)
        self.event_count = len(self.events)
        
        # Update user and session tracking
        if event.user_id:
            self.user_ids.add(event.user_id)
            self.unique_users = len(self.user_ids)
        
        if event.session_id:
            self.session_ids.add(event.session_id)
            self.unique_sessions = len(self.session_ids)
        
        # Update event characteristics
        self.event_types.add(event.event_type)
        self.tags.update(event.tags)
        self.categories.add(event.event_category)
        self.sources.add(event.source)
        self.targets.add(event.target)
        
        # Update geographic tracking
        if event.country:
            self.geographic_spread[event.country] = self.geographic_spread.get(event.country, 0) + 1
            self._update_location_entropy()
        
        # Update temporal properties
        self.updated_at = time.time()
        self.last_activity = max(self.last_activity, event.timestamp)
        
        # Set root event if this is the first or highest priority event
        if not self.root_event_id or event.priority < self._get_root_event_priority():
            self.root_event_id = event_id
        
        # Update time span
        if len(self.events) > 1:
            timestamps = [event.timestamp for event in self._get_events() if event]
            if timestamps:
                self.time_span_seconds = max(timestamps) - min(timestamps)
                self.event_frequency = len(timestamps) / max(1.0, self.time_span_seconds)
        
        # Regenerate pattern signature
        self.pattern_signature = self._generate_pattern_signature()
        
        # Update analytics
        self._update_analytics()
    
    def remove_event(self, event_id: str) -> bool:
        """Remove an event from the cluster."""
        if event_id not in self.events:
            return False
        
        self.events.remove(event_id)
        self.event_count = len(self.events)
        self.updated_at = time.time()
        
        # Regenerate pattern signature
        self.pattern_signature = self._generate_pattern_signature()
        
        # Update analytics
        self._update_analytics()
        
        return True
    
    def merge_with(self, other_cluster: 'CorrelationCluster') -> None:
        """Merge another cluster into this one."""
        if not self.mergeable or not other_cluster.mergeable:
            return
        
        # Merge events (avoiding duplicates)
        for event_id in other_cluster.events:
            if event_id not in self.events:
                self.events.append(event_id)
        
        # Merge user and session data
        self.user_ids.update(other_cluster.user_ids)
        self.session_ids.update(other_cluster.session_ids)
        
        # Merge characteristics
        self.event_types.update(other_cluster.event_types)
        self.tags.update(other_cluster.tags)
        self.categories.update(other_cluster.categories)
        self.sources.update(other_cluster.sources)
        self.targets.update(other_cluster.targets)
        
        # Merge geographic data
        for country, count in other_cluster.geographic_spread.items():
            self.geographic_spread[country] = self.geographic_spread.get(country, 0) + count
        
        # Update temporal properties
        self.created_at = min(self.created_at, other_cluster.created_at)
        self.updated_at = time.time()
        self.last_activity = max(self.last_activity, other_cluster.last_activity)
        
        # Update correlation metrics (weighted average)
        total_events = self.event_count + other_cluster.event_count
        if total_events > 0:
            self.correlation_score = (
                (self.correlation_score * self.event_count +
                 other_cluster.correlation_score * other_cluster.event_count) / total_events
            )
        
        # Update derived fields
        self.event_count = len(self.events)
        self.unique_users = len(self.user_ids)
        self.unique_sessions = len(self.session_ids)
        
        # Regenerate pattern signature
        self.pattern_signature = self._generate_pattern_signature()
        
        # Update analytics
        self._update_analytics()
        self._update_location_entropy()
    
    def is_expired(self) -> bool:
        """Check if cluster has expired based on TTL."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def is_stale(self, max_inactive_seconds: float = 1800) -> bool:
        """Check if cluster is stale (no recent activity)."""
        return (time.time() - self.last_activity) > max_inactive_seconds
    
    def calculate_similarity(self, other_cluster: 'CorrelationCluster') -> float:
        """Calculate similarity score with another cluster."""
        if not other_cluster or other_cluster.cluster_id == self.cluster_id:
            return 0.0
        
        # Event overlap similarity (Jaccard index)
        events1 = set(self.events)
        events2 = set(other_cluster.events)
        event_similarity = len(events1.intersection(events2)) / len(events1.union(events2)) if events1.union(events2) else 0.0
        
        # User overlap similarity
        users1 = self.user_ids
        users2 = other_cluster.user_ids
        user_similarity = len(users1.intersection(users2)) / len(users1.union(users2)) if users1.union(users2) else 0.0
        
        # Event type similarity
        types1 = self.event_types
        types2 = other_cluster.event_types
        type_similarity = len(types1.intersection(types2)) / len(types1.union(types2)) if types1.union(types2) else 0.0
        
        # Temporal similarity
        time_diff = abs(self.created_at - other_cluster.created_at)
        temporal_similarity = max(0.0, 1.0 - time_diff / 3600.0)  # Similar if within 1 hour
        
        # Tag similarity
        tags1 = self.tags
        tags2 = other_cluster.tags
        tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2)) if tags1.union(tags2) else 0.0
        
        # Pattern signature similarity
        pattern_similarity = 0.0
        if self.pattern_signature and other_cluster.pattern_signature:
            pattern_similarity = 1.0 if self.pattern_signature == other_cluster.pattern_signature else 0.0
        
        # Weighted combination
        similarity = (
            event_similarity * 0.3 +
            user_similarity * 0.2 +
            type_similarity * 0.2 +
            temporal_similarity * 0.1 +
            tag_similarity * 0.1 +
            pattern_similarity * 0.1
        )
        
        return min(1.0, similarity)
    
    def _generate_pattern_signature(self) -> str:
        """Generate a pattern signature for the cluster."""
        # Create signature based on event types and sources
        event_types = sorted([et.code for et in self.event_types])
        sources = sorted(list(self.sources))
        
        signature_data = {
            "event_types": event_types,
            "sources": sources,
            "event_count": self.event_count,
            "unique_users": self.unique_users
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    def _get_root_event_priority(self) -> int:
        """Get priority of current root event."""
        # This would require access to event store - simplified implementation
        return 5  # Default priority
    
    def _get_events(self) -> List[CorrelationEvent]:
        """Get actual event objects (requires event store access)."""
        # This would require access to event store - placeholder
        return []
    
    def _update_analytics(self) -> None:
        """Update cluster analytics metrics."""
        self.analytics.update({
            "density": self.event_count / max(1.0, self.time_span_seconds),
            "user_diversity": self.unique_users / max(1, self.event_count),
            "session_diversity": self.unique_sessions / max(1, self.event_count),
            "type_diversity": len(self.event_types) / max(1, self.event_count),
            "geographic_diversity": len(self.geographic_spread),
            "activity_recency": time.time() - self.last_activity
        })
    
    def _update_location_entropy(self) -> None:
        """Update location entropy based on geographic distribution."""
        if not self.geographic_spread:
            self.location_entropy = 0.0
            return
        
        total_events = sum(self.geographic_spread.values())
        if total_events == 0:
            self.location_entropy = 0.0
            return
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in self.geographic_spread.values():
            if count > 0:
                probability = count / total_events
                entropy -= probability * math.log2(probability)
        
        self.location_entropy = entropy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            # Core identification
            "cluster_id": self.cluster_id,
            "cluster_type": self.cluster_type,
            
            # Temporal properties
            "created_at": self.created_at,
            "formatted_created_at": datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "updated_at": self.updated_at,
            "formatted_updated_at": datetime.fromtimestamp(self.updated_at).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "last_activity": self.last_activity,
            "formatted_last_activity": datetime.fromtimestamp(self.last_activity).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "age_seconds": time.time() - self.created_at,
            "ttl_seconds": self.ttl_seconds,
            
            # Cluster composition
            "events": self.events,
            "event_count": self.event_count,
            "root_event_id": self.root_event_id,
            
            # Correlation metrics
            "correlation_level": self.correlation_level.name,
            "correlation_score": self.correlation_score,
            "confidence_interval": self.confidence_interval,
            "statistical_significance": self.statistical_significance,
            
            # User and session tracking
            "user_ids": list(self.user_ids),
            "session_ids": list(self.session_ids),
            "unique_users": self.unique_users,
            "unique_sessions": self.unique_sessions,
            
            # Cluster characteristics
            "event_types": [et.name for et in self.event_types],
            "tags": list(self.tags),
            "categories": list(self.categories),
            "sources": list(self.sources),
            "targets": list(self.targets),
            
            # Temporal analysis
            "duration_ms": self.duration_ms,
            "time_span_seconds": self.time_span_seconds,
            "event_frequency": self.event_frequency,
            "peak_activity_time": self.peak_activity_time,
            
            # Cluster state
            "active": self.active,
            "frozen": self.frozen,
            "mergeable": self.mergeable,
            "splittable": self.splittable,
            
            # Quality metrics
            "cohesion_score": self.cohesion_score,
            "separation_score": self.separation_score,
            "silhouette_score": self.silhouette_score,
            
            # Analytics and metadata
            "analytics": self.analytics,
            "metadata": self.metadata,
            "pattern_signature": self.pattern_signature,
            
            # Prediction and forecasting
            "predicted_next_events": self.predicted_next_events,
            "completion_probability": self.completion_probability,
            "anomaly_score": self.anomaly_score,
            
            # Geographic analysis
            "geographic_spread": self.geographic_spread,
            "location_entropy": self.location_entropy,
            
            # Status checks
            "is_expired": self.is_expired(),
            "is_stale": self.is_stale()
        }

@dataclass
class TemporalPattern:
    """
    Advanced temporal pattern representation for sequence analysis.
    
    Captures complex temporal relationships and enables predictive analysis
    of event sequences with quantum-inspired pattern matching.
    """
    # Core identification
    pattern_id: str = field(default_factory=lambda: f"pattern_{uuid.uuid4().hex[:12]}")
    pattern_type: str = "sequential"  # sequential, concurrent, cyclical, chaotic
    pattern_name: Optional[str] = None
    
    # Temporal properties
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_matched: Optional[float] = None
    
    # Pattern definition
    event_sequence: List[str] = field(default_factory=list)  # Event type codes
    time_intervals: List[float] = field(default_factory=list)  # Intervals between events
    duration_distribution: Dict[str, float] = field(default_factory=dict)  # Statistical distribution
    
    # Pattern statistics
    confidence: float = 0.0
    support: float = 0.0  # Frequency of pattern occurrence
    lift: float = 1.0  # Association strength
    conviction: float = 1.0  # Implication strength
    
    # Occurrence tracking
    occurrences: int = 0
    total_observations: int = 0
    success_rate: float = 0.0
    failure_count: int = 0
    
    # Source information
    source_clusters: List[str] = field(default_factory=list)  # Cluster IDs where pattern found
    source_users: Set[str] = field(default_factory=set)  # Users exhibiting pattern
    source_sessions: Set[str] = field(default_factory=set)  # Sessions with pattern
    
    # Pattern quality
    complexity_score: float = 0.0
    uniqueness_score: float = 0.0
    predictive_power: float = 0.0
    stability_score: float = 0.0
    
    # Variation analysis
    variations: List[Dict[str, Any]] = field(default_factory=list)  # Pattern variations
    tolerance_thresholds: Dict[str, float] = field(default_factory=dict)  # Matching tolerances
    
    # Context and conditions
    preconditions: List[str] = field(default_factory=list)  # Required conditions
    postconditions: List[str] = field(default_factory=list)  # Expected outcomes
    context_tags: Set[str] = field(default_factory=set)  # Contextual tags
    
    # Metadata and analytics
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if not self.pattern_name:
            self.pattern_name = f"Pattern_{self.pattern_id[:8]}"
        
        # Initialize tolerance thresholds if not set
        if not self.tolerance_thresholds:
            self.tolerance_thresholds = {
                "time_tolerance": 0.2,  # 20% time variance allowed
                "sequence_tolerance": 0.1,  # 10% sequence deviation allowed
                "duration_tolerance": 0.3  # 30% duration variance allowed
            }
        
        # Update quality scores
        self._update_quality_scores()
    
    def add_occurrence(self, cluster_id: str, user_id: Optional[str] = None, 
                      session_id: Optional[str] = None, success: bool = True) -> None:
        """Record a new occurrence of this pattern."""
        self.occurrences += 1
        self.total_observations += 1
        self.last_matched = time.time()
        self.updated_at = time.time()
        
        # Track success/failure
        if success:
            self.success_rate = (self.success_rate * (self.occurrences - 1) + 1.0) / self.occurrences
        else:
            self.failure_count += 1
            self.success_rate = (self.success_rate * (self.occurrences - 1)) / max(1, self.occurrences)
        
        # Update source tracking
        if cluster_id not in self.source_clusters:
            self.source_clusters.append(cluster_id)
        
        if user_id:
            self.source_users.add(user_id)
        
        if session_id:
            self.source_sessions.add(session_id)
        
        # Update quality scores
        self._update_quality_scores()
        
        # Update support (frequency in observations)
        self.support = self.occurrences / max(1, self.total_observations)
        
        # Update confidence based on success rate and occurrences
        occurrence_factor = min(1.0, self.occurrences / 10.0)  # Normalize to 10 occurrences
        self.confidence = self.success_rate * occurrence_factor
    
    def matches_sequence(self, event_sequence: List[str], time_intervals: List[float] = None) -> Tuple[bool, float]:
        """Check if given sequence matches this pattern."""
        if not event_sequence or len(event_sequence) != len(self.event_sequence):
            return False, 0.0
        
        # Check sequence match
        sequence_score = self._calculate_sequence_similarity(event_sequence)
        
        # Check time interval match if provided
        time_score = 1.0
        if time_intervals and self.time_intervals:
            time_score = self._calculate_time_similarity(time_intervals)
        
        # Combined match score
        match_score = sequence_score * 0.7 + time_score * 0.3
        
        # Check if it exceeds tolerance thresholds
        sequence_threshold = 1.0 - self.tolerance_thresholds.get("sequence_tolerance", 0.1)
        time_threshold = 1.0 - self.tolerance_thresholds.get("time_tolerance", 0.2)
        
        matches = (sequence_score >= sequence_threshold and 
                  time_score >= time_threshold)
        
        return matches, match_score
    
    def predict_next_event(self, current_position: int) -> Optional[Dict[str, Any]]:
        """Predict the next event in the pattern sequence."""
        if current_position < 0 or current_position >= len(self.event_sequence) - 1:
            return None
        
        next_event_type = self.event_sequence[current_position + 1]
        
        # Predict timing
        expected_delay = None
        if current_position < len(self.time_intervals):
            expected_delay = self.time_intervals[current_position]
        
        # Calculate confidence for this prediction
        position_confidence = 1.0 - (current_position / len(self.event_sequence))
        prediction_confidence = self.confidence * position_confidence * self.predictive_power
        
        return {
            "event_type": next_event_type,
            "expected_delay_seconds": expected_delay,
            "confidence": prediction_confidence,
            "position": current_position + 1,
            "pattern_id": self.pattern_id
        }
    
    def add_variation(self, sequence: List[str], intervals: List[float], 
                     similarity_score: float) -> None:
        """Add a variation of this pattern."""
        variation = {
            "sequence": sequence,
            "intervals": intervals,
            "similarity_score": similarity_score,
            "added_at": time.time(),
            "occurrence_count": 1
        }
        
        # Check if variation already exists
        for existing in self.variations:
            if existing["sequence"] == sequence:
                existing["occurrence_count"] += 1
                existing["similarity_score"] = max(existing["similarity_score"], similarity_score)
                return
        
        self.variations.append(variation)
        
        # Limit number of variations
        if len(self.variations) > 20:
            # Keep top variations by occurrence count and similarity
            self.variations.sort(key=lambda v: v["occurrence_count"] * v["similarity_score"], reverse=True)
            self.variations = self.variations[:20]
        
        self._update_quality_scores()
    
    def _calculate_sequence_similarity(self, other_sequence: List[str]) -> float:
        """Calculate similarity between event sequences."""
        if not other_sequence or not self.event_sequence:
            return 0.0
        
        if len(other_sequence) != len(self.event_sequence):
            return 0.0
        
        # Calculate exact matches
        matches = sum(1 for a, b in zip(self.event_sequence, other_sequence) if a == b)
        return matches / len(self.event_sequence)
    
    def _calculate_time_similarity(self, other_intervals: List[float]) -> float:
        """Calculate similarity between time interval sequences."""
        if not other_intervals or not self.time_intervals:
            return 1.0  # No time constraint
        
        if len(other_intervals) != len(self.time_intervals):
            return 0.0
        
        # Calculate relative differences
        similarities = []
        for expected, actual in zip(self.time_intervals, other_intervals):
            if expected == 0 and actual == 0:
                similarities.append(1.0)
            elif expected == 0 or actual == 0:
                similarities.append(0.0)
            else:
                # Relative difference
                rel_diff = abs(expected - actual) / max(expected, actual)
                similarity = max(0.0, 1.0 - rel_diff)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _update_quality_scores(self) -> None:
        """Update pattern quality metrics."""
        # Complexity score based on sequence length and variations
        self.complexity_score = min(1.0, len(self.event_sequence) / 10.0 + len(self.variations) / 20.0)
        
        # Uniqueness score based on number of unique users/sessions
        user_uniqueness = len(self.source_users) / max(1, self.occurrences)
        session_uniqueness = len(self.source_sessions) / max(1, self.occurrences)
        self.uniqueness_score = (user_uniqueness + session_uniqueness) / 2.0
        
        # Predictive power based on success rate and confidence
        self.predictive_power = self.success_rate * self.confidence
        
        # Stability score based on variations and failure rate
        variation_penalty = min(0.5, len(self.variations) / 40.0)  # Penalty for too many variations
        failure_penalty = self.failure_count / max(1, self.total_observations)
        self.stability_score = max(0.0, 1.0 - variation_penalty - failure_penalty)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            # Core identification
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "pattern_name": self.pattern_name,
            
            # Temporal properties
            "created_at": self.created_at,
            "formatted_created_at": datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "updated_at": self.updated_at,
            "formatted_updated_at": datetime.fromtimestamp(self.updated_at).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "last_matched": self.last_matched,
            "formatted_last_matched": datetime.fromtimestamp(self.last_matched).strftime("%Y-%m-%d %H:%M:%S.%f") if self.last_matched else None,
            "age_seconds": time.time() - self.created_at,
            
            # Pattern definition
            "event_sequence": self.event_sequence,
            "sequence_length": len(self.event_sequence),
            "time_intervals": self.time_intervals,
            "duration_distribution": self.duration_distribution,
            
            # Pattern statistics
            "confidence": self.confidence,
            "support": self.support,
            "lift": self.lift,
            "conviction": self.conviction,
            
            # Occurrence tracking
            "occurrences": self.occurrences,
            "total_observations": self.total_observations,
            "success_rate": self.success_rate,
            "failure_count": self.failure_count,
            
            # Source information
            "source_clusters": self.source_clusters,
            "source_users": list(self.source_users),
            "source_sessions": list(self.source_sessions),
            "unique_users": len(self.source_users),
            "unique_sessions": len(self.source_sessions),
            "unique_clusters": len(self.source_clusters),
            
            # Pattern quality
            "complexity_score": self.complexity_score,
            "uniqueness_score": self.uniqueness_score,
            "predictive_power": self.predictive_power,
            "stability_score": self.stability_score,
            
            # Variation analysis
            "variations": self.variations,
            "variation_count": len(self.variations),
            "tolerance_thresholds": self.tolerance_thresholds,
            
            # Context and conditions
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "context_tags": list(self.context_tags),
            
            # Analytics
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics
        }

class EventStore:
    """
    Enterprise-grade event storage and retrieval system.
    
    Provides high-performance storage with advanced indexing, caching,
    and query capabilities for correlation events.
    """
    
    def __init__(self, 
                 max_events: int = 1000000,
                 enable_redis: bool = True,
                 redis_config: Optional[Dict[str, Any]] = None,
                 enable_persistence: bool = True,
                 persistence_path: Optional[str] = None):
        """
        Initialize the enterprise event store.
        
        Args:
            max_events: Maximum number of events to store in memory
            enable_redis: Whether to use Redis for distributed storage
            redis_config: Redis configuration parameters
            enable_persistence: Whether to enable disk persistence
            persistence_path: Path for persistent storage
        """
        # Core storage
        self.events: Dict[str, CorrelationEvent] = {}
        self.max_events = max_events
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or "event_store.db"
        
        # Advanced indexing structures
        self.event_ids_by_time: List[Tuple[float, str]] = []  # (timestamp, event_id)
        self.events_by_correlation_id: Dict[str, List[str]] = {}
        self.events_by_user: Dict[str, List[str]] = {}
        self.events_by_session: Dict[str, List[str]] = {}
        self.events_by_type: Dict[EventType, List[str]] = {}
        self.events_by_source: Dict[str, List[str]] = {}
        self.events_by_target: Dict[str, List[str]] = {}
        self.events_by_priority: Dict[int, List[str]] = {}
        self.events_by_severity: Dict[str, List[str]] = {}
        self.events_by_status: Dict[ProcessingStatus, List[str]] = {}
        self.events_by_tags: Dict[str, List[str]] = {}
        self.events_by_country: Dict[str, List[str]] = {}
        
        # Hash-based deduplication
        self.event_hashes: Dict[str, str] = {}  # hash -> event_id
        
        # Performance optimization
        self.event_cache: Dict[str, CorrelationEvent] = {}  # LRU cache
        self.cache_max_size = 10000
        self.cache_access_times: Dict[str, float] = {}
        
        # Statistics and monitoring
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "duplicate_events": 0,
            "expired_events": 0,
            "storage_operations": 0,
            "query_operations": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        
        # Redis integration
        self.redis_client = None
        self.enable_redis = enable_redis
        if enable_redis and REDIS_AVAILABLE:
            try:
                redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 0}
                self.redis_client = redis.Redis(**redis_config, decode_responses=True)
                self.redis_client.ping()  # Test connection
                logger.info("Redis connection established for EventStore")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory storage only: {e}")
                self.enable_redis = False
        
        # Background cleanup task
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()
        
        # Load persisted data if available
        if self.enable_persistence:
            self._load_from_persistence()
    
    def store_event(self, event: CorrelationEvent) -> str:
        """
        Store a correlation event with advanced features.
        
        Args:
            event: The correlation event to store
            
        Returns:
            str: The event ID
        """
        with self._write_lock:
            event_id = event.event_id
            
            # Check for duplicates
            event_hash = event.calculate_hash()
            if event_hash in self.event_hashes:
                self.stats["duplicate_events"] += 1
                logger.debug(f"Duplicate event detected: {event_id}")
                return self.event_hashes[event_hash]
            
            # Store event
            self.events[event_id] = event
            self.event_hashes[event_hash] = event_id
            self.event_ids_by_time.append((event.timestamp, event_id))
            
            # Update all indexes
            self._update_indexes(event, event_id)
            
            # Cache the event
            self._cache_event(event_id, event)
            
            # Store in Redis if enabled
            if self.enable_redis and self.redis_client:
                try:
                    self.redis_client.hset(
                        "events",
                        event_id,
                        json.dumps(event.to_dict(), default=str)
                    )
                    # Set expiration for Redis entries
                    self.redis_client.expire(f"events:{event_id}", 3600 * 24)  # 24 hours
                except Exception as e:
                    logger.warning(f"Redis storage failed for event {event_id}: {e}")
            
            # Enforce storage limits
            self._enforce_limits()
            
            # Update statistics
            self.stats["total_stored"] += 1
            self.stats["storage_operations"] += 1
            
            # Persist if enabled
            if self.enable_persistence:
                self._persist_event(event)
            
            return event_id
    
    def get_event(self, event_id: str) -> Optional[CorrelationEvent]:
        """
        Retrieve an event by ID with caching.
        
        Args:
            event_id: The event identifier
            
        Returns:
            CorrelationEvent or None if not found
        """
        with self._lock:
            # Check cache first
            if event_id in self.event_cache:
                self.cache_access_times[event_id] = time.time()
                self.stats["cache_hits"] += 1
                self.stats["total_retrieved"] += 1
                return self.event_cache[event_id]
            
            # Check main storage
            event = self.events.get(event_id)
            if event:
                self._cache_event(event_id, event)
                self.stats["total_retrieved"] += 1
                return event
            
            # Check Redis if enabled
            if self.enable_redis and self.redis_client:
                try:
                    event_data = self.redis_client.hget("events", event_id)
                    if event_data:
                        event_dict = json.loads(event_data)
                        event = CorrelationEvent.from_dict(event_dict)
                        self._cache_event(event_id, event)
                        self.stats["total_retrieved"] += 1
                        return event
                except Exception as e:
                    logger.warning(f"Redis retrieval failed for event {event_id}: {e}")
            
            self.stats["cache_misses"] += 1
            return None
    
    def get_events_by_correlation_id(self, correlation_id: str) -> List[CorrelationEvent]:
        """Get all events with a specific correlation ID."""
        event_ids = self.events_by_correlation_id.get(correlation_id, [])
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_user(self, user_id: str, limit: Optional[int] = None) -> List[CorrelationEvent]:
        """Get events for a specific user with optional limit."""
        event_ids = self.events_by_user.get(user_id, [])
        if limit:
            event_ids = event_ids[-limit:]  # Get most recent
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_session(self, session_id: str) -> List[CorrelationEvent]:
        """Get all events for a specific session."""
        event_ids = self.events_by_session.get(session_id, [])
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_time_range(self, start_time: float, end_time: float, 
                                limit: Optional[int] = None) -> List[CorrelationEvent]:
        """Get events within a time range with optional limit."""
        events = []
        count = 0
        
        # Binary search for efficiency on large datasets
        for timestamp, event_id in reversed(self.event_ids_by_time):
            if timestamp < start_time:
                break
            if start_time <= timestamp <= end_time:
                event = self.get_event(event_id)
                if event:
                    events.append(event)
                    count += 1
                    if limit and count >= limit:
                        break
        
        return events
    
    def get_events_by_type(self, event_type: EventType, limit: Optional[int] = None) -> List[CorrelationEvent]:
        """Get events of a specific type with optional limit."""
        event_ids = self.events_by_type.get(event_type, [])
        if limit:
            event_ids = event_ids[-limit:]  # Get most recent
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_tags(self, tags: List[str], match_all: bool = False) -> List[CorrelationEvent]:
        """Get events matching specified tags."""
        if not tags:
            return []
        
        event_sets = []
        for tag in tags:
            if tag in self.events_by_tags:
                event_sets.append(set(self.events_by_tags[tag]))
        
        if not event_sets:
            return []
        
        if match_all:
            # Intersection of all sets (events must have ALL tags)
            matching_ids = set.intersection(*event_sets)
        else:
            # Union of all sets (events must have ANY tag)
            matching_ids = set.union(*event_sets)
        
        return [event for event in [self.get_event(eid) for eid in matching_ids] if event]
    
    def search_events(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[CorrelationEvent]:
        """
        Advanced event search with multiple criteria.
        
        Args:
            query: Search criteria dictionary
            limit: Maximum number of results to return
            
        Returns:
            List of matching events
        """
        with self._lock:
            self.stats["query_operations"] += 1
            
            # Start with all events or use time range if specified
            start_time = query.get("start_time")
            end_time = query.get("end_time")
            
            if start_time and end_time:
                candidate_events = self.get_events_by_time_range(start_time, end_time)
            elif query.get("event_type"):
                candidate_events = self.get_events_by_type(EventType.from_string(query["event_type"]))
            elif query.get("user_id"):
                candidate_events = self.get_events_by_user(query["user_id"])
            elif query.get("tags"):
                candidate_events = self.get_events_by_tags(query["tags"])
            else:
                candidate_events = list(self.events.values())
            
            # Apply filters
            results = []
            for event in candidate_events:
                if self._matches_query(event, query):
                    results.append(event)
                    if limit and len(results) >= limit:
                        break
            
            return results
    
    def get_recent_events(self, minutes: int = 60, limit: Optional[int] = None) -> List[CorrelationEvent]:
        """Get events from the last N minutes."""
        end_time = time.time()
        start_time = end_time - (minutes * 60)
        return self.get_events_by_time_range(start_time, end_time, limit)
    
    def get_error_events(self, hours: int = 24) -> List[CorrelationEvent]:
        """Get error events from the last N hours."""
        error_events = []
        for severity in ["error", "critical"]:
            if severity in self.events_by_severity:
                event_ids = self.events_by_severity[severity]
                for event_id in event_ids:
                    event = self.get_event(event_id)
                    if event and event.get_age_seconds() < (hours * 3600):
                        error_events.append(event)
        return error_events
    
    def cleanup_expired_events(self, max_age_hours: int = 24) -> int:
        """Remove expired events and return count of removed events."""
        removed_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self._write_lock:
            expired_events = []
            for event_id, event in self.events.items():
                if (current_time - event.timestamp) > max_age_seconds:
                    expired_events.append(event_id)
            
            for event_id in expired_events:
                if self._remove_event(event_id):
                    removed_count += 1
            
            self.stats["expired_events"] += removed_count
        
        return removed_count
    
    def _update_indexes(self, event: CorrelationEvent, event_id: str) -> None:
        """Update all index structures for an event."""
        # Correlation ID index
        if event.correlation_id:
            if event.correlation_id not in self.events_by_correlation_id:
                self.events_by_correlation_id[event.correlation_id] = []
            self.events_by_correlation_id[event.correlation_id].append(event_id)
        
        # User index
        if event.user_id:
            if event.user_id not in self.events_by_user:
                self.events_by_user[event.user_id] = []
            self.events_by_user[event.user_id].append(event_id)
        
        # Session index
        if event.session_id:
            if event.session_id not in self.events_by_session:
                self.events_by_session[event.session_id] = []
            self.events_by_session[event.session_id].append(event_id)
        
        # Type index
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event_id)
        
        # Source and target indexes
        if event.source not in self.events_by_source:
            self.events_by_source[event.source] = []
        self.events_by_source[event.source].append(event_id)
        
        if event.target not in self.events_by_target:
            self.events_by_target[event.target] = []
        self.events_by_target[event.target].append(event_id)
        
        # Priority index
        if event.priority not in self.events_by_priority:
            self.events_by_priority[event.priority] = []
        self.events_by_priority[event.priority].append(event_id)
        
        # Severity index
        if event.severity not in self.events_by_severity:
            self.events_by_severity[event.severity] = []
        self.events_by_severity[event.severity].append(event_id)
        
        # Status index
        if event.processing_status not in self.events_by_status:
            self.events_by_status[event.processing_status] = []
        self.events_by_status[event.processing_status].append(event_id)
        
        # Tags index
        for tag in event.tags:
            if tag not in self.events_by_tags:
                self.events_by_tags[tag] = []
            self.events_by_tags[tag].append(event_id)
        
        # Country index
        if event.country:
            if event.country not in self.events_by_country:
                self.events_by_country[event.country] = []
            self.events_by_country[event.country].append(event_id)
    
    def _cache_event(self, event_id: str, event: CorrelationEvent) -> None:
        """Add event to cache with LRU eviction."""
        self.event_cache[event_id] = event
        self.cache_access_times[event_id] = time.time()
        
        # Enforce cache size limit
        if len(self.event_cache) > self.cache_max_size:
            # Remove least recently used event
            lru_id = min(self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k])
            del self.event_cache[lru_id]
            del self.cache_access_times[lru_id]
    
    def _matches_query(self, event: CorrelationEvent, query: Dict[str, Any]) -> bool:
        """Check if an event matches query criteria."""
        for key, value in query.items():
            if key in ["start_time", "end_time", "limit"]:
                continue
            
            if key == "tags" and isinstance(value, list):
                if not any(tag in event.tags for tag in value):
                    return False
            elif key == "severity_level" and isinstance(value, list):
                if event.severity not in value:
                    return False
            elif key == "priority_range" and isinstance(value, tuple):
                min_priority, max_priority = value
                if not (min_priority <= event.priority <= max_priority):
                    return False
            elif hasattr(event, key):
                attr_value = getattr(event, key)
                if isinstance(value, list):
                    if attr_value not in value:
                        return False
                elif attr_value != value:
                    return False
            elif key in event.data:
                if event.data[key] != value:
                    return False
            else:
                return False
        
        return True
    
    def _enforce_limits(self) -> None:
        """Enforce storage limits by removing oldest events."""
        while len(self.events) > self.max_events:
            if self.event_ids_by_time:
                oldest_timestamp, oldest_id = self.event_ids_by_time[0]
                self._remove_event(oldest_id)
    
    def _remove_event(self, event_id: str) -> bool:
        """Remove an event from all storage structures."""
        if event_id not in self.events:
            return False
        
        event = self.events[event_id]
        
        # Remove from main storage
        del self.events[event_id]
        
        # Remove from cache
        if event_id in self.event_cache:
            del self.event_cache[event_id]
        if event_id in self.cache_access_times:
            del self.cache_access_times[event_id]
        
        # Remove from time index
        self.event_ids_by_time = [(t, eid) for t, eid in self.event_ids_by_time if eid != event_id]
        
        # Remove from hash index
        event_hash = event.calculate_hash()
        if event_hash in self.event_hashes:
            del self.event_hashes[event_hash]
        
        # Remove from all other indexes
        self._remove_from_indexes(event, event_id)
        
        # Remove from Redis
        if self.enable_redis and self.redis_client:
            try:
                self.redis_client.hdel("events", event_id)
            except Exception as e:
                logger.warning(f"Redis deletion failed for event {event_id}: {e}")
        
        return True
    
    def _remove_from_indexes(self, event: CorrelationEvent, event_id: str) -> None:
        """Remove event from all index structures."""
        # Helper function to safely remove from index
        def safe_remove(index_dict, key, event_id):
            if key in index_dict and event_id in index_dict[key]:
                index_dict[key].remove(event_id)
                if not index_dict[key]:  # Remove empty lists
                    del index_dict[key]
        
        # Remove from all indexes
        if event.correlation_id:
            safe_remove(self.events_by_correlation_id, event.correlation_id, event_id)
        if event.user_id:
            safe_remove(self.events_by_user, event.user_id, event_id)
        if event.session_id:
            safe_remove(self.events_by_session, event.session_id, event_id)
        
        safe_remove(self.events_by_type, event.event_type, event_id)
        safe_remove(self.events_by_source, event.source, event_id)
        safe_remove(self.events_by_target, event.target, event_id)
        safe_remove(self.events_by_priority, event.priority, event_id)
        safe_remove(self.events_by_severity, event.severity, event_id)
        safe_remove(self.events_by_status, event.processing_status, event_id)
        
        for tag in event.tags:
            safe_remove(self.events_by_tags, tag, event_id)
        
        if event.country:
            safe_remove(self.events_by_country, event.country, event_id)
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._shutdown_event.wait(300):  # Run every 5 minutes
                try:
                    self.cleanup_expired_events(24)  # Remove events older than 24 hours
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _persist_event(self, event: CorrelationEvent) -> None:
        """Persist event to disk (simplified implementation)."""
        # This would implement disk persistence using SQLite or similar
        pass
    
    def _load_from_persistence(self) -> None:
        """Load events from persistent storage (simplified implementation)."""
        # This would implement loading from disk
        pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the event store."""
        with self._lock:
            return {
                "storage_stats": {
                    "total_events": len(self.events),
                    "max_events": self.max_events,
                    "storage_utilization": len(self.events) / self.max_events,
                    "cache_size": len(self.event_cache),
                    "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                },
                "index_stats": {
                    "unique_correlation_ids": len(self.events_by_correlation_id),
                    "unique_users": len(self.events_by_user),
                    "unique_sessions": len(self.events_by_session),
                    "unique_sources": len(self.events_by_source),
                    "unique_targets": len(self.events_by_target),
                    "unique_tags": len(self.events_by_tags),
                    "unique_countries": len(self.events_by_country)
                },
                "type_distribution": {
                    event_type.name: len(events) 
                    for event_type, events in self.events_by_type.items()
                },
                "severity_distribution": dict(self.events_by_severity),
                "priority_distribution": dict(self.events_by_priority),
                "performance_stats": self.stats,
                "redis_enabled": self.enable_redis,
                "persistence_enabled": self.enable_persistence
            }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the event store."""
        self._shutdown_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        if self.enable_persistence:
            # Persist remaining events
            pass
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass

class ClusterStore:
    """
    Enterprise-grade correlation cluster storage and management system.
    
    Provides advanced clustering algorithms, cluster lifecycle management,
    and high-performance storage with sophisticated indexing capabilities.
    """
    
    def __init__(self, 
                 max_clusters: int = 100000,
                 enable_redis: bool = True,
                 redis_config: Optional[Dict[str, Any]] = None,
                 cluster_ttl_hours: int = 48,
                 auto_merge_similar: bool = True,
                 similarity_threshold: float = 0.8):
        """
        Initialize the enterprise cluster store.
        
        Args:
            max_clusters: Maximum number of clusters to store
            enable_redis: Whether to use Redis for distributed storage
            redis_config: Redis configuration parameters
            cluster_ttl_hours: Default TTL for clusters in hours
            auto_merge_similar: Whether to automatically merge similar clusters
            similarity_threshold: Threshold for automatic cluster merging
        """
        # Core storage
        self.clusters: Dict[str, CorrelationCluster] = {}
        self.max_clusters = max_clusters
        self.cluster_ttl_hours = cluster_ttl_hours
        self.auto_merge_similar = auto_merge_similar
        self.similarity_threshold = similarity_threshold
        
        # Advanced indexing structures
        self.cluster_ids_by_time: List[Tuple[float, str]] = []
        self.clusters_by_event: Dict[str, List[str]] = {}  # event_id -> [cluster_id]
        self.clusters_by_user: Dict[str, List[str]] = {}  # user_id -> [cluster_id]
        self.clusters_by_session: Dict[str, List[str]] = {}  # session_id -> [cluster_id]
        self.clusters_by_level: Dict[CorrelationLevel, List[str]] = {}  # level -> [cluster_id]
        self.clusters_by_type: Dict[str, List[str]] = {}  # cluster_type -> [cluster_id]
        self.clusters_by_source: Dict[str, List[str]] = {}  # source -> [cluster_id]
        self.clusters_by_target: Dict[str, List[str]] = {}  # target -> [cluster_id]
        self.clusters_by_tags: Dict[str, List[str]] = {}  # tag -> [cluster_id]
        self.clusters_by_pattern: Dict[str, List[str]] = {}  # pattern_signature -> [cluster_id]
        self.clusters_by_country: Dict[str, List[str]] = {}  # country -> [cluster_id]
        
        # Cluster state tracking
        self.active_clusters: Set[str] = set()
        self.frozen_clusters: Set[str] = set()
        self.mergeable_clusters: Set[str] = set()
        self.expired_clusters: Set[str] = set()
        
        # Performance optimization
        self.cluster_cache: Dict[str, CorrelationCluster] = {}
        self.cache_max_size = 5000
        self.cache_access_times: Dict[str, float] = {}
        
        # Statistics and monitoring
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "clusters_merged": 0,
            "clusters_split": 0,
            "expired_clusters": 0,
            "storage_operations": 0,
            "query_operations": 0,
            "similarity_calculations": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        
        # Redis integration
        self.redis_client = None
        self.enable_redis = enable_redis
        if enable_redis and REDIS_AVAILABLE:
            try:
                redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 1}
                self.redis_client = redis.Redis(**redis_config, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis connection established for ClusterStore")
            except Exception as e:
                logger.warning(f"Redis connection failed for ClusterStore: {e}")
                self.enable_redis = False
        
        # Background management tasks
        self._management_thread = None
        self._shutdown_event = threading.Event()
        self._start_management_thread()
    
    def store_cluster(self, cluster: CorrelationCluster) -> str:
        """
        Store a correlation cluster with advanced features.
        
        Args:
            cluster: The correlation cluster to store
            
        Returns:
            str: The cluster ID
        """
        with self._write_lock:
            cluster_id = cluster.cluster_id
            
            # Check for similar clusters and merge if enabled
            if self.auto_merge_similar:
                similar_cluster_id = self._find_similar_cluster(cluster)
                if similar_cluster_id:
                    return self._merge_with_existing(cluster, similar_cluster_id)
            
            # Store cluster
            self.clusters[cluster_id] = cluster
            self.cluster_ids_by_time.append((cluster.created_at, cluster_id))
            
            # Update all indexes
            self._update_indexes(cluster, cluster_id)
            
            # Update state tracking
            if cluster.active:
                self.active_clusters.add(cluster_id)
            if cluster.frozen:
                self.frozen_clusters.add(cluster_id)
            if cluster.mergeable:
                self.mergeable_clusters.add(cluster_id)
            
            # Cache the cluster
            self._cache_cluster(cluster_id, cluster)
            
            # Store in Redis if enabled
            if self.enable_redis and self.redis_client:
                try:
                    self.redis_client.hset(
                        "clusters",
                        cluster_id,
                        json.dumps(cluster.to_dict(), default=str)
                    )
                    self.redis_client.expire(f"clusters:{cluster_id}", 
                                           self.cluster_ttl_hours * 3600)
                except Exception as e:
                    logger.warning(f"Redis storage failed for cluster {cluster_id}: {e}")
            
            # Enforce storage limits
            self._enforce_limits()
            
            # Update statistics
            self.stats["total_stored"] += 1
            self.stats["storage_operations"] += 1
            
            return cluster_id
    
    def get_cluster(self, cluster_id: str) -> Optional[CorrelationCluster]:
        """
        Retrieve a cluster by ID with caching.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            CorrelationCluster or None if not found
        """
        with self._lock:
            # Check cache first
            if cluster_id in self.cluster_cache:
                self.cache_access_times[cluster_id] = time.time()
                self.stats["cache_hits"] += 1
                self.stats["total_retrieved"] += 1
                return self.cluster_cache[cluster_id]
            
            # Check main storage
            cluster = self.clusters.get(cluster_id)
            if cluster:
                self._cache_cluster(cluster_id, cluster)
                self.stats["total_retrieved"] += 1
                return cluster
            
            # Check Redis if enabled
            if self.enable_redis and self.redis_client:
                try:
                    cluster_data = self.redis_client.hget("clusters", cluster_id)
                    if cluster_data:
                        cluster_dict = json.loads(cluster_data)
                        # Note: Would need CorrelationCluster.from_dict method
                        # For now, return None to avoid errors
                        pass
                except Exception as e:
                    logger.warning(f"Redis retrieval failed for cluster {cluster_id}: {e}")
            
            self.stats["cache_misses"] += 1
            return None
    
    def get_clusters_by_event(self, event_id: str) -> List[CorrelationCluster]:
        """Get all clusters containing a specific event."""
        cluster_ids = self.clusters_by_event.get(event_id, [])
        return [cluster for cluster in [self.get_cluster(cid) for cid in cluster_ids] if cluster]
    
    def get_clusters_by_user(self, user_id: str, active_only: bool = True) -> List[CorrelationCluster]:
        """Get clusters for a specific user."""
        cluster_ids = self.clusters_by_user.get(user_id, [])
        clusters = [cluster for cluster in [self.get_cluster(cid) for cid in cluster_ids] if cluster]
        
        if active_only:
            clusters = [cluster for cluster in clusters if cluster.active]
        
        return clusters
    
    def get_active_clusters(self, limit: Optional[int] = None) -> List[CorrelationCluster]:
        """Get all active clusters with optional limit."""
        cluster_ids = list(self.active_clusters)
        if limit:
            cluster_ids = cluster_ids[-limit:]  # Get most recent
        return [cluster for cluster in [self.get_cluster(cid) for cid in cluster_ids] if cluster]
    
    def get_clusters_by_level(self, level: CorrelationLevel) -> List[CorrelationCluster]:
        """Get clusters with a specific correlation level."""
        cluster_ids = self.clusters_by_level.get(level, [])
        return [cluster for cluster in [self.get_cluster(cid) for cid in cluster_ids] if cluster]
    
    def get_clusters_by_pattern(self, pattern_signature: str) -> List[CorrelationCluster]:
        """Get clusters with a specific pattern signature."""
        cluster_ids = self.clusters_by_pattern.get(pattern_signature, [])
        return [cluster for cluster in [self.get_cluster(cid) for cid in cluster_ids] if cluster]
    
    def get_clusters_by_tags(self, tags: List[str], match_all: bool = False) -> List[CorrelationCluster]:
        """Get clusters matching specified tags."""
        if not tags:
            return []
        
        cluster_sets = []
        for tag in tags:
            if tag in self.clusters_by_tags:
                cluster_sets.append(set(self.clusters_by_tags[tag]))
        
        if not cluster_sets:
            return []
        
        if match_all:
            matching_ids = set.intersection(*cluster_sets)
        else:
            matching_ids = set.union(*cluster_sets)
        
        return [cluster for cluster in [self.get_cluster(cid) for cid in matching_ids] if cluster]
    
    def search_clusters(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[CorrelationCluster]:
        """
        Advanced cluster search with multiple criteria.
        
        Args:
            query: Search criteria dictionary
            limit: Maximum number of results to return
            
        Returns:
            List of matching clusters
        """
        with self._lock:
            self.stats["query_operations"] += 1
            
            # Start with filtered clusters based on primary criteria
            if query.get("correlation_level"):
                level = CorrelationLevel.from_string(query["correlation_level"])
                candidate_clusters = self.get_clusters_by_level(level)
            elif query.get("user_id"):
                candidate_clusters = self.get_clusters_by_user(query["user_id"])
            elif query.get("tags"):
                candidate_clusters = self.get_clusters_by_tags(query["tags"])
            elif query.get("active_only", True):
                candidate_clusters = self.get_active_clusters()
            else:
                candidate_clusters = list(self.clusters.values())
            
            # Apply additional filters
            results = []
            for cluster in candidate_clusters:
                if self._matches_cluster_query(cluster, query):
                    results.append(cluster)
                    if limit and len(results) >= limit:
                        break
            
            return results
    
    def update_cluster_status(self, cluster_id: str, active: bool) -> bool:
        """Update a cluster's active status."""
        if cluster_id not in self.clusters:
            return False
        
        with self._write_lock:
            cluster = self.clusters[cluster_id]
            cluster.active = active
            cluster.updated_at = time.time()
            
            if active:
                self.active_clusters.add(cluster_id)
            else:
                self.active_clusters.discard(cluster_id)
            
            # Update cache
            self._cache_cluster(cluster_id, cluster)
            
            return True
    
    def merge_clusters(self, cluster_ids: List[str], new_cluster_id: Optional[str] = None) -> Optional[str]:
        """
        Merge multiple clusters into a new cluster.
        
        Args:
            cluster_ids: List of cluster IDs to merge
            new_cluster_id: Optional ID for the new cluster
            
        Returns:
            ID of the merged cluster or None if merge failed
        """
        if not cluster_ids or len(cluster_ids) < 2:
            return None
        
        # Validate all clusters exist and are mergeable
        clusters_to_merge = []
        for cluster_id in cluster_ids:
            cluster = self.get_cluster(cluster_id)
            if not cluster or not cluster.mergeable:
                logger.warning(f"Cannot merge cluster {cluster_id}: not found or not mergeable")
                return None
            clusters_to_merge.append(cluster)
        
        with self._write_lock:
            # Create new merged cluster
            new_cluster = CorrelationCluster(
                cluster_id=new_cluster_id or f"merged_{uuid.uuid4().hex[:12]}",
                cluster_type="merged"
            )
            
            # Merge the first cluster into the new one
            base_cluster = clusters_to_merge[0]
            new_cluster.merge_with(base_cluster)
            
            # Merge remaining clusters
            for cluster in clusters_to_merge[1:]:
                new_cluster.merge_with(cluster)
            
            # Store the new merged cluster
            merged_id = self.store_cluster(new_cluster)
            
            # Deactivate merged clusters
            for cluster_id in cluster_ids:
                self.update_cluster_status(cluster_id, False)
                if cluster_id in self.mergeable_clusters:
                    self.mergeable_clusters.remove(cluster_id)
            
            self.stats["clusters_merged"] += 1
            
            return merged_id
    
    def split_cluster(self, cluster_id: str, split_criteria: Dict[str, Any]) -> List[str]:
        """
        Split a cluster based on specified criteria.
        
        Args:
            cluster_id: ID of cluster to split
            split_criteria: Criteria for splitting
            
        Returns:
            List of new cluster IDs created from the split
        """
        cluster = self.get_cluster(cluster_id)
        if not cluster or not cluster.splittable:
            return []
        
        with self._write_lock:
            # Implementation would depend on split criteria
            # For now, return empty list
            # This is a placeholder for complex cluster splitting logic
            
            self.stats["clusters_split"] += 1
            return []
    
    def cleanup_expired_clusters(self) -> int:
        """Remove expired clusters and return count."""
        removed_count = 0
        current_time = time.time()
        
        with self._write_lock:
            expired_clusters = []
            for cluster_id, cluster in self.clusters.items():
                if cluster.is_expired():
                    expired_clusters.append(cluster_id)
            
            for cluster_id in expired_clusters:
                if self._remove_cluster(cluster_id):
                    removed_count += 1
            
            self.stats["expired_clusters"] += removed_count
        
        return removed_count
    
    def _find_similar_cluster(self, cluster: CorrelationCluster) -> Optional[str]:
        """Find a similar existing cluster for potential merging."""
        if not cluster.pattern_signature:
            return None
        
        # Look for clusters with same pattern signature first
        similar_clusters = self.clusters_by_pattern.get(cluster.pattern_signature, [])
        
        for cluster_id in similar_clusters:
            existing_cluster = self.get_cluster(cluster_id)
            if existing_cluster and existing_cluster.mergeable:
                self.stats["similarity_calculations"] += 1
                similarity = cluster.calculate_similarity(existing_cluster)
                if similarity >= self.similarity_threshold:
                    return cluster_id
        
        return None
    
    def _merge_with_existing(self, new_cluster: CorrelationCluster, existing_id: str) -> str:
        """Merge a new cluster with an existing one."""
        existing_cluster = self.get_cluster(existing_id)
        if not existing_cluster:
            return self.store_cluster(new_cluster)
        
        # Merge new cluster into existing one
        existing_cluster.merge_with(new_cluster)
        
        # Update cache and storage
        self._cache_cluster(existing_id, existing_cluster)
        
        self.stats["clusters_merged"] += 1
        return existing_id
    
    def _update_indexes(self, cluster: CorrelationCluster, cluster_id: str) -> None:
        """Update all index structures for a cluster."""
        # Event index
        for event_id in cluster.events:
            if event_id not in self.clusters_by_event:
                self.clusters_by_event[event_id] = []
            self.clusters_by_event[event_id].append(cluster_id)
        
        # User index
        for user_id in cluster.user_ids:
            if user_id not in self.clusters_by_user:
                self.clusters_by_user[user_id] = []
            self.clusters_by_user[user_id].append(cluster_id)
        
        # Session index
        for session_id in cluster.session_ids:
            if session_id not in self.clusters_by_session:
                self.clusters_by_session[session_id] = []
            self.clusters_by_session[session_id].append(cluster_id)
        
        # Level index
        if cluster.correlation_level not in self.clusters_by_level:
            self.clusters_by_level[cluster.correlation_level] = []
        self.clusters_by_level[cluster.correlation_level].append(cluster_id)
        
        # Type index
        if cluster.cluster_type not in self.clusters_by_type:
            self.clusters_by_type[cluster.cluster_type] = []
        self.clusters_by_type[cluster.cluster_type].append(cluster_id)
        
        # Source and target indexes
        for source in cluster.sources:
            if source not in self.clusters_by_source:
                self.clusters_by_source[source] = []
            self.clusters_by_source[source].append(cluster_id)
        
        for target in cluster.targets:
            if target not in self.clusters_by_target:
                self.clusters_by_target[target] = []
            self.clusters_by_target[target].append(cluster_id)
        
        # Tags index
        for tag in cluster.tags:
            if tag not in self.clusters_by_tags:
                self.clusters_by_tags[tag] = []
            self.clusters_by_tags[tag].append(cluster_id)
        
        # Pattern index
        if cluster.pattern_signature:
            if cluster.pattern_signature not in self.clusters_by_pattern:
                self.clusters_by_pattern[cluster.pattern_signature] = []
            self.clusters_by_pattern[cluster.pattern_signature].append(cluster_id)
        
        # Country index
        for country in cluster.geographic_spread:
            if country not in self.clusters_by_country:
                self.clusters_by_country[country] = []
            self.clusters_by_country[country].append(cluster_id)
    
    def _cache_cluster(self, cluster_id: str, cluster: CorrelationCluster) -> None:
        """Add cluster to cache with LRU eviction."""
        self.cluster_cache[cluster_id] = cluster
        self.cache_access_times[cluster_id] = time.time()
        
        # Enforce cache size limit
        if len(self.cluster_cache) > self.cache_max_size:
            lru_id = min(self.cache_access_times.keys(), 
                        key=lambda k: self.cache_access_times[k])
            del self.cluster_cache[lru_id]
            del self.cache_access_times[lru_id]
    
    def _matches_cluster_query(self, cluster: CorrelationCluster, query: Dict[str, Any]) -> bool:
        """Check if a cluster matches query criteria."""
        for key, value in query.items():
            if key in ["limit", "active_only"]:
                continue
            
            if key == "min_events" and len(cluster.events) < value:
                return False
            elif key == "max_events" and len(cluster.events) > value:
                return False
            elif key == "min_score" and cluster.correlation_score < value:
                return False
            elif key == "max_age_hours":
                age_hours = (time.time() - cluster.created_at) / 3600
                if age_hours > value:
                    return False
            elif key == "tags" and isinstance(value, list):
                if not any(tag in cluster.tags for tag in value):
                    return False
            elif hasattr(cluster, key):
                attr_value = getattr(cluster, key)
                if isinstance(value, list):
                    if attr_value not in value:
                        return False
                elif attr_value != value:
                    return False
        
        return True
    
    def _enforce_limits(self) -> None:
        """Enforce storage limits by removing old inactive clusters."""
        while len(self.clusters) > self.max_clusters:
            # Find oldest inactive cluster
            oldest_inactive = None
            oldest_time = float('inf')
            
            for timestamp, cluster_id in self.cluster_ids_by_time:
                if cluster_id not in self.active_clusters and timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_inactive = cluster_id
            
            if oldest_inactive:
                self._remove_cluster(oldest_inactive)
            else:
                break  # No inactive clusters to remove
    
    def _remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from all storage structures."""
        if cluster_id not in self.clusters:
            return False
        
        cluster = self.clusters[cluster_id]
        
        # Remove from main storage
        del self.clusters[cluster_id]
        
        # Remove from cache
        if cluster_id in self.cluster_cache:
            del self.cluster_cache[cluster_id]
        if cluster_id in self.cache_access_times:
            del self.cache_access_times[cluster_id]
        
        # Remove from state tracking
        self.active_clusters.discard(cluster_id)
        self.frozen_clusters.discard(cluster_id)
        self.mergeable_clusters.discard(cluster_id)
        self.expired_clusters.discard(cluster_id)
        
        # Remove from time index
        self.cluster_ids_by_time = [(t, cid) for t, cid in self.cluster_ids_by_time if cid != cluster_id]
        
        # Remove from all other indexes
        self._remove_from_indexes(cluster, cluster_id)
        
        return True
    
    def _remove_from_indexes(self, cluster: CorrelationCluster, cluster_id: str) -> None:
        """Remove cluster from all index structures."""
        def safe_remove(index_dict, key, cluster_id):
            if key in index_dict and cluster_id in index_dict[key]:
                index_dict[key].remove(cluster_id)
                if not index_dict[key]:
                    del index_dict[key]
        
        # Remove from all indexes
        for event_id in cluster.events:
            safe_remove(self.clusters_by_event, event_id, cluster_id)
        
        for user_id in cluster.user_ids:
            safe_remove(self.clusters_by_user, user_id, cluster_id)
        
        for session_id in cluster.session_ids:
            safe_remove(self.clusters_by_session, session_id, cluster_id)
        
        safe_remove(self.clusters_by_level, cluster.correlation_level, cluster_id)
        safe_remove(self.clusters_by_type, cluster.cluster_type, cluster_id)
        
        for source in cluster.sources:
            safe_remove(self.clusters_by_source, source, cluster_id)
        
        for target in cluster.targets:
            safe_remove(self.clusters_by_target, target, cluster_id)
        
        for tag in cluster.tags:
            safe_remove(self.clusters_by_tags, tag, cluster_id)
        
        if cluster.pattern_signature:
            safe_remove(self.clusters_by_pattern, cluster.pattern_signature, cluster_id)
        
        for country in cluster.geographic_spread:
            safe_remove(self.clusters_by_country, country, cluster_id)
    
    def _start_management_thread(self) -> None:
        """Start background management thread."""
        def management_worker():
            while not self._shutdown_event.wait(600):  # Run every 10 minutes
                try:
                    # Cleanup expired clusters
                    self.cleanup_expired_clusters()
                    
                    # Perform automatic merging if enabled
                    if self.auto_merge_similar:
                        self._perform_automatic_merging()
                        
                except Exception as e:
                    logger.error(f"Error in cluster management thread: {e}")
        
        self._management_thread = threading.Thread(target=management_worker, daemon=True)
        self._management_thread.start()
    
    def _perform_automatic_merging(self) -> None:
        """Perform automatic merging of similar clusters."""
        # This would implement automatic cluster merging logic
        # Based on similarity thresholds and pattern analysis
        pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the cluster store."""
        with self._lock:
            return {
                "storage_stats": {
                    "total_clusters": len(self.clusters),
                    "active_clusters": len(self.active_clusters),
                    "frozen_clusters": len(self.frozen_clusters),
                    "mergeable_clusters": len(self.mergeable_clusters),
                    "expired_clusters": len(self.expired_clusters),
                    "max_clusters": self.max_clusters,
                    "storage_utilization": len(self.clusters) / self.max_clusters,
                    "cache_size": len(self.cluster_cache),
                    "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                },
                "index_stats": {
                    "unique_events": len(self.clusters_by_event),
                    "unique_users": len(self.clusters_by_user),
                    "unique_sessions": len(self.clusters_by_session),
                    "unique_sources": len(self.clusters_by_source),
                    "unique_targets": len(self.clusters_by_target),
                    "unique_tags": len(self.clusters_by_tags),
                    "unique_patterns": len(self.clusters_by_pattern),
                    "unique_countries": len(self.clusters_by_country)
                },
                "correlation_level_distribution": {
                    level.name: len(clusters) 
                    for level, clusters in self.clusters_by_level.items()
                },
                "cluster_type_distribution": dict(self.clusters_by_type),
                "performance_stats": self.stats,
                "configuration": {
                    "cluster_ttl_hours": self.cluster_ttl_hours,
                    "auto_merge_similar": self.auto_merge_similar,
                    "similarity_threshold": self.similarity_threshold,
                    "redis_enabled": self.enable_redis
                }
            }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the cluster store."""
        self._shutdown_event.set()
        if self._management_thread:
            self._management_thread.join(timeout=5)
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass

class PatternStore:
    """Stores and retrieves temporal patterns"""
    
    def __init__(self, max_patterns: int = 1000):
        """Initialize the pattern store"""
        self.patterns: Dict[str, TemporalPattern] = {}
        self.pattern_ids_by_confidence: List[Tuple[float, str]] = []  # (confidence, pattern_id)
        self.max_patterns = max_patterns
        self.patterns_by_event_type: Dict[str, List[str]] = {}  # event_type -> [pattern_id]
        self._lock = threading.RLock()
        
    def store_pattern(self, pattern: TemporalPattern) -> str:
        """Store a temporal pattern"""
        with self._lock:
            pattern_id = pattern.pattern_id
            self.patterns[pattern_id] = pattern
            self.pattern_ids_by_confidence.append((pattern.confidence, pattern_id))
            
            # Update indexes
            for event_type in pattern.event_sequence:
                if event_type not in self.patterns_by_event_type:
                    self.patterns_by_event_type[event_type] = []
                self.patterns_by_event_type[event_type].append(pattern_id)
                
            # Enforce maximum patterns limit
            self._enforce_max_patterns()
            
            return pattern_id
    
    def get_pattern(self, pattern_id: str) -> Optional[TemporalPattern]:
        """Get a pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_event_type(self, event_type: str) -> List[TemporalPattern]:
        """Get all patterns that include a specific event type"""
        pattern_ids = self.patterns_by_event_type.get(event_type, [])
        return [self.patterns[pattern_id] for pattern_id in pattern_ids if pattern_id in self.patterns]
    
    def get_patterns_by_confidence(self, min_confidence: float = 0.0) -> List[TemporalPattern]:
        """Get patterns with confidence above threshold, ordered by confidence"""
        patterns = []
        for confidence, pattern_id in sorted(self.pattern_ids_by_confidence, reverse=True):
            if confidence >= min_confidence and pattern_id in self.patterns:
                patterns.append(self.patterns[pattern_id])
        return patterns
    
    def update_pattern_confidence(self, pattern_id: str, confidence: float) -> bool:
        """Update a pattern's confidence score"""
        if pattern_id not in self.patterns:
            return False
            
        # Update confidence
        old_confidence = self.patterns[pattern_id].confidence
        self.patterns[pattern_id].confidence = confidence
        
        # Update index
        for i, (conf, pid) in enumerate(self.pattern_ids_by_confidence):
            if pid == pattern_id:
                self.pattern_ids_by_confidence[i] = (confidence, pattern_id)
                break
                
        return True
    
    def increment_pattern_occurrence(self, pattern_id: str) -> bool:
        """Increment a pattern's occurrence count"""
        if pattern_id not in self.patterns:
            return False
            
        self.patterns[pattern_id].occurrences += 1
        return True
    
    def find_matching_pattern(self, event_sequence: List[str], time_intervals: List[float], 
                            similarity_threshold: float = 0.8) -> Optional[str]:
        """Find a pattern that matches the given sequence and intervals"""
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern in self.patterns.items():
            # Skip if sequences are of different lengths
            if len(pattern.event_sequence) != len(event_sequence):
                continue
                
            # Check sequence similarity
            sequence_match = all(a == b for a, b in zip(pattern.event_sequence, event_sequence))
            
            if sequence_match:
                # Check time interval similarity
                interval_similarity = self._calculate_interval_similarity(
                    pattern.time_intervals, time_intervals
                )
                
                overall_similarity = interval_similarity
                
                if overall_similarity > similarity_threshold and overall_similarity > best_similarity:
                    best_match = pattern_id
                    best_similarity = overall_similarity
                    
        return best_match
    
    def _calculate_interval_similarity(self, intervals1: List[float], intervals2: List[float]) -> float:
        """Calculate similarity between two sequences of time intervals"""
        if not intervals1 or not intervals2 or len(intervals1) != len(intervals2):
            return 0.0
            
        similarities = []
        for t1, t2 in zip(intervals1, intervals2):
            # Calculate similarity based on relative difference
            max_time = max(t1, t2)
            if max_time > 0:
                rel_diff = abs(t1 - t2) / max_time
                # Convert to similarity (1.0 = identical, 0.0 = completely different)
                similarity = max(0.0, 1.0 - min(1.0, rel_diff))
                similarities.append(similarity)
            else:
                # Both intervals are 0, consider them identical
                similarities.append(1.0)
                
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def _enforce_max_patterns(self) -> None:
        """Enforce the maximum number of patterns by removing lowest confidence patterns"""
        while len(self.patterns) > self.max_patterns:
            # Find lowest confidence pattern
            lowest_confidence, lowest_id = min(self.pattern_ids_by_confidence)
            
            # Remove from indexes
            if lowest_id in self.patterns:
                pattern = self.patterns[lowest_id]
                
                for event_type in pattern.event_sequence:
                    if event_type in self.patterns_by_event_type and lowest_id in self.patterns_by_event_type[event_type]:
                        self.patterns_by_event_type[event_type].remove(lowest_id)
                
                # Remove from main storage
                del self.patterns[lowest_id]
                
                # Remove from confidence index
                self.pattern_ids_by_confidence.remove((lowest_confidence, lowest_id))
    
    def get_stats(self) -> Dict:
        """Get statistics about the pattern store"""
        return {
            "total_patterns": len(self.patterns),
            "patterns_by_confidence": {
                "high": len([p for p in self.patterns.values() if p.confidence >= 0.7]),
                "medium": len([p for p in self.patterns.values() if 0.4 <= p.confidence < 0.7]),
                "low": len([p for p in self.patterns.values() if p.confidence < 0.4])
            },
            "average_occurrences": sum(p.occurrences for p in self.patterns.values()) / max(1, len(self.patterns)),
            "unique_event_types": len(self.patterns_by_event_type)
        }

class TemporalCorrelationEngine:
    """Identifies temporal correlations between events"""
    
    def __init__(self, event_store: EventStore):
        """Initialize the temporal correlation engine"""
        self.event_store = event_store
        self.temporal_window = 300.0  # 5 minutes in seconds
        self.max_sequence_length = 10  # Maximum events in a sequence
        self.min_pattern_confidence = 0.4  # Minimum confidence for pattern recognition
        self.similarity_threshold = 0.7  # Minimum similarity for pattern matching
        
    def find_correlated_events(self, event_id: str, 
                             window: Optional[float] = None) -> List[Tuple[str, float, CorrelationLevel]]:
        """Find events correlated with the given event"""
        event = self.event_store.get_event(event_id)
        if not event:
            return []
            
        # Use provided window or default
        if window is None:
            window = self.temporal_window
            
        # Define time range
        start_time = max(0, event.timestamp - window/2)
        end_time = event.timestamp + window/2
        
        # Get events in the time window
        candidate_events = self.event_store.get_events_by_time_range(start_time, end_time)
        
        # Calculate correlation scores
        correlated_events = []
        for candidate in candidate_events:
            if candidate.event_id == event_id:
                continue  # Skip self
                
            # Calculate correlation score and level
            score, level = self._calculate_correlation(event, candidate)
            
            # Only include if meaningful correlation
            if score > 0.1:
                correlated_events.append((candidate.event_id, score, level))
                
        # Sort by correlation score (descending)
        correlated_events.sort(key=lambda x: x[1], reverse=True)
        
        return correlated_events
    
    def find_event_sequence(self, start_event_id: str, max_length: int = None) -> List[str]:
        """Find a temporal sequence starting from the given event"""
        if max_length is None:
            max_length = self.max_sequence_length
            
        event = self.event_store.get_event(start_event_id)
        if not event:
            return []
            
        # Start sequence with the given event
        sequence = [start_event_id]
        current_event = event
        
        # Repeatedly find the next most likely event
        for _ in range(max_length - 1):
            # Find correlated events
            correlated = self.find_correlated_events(current_event.event_id)
            
            # Filter to only include events after the current one
            future_events = [
                (event_id, score, level) for event_id, score, level in correlated
                if self.event_store.get_event(event_id).timestamp > current_event.timestamp
                and event_id not in sequence
            ]
            
            if not future_events:
                break  # No more future events
                
            # Select the most strongly correlated future event
            next_event_id = max(future_events, key=lambda x: x[1])[0]
            sequence.append(next_event_id)
            current_event = self.event_store.get_event(next_event_id)
            
        return sequence
    
    def _calculate_correlation(self, event1: CorrelationEvent, event2: CorrelationEvent) -> Tuple[float, CorrelationLevel]:
        """Calculate correlation score and level between two events"""
        score = 0.0
        
        # Correlation by explicit correlation_id
        if event1.correlation_id and event1.correlation_id == event2.correlation_id:
            return 1.0, CorrelationLevel.CERTAIN
            
        # Correlation by parent-child relationship
        if event1.event_id == event2.parent_id or event2.event_id == event1.parent_id:
            return 0.9, CorrelationLevel.CERTAIN
            
        # Correlation by user and session
        if event1.user_id and event1.user_id == event2.user_id:
            score += 0.4
            
        if event1.session_id and event1.session_id == event2.session_id:
            score += 0.4
            
        # Correlation by source/target
        if event1.source == event2.source or event1.target == event2.target:
            score += 0.2
            
        if event1.source == event2.target or event1.target == event2.source:
            score += 0.3
            
        # Correlation by temporal proximity
        time_diff = abs(event1.timestamp - event2.timestamp)
        temporal_score = max(0.0, 1.0 - time_diff / self.temporal_window)
        score += temporal_score * 0.3
        
        # Correlation by shared tags
        common_tags = event1.tags.intersection(event2.tags)
        tag_score = len(common_tags) / max(1, len(event1.tags.union(event2.tags)))
        score += tag_score * 0.2
        
        # Determine correlation level
        level = CorrelationLevel.WEAK
        if score >= 0.8:
            level = CorrelationLevel.STRONG
        elif score >= 0.5:
            level = CorrelationLevel.MODERATE
            
        return min(1.0, score), level

class ClusteringEngine:
    """Identifies clusters of correlated events"""
    
    def __init__(self, event_store: EventStore, cluster_store: ClusterStore):
        """Initialize the clustering engine"""
        self.event_store = event_store
        self.cluster_store = cluster_store
        self.temporal_engine = TemporalCorrelationEngine(event_store)
        self.min_correlation_score = 0.3  # Minimum score to consider correlation
        self.cluster_merge_threshold = 0.6  # Similarity threshold for merging clusters
        
    def process_event(self, event: CorrelationEvent) -> List[str]:
        """Process a new event and update clusters"""
        # Store the event
        self.event_store.store_event(event)
        
        # Find correlated events
        correlated_events = self.temporal_engine.find_correlated_events(event.event_id)
        
        # Filter by minimum correlation score
        significant_correlations = [
            (event_id, score, level) for event_id, score, level in correlated_events
            if score >= self.min_correlation_score
        ]
        
        # Find clusters containing correlated events
        affected_clusters = set()
        for event_id, _, _ in significant_correlations:
            clusters = self.cluster_store.get_clusters_by_event(event_id)
            for cluster in clusters:
                affected_clusters.add(cluster.cluster_id)
                
        # Update existing clusters or create new ones
        updated_clusters = []
        
        if significant_correlations and not affected_clusters:
            # Create new cluster
            new_cluster = self._create_cluster([event.event_id] + [e[0] for e in significant_correlations])
            updated_clusters.append(new_cluster)
        else:
            # Add event to existing clusters
            for cluster_id in affected_clusters:
                cluster = self.cluster_store.get_cluster(cluster_id)
                if cluster:
                    if event.event_id not in cluster.events:
                        cluster.add_event(event.event_id, event)
                        self.cluster_store.store_cluster(cluster)
                        updated_clusters.append(cluster)
                        
        # Check for clusters that should be merged
        if len(updated_clusters) > 1:
            self._check_cluster_merges([c.cluster_id for c in updated_clusters])
            
        return [c.cluster_id for c in updated_clusters]
    
    def _create_cluster(self, event_ids: List[str]) -> CorrelationCluster:
        """Create a new cluster from a set of events"""
        # Create the cluster
        cluster = CorrelationCluster()
        
        # Add events
        events = []
        for event_id in event_ids:
            event = self.event_store.get_event(event_id)
            if event:
                events.append(event)
                cluster.add_event(event_id, event)
                
        # Set created time to earliest event time
        if events:
            cluster.created_at = min(event.timestamp for event in events)
            
        # Calculate correlation level
        if len(events) >= 2:
            # Calculate all pairwise correlations
            correlation_scores = []
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    score, level = self.temporal_engine._calculate_correlation(events[i], events[j])
                    correlation_scores.append((score, level))
                    
            # Use average score and maximum level
            avg_score = sum(score for score, _ in correlation_scores) / max(1, len(correlation_scores))
            max_level = max((level for _, level in correlation_scores), default=CorrelationLevel.WEAK)
            
            cluster.score = avg_score
            cluster.correlation_level = max_level
        else:
            cluster.score = 0.5  # Default score for single-event clusters
            cluster.correlation_level = CorrelationLevel.WEAK
            
        # Store the cluster
        self.cluster_store.store_cluster(cluster)
        
        return cluster
    
    def _check_cluster_merges(self, cluster_ids: List[str]) -> Optional[str]:
        """Check if clusters should be merged and merge them if needed"""
        if len(cluster_ids) < 2:
            return None
            
        # Get clusters
        clusters = [self.cluster_store.get_cluster(cid) for cid in cluster_ids]
        clusters = [c for c in clusters if c]
        
        if len(clusters) < 2:
            return None
            
        # Calculate pairwise similarities
        merge_candidates = []
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                similarity = self._calculate_cluster_similarity(clusters[i], clusters[j])
                if similarity >= self.cluster_merge_threshold:
                    merge_candidates.append((clusters[i].cluster_id, clusters[j].cluster_id, similarity))
                    
        # Sort by similarity (highest first)
        merge_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Perform merges
        merged_clusters = set()
        for c1, c2, _ in merge_candidates:
            if c1 not in merged_clusters and c2 not in merged_clusters:
                merged_id = self.cluster_store.merge_clusters([c1, c2])
                if merged_id:
                    merged_clusters.add(c1)
                    merged_clusters.add(c2)
                    
        return None if not merged_clusters else "merged"
    
    def _calculate_cluster_similarity(self, cluster1: CorrelationCluster, cluster2: CorrelationCluster) -> float:
        """Calculate similarity between two clusters"""
        # Calculate Jaccard similarity of events
        events1 = set(cluster1.events)
        events2 = set(cluster2.events)
        
        if not events1 or not events2:
            return 0.0
            
        intersection = len(events1.intersection(events2))
        union = len(events1.union(events2))
        
        event_similarity = intersection / union if union > 0 else 0.0
        
        # Calculate user overlap
        users1 = cluster1.user_ids
        users2 = cluster2.user_ids
        
        user_similarity = 0.0
        if users1 and users2:
            user_intersection = len(users1.intersection(users2))
            user_union = len(users1.union(users2))
            user_similarity = user_intersection / user_union if user_union > 0 else 0.0
            
        # Calculate temporal similarity
        time_diff = abs(cluster1.created_at - cluster2.created_at)
        temporal_similarity = max(0.0, 1.0 - time_diff / (24 * 3600))  # Within a day is similar
        
        # Combined similarity
        return event_similarity * 0.6 + user_similarity * 0.3 + temporal_similarity * 0.1

class PatternMiningEngine:
    """Discovers temporal patterns in event sequences"""
    
    def __init__(self, event_store: EventStore, pattern_store: PatternStore):
        """Initialize the pattern mining engine"""
        self.event_store = event_store
        self.pattern_store = pattern_store
        self.temporal_engine = TemporalCorrelationEngine(event_store)
        self.min_pattern_confidence = 0.5  # Minimum confidence for pattern recognition
        self.min_sequence_length = 3  # Minimum events in a sequence
        self.max_sequence_length = 10  # Maximum events in a sequence
        
    def mine_patterns(self, start_event_ids: Optional[List[str]] = None, limit: int = 10) -> List[str]:
        """Mine temporal patterns from event sequences"""
        if not start_event_ids:
            # Use recent events as starting points
            events = list(self.event_store.events.values())
            # Sort by time (most recent first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            # Take most recent events as starting points
            start_event_ids = [e.event_id for e in events[:limit]]
            
        discovered_patterns = []
        
        # Process each starting event
        for event_id in start_event_ids[:limit]:
            # Find a sequence starting from this event
            sequence = self.temporal_engine.find_event_sequence(
                event_id, self.max_sequence_length
            )
            
            # Skip sequences that are too short
            if len(sequence) < self.min_sequence_length:
                continue
                
            # Extract event types and time intervals
            event_types = []
            timestamps = []
            
            for seq_event_id in sequence:
                event = self.event_store.get_event(seq_event_id)
                if event:
                    event_types.append(event.event_type.name)
                    timestamps.append(event.timestamp)
                    
            # Calculate time intervals
            time_intervals = [
                timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            
            # Check if this matches an existing pattern
            pattern_id = self.pattern_store.find_matching_pattern(
                event_types, time_intervals
            )
            
            if pattern_id:
                # Update existing pattern
                pattern = self.pattern_store.get_pattern(pattern_id)
                if pattern:
                    self.pattern_store.increment_pattern_occurrence(pattern_id)
                    
                    # Update confidence based on occurrences
                    new_confidence = min(0.95, 0.5 + 0.1 * pattern.occurrences)
                    self.pattern_store.update_pattern_confidence(pattern_id, new_confidence)
                    
                    discovered_patterns.append(pattern_id)
            else:
                # Create new pattern
                new_pattern = TemporalPattern(
                    event_sequence=event_types,
                    time_intervals=time_intervals,
                    confidence=0.5,  # Initial confidence
                    occurrences=1
                )
                
                new_pattern_id = self.pattern_store.store_pattern(new_pattern)
                discovered_patterns.append(new_pattern_id)
                
        return discovered_patterns
    
    def match_pattern(self, event_ids: List[str]) -> Optional[Tuple[str, float]]:
        """Match a sequence of events against known patterns"""
        if not event_ids or len(event_ids) < self.min_sequence_length:
            return None
            
        # Extract event types and time intervals
        event_types = []
        timestamps = []
        
        for event_id in event_ids:
            event = self.event_store.get_event(event_id)
            if event:
                event_types.append(event.event_type.name)
                timestamps.append(event.timestamp)
                
        # Calculate time intervals
        time_intervals = [
            timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]
        
        # Find best matching pattern
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern in self.pattern_store.patterns.items():
            # Skip if sequence lengths don't match
            if len(pattern.event_sequence) != len(event_types):
                continue
                
            # Calculate sequence similarity
            type_similarity = sum(1 for a, b in zip(pattern.event_sequence, event_types) if a == b)
            type_similarity /= len(event_types)
            
            # Calculate time interval similarity
            interval_similarity = 1.0
            if time_intervals and pattern.time_intervals:
                # Use helper method from pattern store
                interval_similarity = self.pattern_store._calculate_interval_similarity(
                    pattern.time_intervals, time_intervals
                )
                
            # Combined similarity
            combined_similarity = type_similarity * 0.7 + interval_similarity * 0.3
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = pattern_id
                
        if best_match and best_similarity >= self.min_pattern_confidence:
            return best_match, best_similarity
            
        return None
    
    def predict_next_event(self, event_ids: List[str]) -> Optional[Dict]:
        """Predict the next event based on pattern matching"""
        if not event_ids:
            return None
            
        # Try to match against known patterns
        match_result = self.match_pattern(event_ids)
        
        if not match_result:
            return None
            
        pattern_id, similarity = match_result
        pattern = self.pattern_store.get_pattern(pattern_id)
        
        if not pattern or len(pattern.event_sequence) <= len(event_ids):
            return None
            
        # Predict next event type
        next_event_type = pattern.event_sequence[len(event_ids)]
        
        # Predict when it will happen
        current_time = time.time()
        if pattern.time_intervals and len(event_ids) <= len(pattern.time_intervals):
            time_interval = pattern.time_intervals[len(event_ids) - 1]
            last_event = self.event_store.get_event(event_ids[-1])
            if last_event:
                predicted_time = last_event.timestamp + time_interval
            else:
                predicted_time = current_time + time_interval
        else:
            # Default to soon
            predicted_time = current_time + 60  # 1 minute
            
        return {
            "predicted_event_type": next_event_type,
            "predicted_timestamp": predicted_time,
            "confidence": similarity * pattern.confidence,
            "pattern_id": pattern_id,
            "current_time": current_time
        }

class RealTimeCorrelationSystem:
    """Main system for real-time correlation tracking"""
    
    def __init__(self, max_events: int = 100000, max_clusters: int = 10000, max_patterns: int = 1000):
        """Initialize the correlation system"""
        self.event_store = EventStore(max_events)
        self.cluster_store = ClusterStore(max_clusters)
        self.pattern_store = PatternStore(max_patterns)
        
        self.clustering_engine = ClusteringEngine(self.event_store, self.cluster_store)
        self.pattern_mining_engine = PatternMiningEngine(self.event_store, self.pattern_store)
        
        self.pattern_mining_interval = 300  # 5 minutes
        self.last_pattern_mining = 0
        
        # Start background tasks
        self.stop_event = threading.Event()
        self.background_thread = threading.Thread(target=self._background_task, daemon=True)
        self.background_thread.start()
        
        logger.info("Real-Time Correlation System initialized")
        
    def track_event(self, event: Dict) -> Dict:
        """Track an event and identify correlations"""
        # Create a correlation event
        correlation_event = CorrelationEvent(
            event_id=event.get("event_id", str(uuid.uuid4())),
            correlation_id=event.get("correlation_id"),
            user_id=event.get("user_id"),
            session_id=event.get("session_id"),
            timestamp=event.get("timestamp", time.time()),
            event_type=EventType[event.get("event_type", "USER_ACTION")],
            source=event.get("source", "unknown"),
            target=event.get("target", "unknown"),
            data=event.get("data", {}),
            tags=set(event.get("tags", [])),
            parent_id=event.get("parent_id"),
            quantum_state=event.get("quantum_state"),
            duration_ms=event.get("duration_ms")
        )
        
        # Process the event
        affected_clusters = self.clustering_engine.process_event(correlation_event)
        
        # Check if we should mine patterns
        current_time = time.time()
        if current_time - self.last_pattern_mining >= self.pattern_mining_interval:
            self.pattern_mining_engine.mine_patterns()
            self.last_pattern_mining = current_time
            
        # Return event info
        return {
            "event_id": correlation_event.event_id,
            "correlation_id": correlation_event.correlation_id,
            "timestamp": correlation_event.timestamp,
            "affected_clusters": affected_clusters
        }
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """Get an event by ID"""
        event = self.event_store.get_event(event_id)
        if event:
            return event.to_dict()
        return None
    
    def get_cluster(self, cluster_id: str) -> Optional[Dict]:
        """Get a cluster by ID"""
        cluster = self.cluster_store.get_cluster(cluster_id)
        if cluster:
            return cluster.to_dict()
        return None
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """Get a pattern by ID"""
        pattern = self.pattern_store.get_pattern(pattern_id)
        if pattern:
            return pattern.to_dict()
        return None
    
    def get_clusters_by_user(self, user_id: str) -> List[Dict]:
        """Get all clusters for a specific user"""
        clusters = self.cluster_store.get_clusters_by_user(user_id)
        return [cluster.to_dict() for cluster in clusters]
    
    def get_events_by_user(self, user_id: str) -> List[Dict]:
        """Get all events for a specific user"""
        events = self.event_store.get_events_by_user(user_id)
        return [event.to_dict() for event in events]
    
    def get_correlated_events(self, event_id: str) -> List[Dict]:
        """Get events correlated with the given event"""
        temporal_engine = TemporalCorrelationEngine(self.event_store)
        correlated = temporal_engine.find_correlated_events(event_id)
        
        result = []
        for corr_event_id, score, level in correlated:
            event = self.event_store.get_event(corr_event_id)
            if event:
                result.append({
                    "event": event.to_dict(),
                    "correlation_score": score,
                    "correlation_level": level.name
                })
                
        return result
    
    def predict_next_events(self, event_ids: List[str]) -> List[Dict]:
        """Predict next events based on patterns"""
        prediction = self.pattern_mining_engine.predict_next_event(event_ids)
        if prediction:
            return [prediction]
        return []
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        current_time = time.time()
        
        return {
            "current_time": current_time,
            "formatted_time": datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S"),
            "user": os.environ.get("USER_LOGIN", "Shriram-2005"),  # Get from environment
            "event_stats": self.event_store.get_stats(),
            "cluster_stats": self.cluster_store.get_stats(),
            "pattern_stats": self.pattern_store.get_stats(),
            "last_pattern_mining": self.last_pattern_mining,
            "uptime_seconds": current_time - self.last_pattern_mining
        }
    
    def shutdown(self):
        """Shutdown the correlation system"""
        self.stop_event.set()
        self.background_thread.join(timeout=2.0)
        logger.info("Real-Time Correlation System shut down")
        
    def _background_task(self):
        """Background task for maintenance operations"""
        logger.info("Background task started")
        
        while not self.stop_event.is_set():
            try:
                # Periodically mine for patterns
                current_time = time.time()
                if current_time - self.last_pattern_mining >= self.pattern_mining_interval:
                    self.pattern_mining_engine.mine_patterns()
                    self.last_pattern_mining = current_time
                    logger.debug("Background pattern mining completed")
                    
            except Exception as e:
                logger.error(f"Error in background task: {str(e)}")
                
            # Sleep for a while
            self.stop_event.wait(60)  # Check every minute

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Get current date/time
    current_time = "2025-07-24 06:54:32"
    print(f"Current Date/Time: {current_time}")
    
    # Get user login
    user_login = "Shriram-2005"
    print(f"User: {user_login}")
    
    # Create the correlation system
    system = RealTimeCorrelationSystem()
    
    # Track some example events
    session_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    
    # Login event
    login_event = {
        "event_type": "USER_ACTION",
        "user_id": user_login,
        "session_id": session_id,
        "correlation_id": correlation_id,
        "source": "login_page",
        "target": "auth_system",
        "data": {"action": "login", "success": True},
        "tags": ["login", "authentication"]
    }
    login_result = system.track_event(login_event)
    
    # Dashboard view event
    dashboard_event = {
        "event_type": "USER_ACTION",
        "user_id": user_login,
        "session_id": session_id,
        "correlation_id": correlation_id,
        "source": "dashboard",
        "target": "ui",
        "data": {"action": "view", "page": "dashboard"},
        "tags": ["dashboard", "view"],
        "parent_id": login_result["event_id"]
    }
    dashboard_result = system.track_event(dashboard_event)
    
    # API request event
    api_event = {
        "event_type": "SYSTEM_RESPONSE",
        "user_id": user_login,
        "session_id": session_id,
        "correlation_id": correlation_id,
        "source": "api",
        "target": "dashboard",
        "data": {"action": "data_load", "endpoint": "/api/data"},
        "tags": ["api", "data_load"],
        "parent_id": dashboard_result["event_id"]
    }
    api_result = system.track_event(api_event)
    
    # Get correlated events
    correlated = system.get_correlated_events(login_result["event_id"])
    
    # Print correlation results
    print("\nCorrelated events:")
    for event in correlated:
        print(f"  Event: {event['event']['event_id']}")
        print(f"  Score: {event['correlation_score']:.2f}")
        print(f"  Level: {event['correlation_level']}")
        print(f"  Type: {event['event']['event_type']}")
        print()
    
    # Get clusters for the user
    clusters = system.get_clusters_by_user(user_login)
    
    print(f"Found {len(clusters)} clusters for user {user_login}")
    for cluster in clusters:
        print(f"  Cluster: {cluster['cluster_id']}")
        print(f"  Score: {cluster['score']:.2f}")
        print(f"  Level: {cluster['correlation_level']}")
        print(f"  Events: {cluster['event_count']}")
        print()
        
    # Get system stats
    stats = system.get_system_stats()
    
    print("System Statistics:")
    print(f"  Current Time: {stats['formatted_time']}")
    print(f"  User: {stats['user']}")
    print(f"  Events: {stats['event_stats']['total_events']}")
    print(f"  Clusters: {stats['cluster_stats']['total_clusters']}")
    print(f"  Patterns: {stats['pattern_stats']['total_patterns']}")