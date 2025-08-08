"""
Enterprise Security Event Logging System

This module provides comprehensive security event logging capabilities with advanced
threat detection, real-time monitoring, compliance reporting, and forensic analysis.

Features:
- Multi-tier security event classification
- Real-time threat correlation and analysis
- Compliance framework integration (SOX, GDPR, HIPAA, PCI-DSS)
- Advanced forensic data collection
- Distributed logging with encryption
- Behavioral anomaly detection
- Automated incident response triggers
- Performance monitoring and optimization

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
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
import sqlite3
import queue
import concurrent.futures

# Optional dependencies with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    cryptography = None

try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
    geoip2 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Configure module logger
logger = logging.getLogger(__name__)


class ThreatLevel(IntEnum):
    """
    Enhanced threat level classification with numerical scoring.
    
    Provides comprehensive threat assessment with integration points
    for automated response systems and compliance frameworks.
    """
    INFORMATIONAL = 0
    LOW = 1
    MEDIUM = 3
    HIGH = 7
    CRITICAL = 9
    CATASTROPHIC = 10
    
    @property
    def description(self) -> str:
        """Get threat level description."""
        descriptions = {
            ThreatLevel.INFORMATIONAL: "Informational event with no immediate threat",
            ThreatLevel.LOW: "Low-level threat with minimal risk",
            ThreatLevel.MEDIUM: "Medium threat requiring attention",
            ThreatLevel.HIGH: "High threat requiring immediate response",
            ThreatLevel.CRITICAL: "Critical threat requiring emergency response",
            ThreatLevel.CATASTROPHIC: "Catastrophic threat requiring all-hands response"
        }
        return descriptions.get(self, "Unknown threat level")
    
    @property
    def score(self) -> int:
        """Get numerical threat score for automated processing."""
        return self.value
    
    @classmethod
    def from_score(cls, score: int) -> 'ThreatLevel':
        """Convert numerical score to threat level."""
        for level in cls:
            if level.score >= score:
                return level
        return cls.CATASTROPHIC
    
    def escalate(self) -> 'ThreatLevel':
        """Escalate to next higher threat level."""
        current_score = self.score
        for level in sorted(ThreatLevel, key=lambda x: x.score):
            if level.score > current_score:
                return level
        return self  # Already at maximum
    
    def get_response_time_sla(self) -> int:
        """Get required response time in minutes based on threat level."""
        response_times = {
            ThreatLevel.INFORMATIONAL: 1440,  # 24 hours
            ThreatLevel.LOW: 480,             # 8 hours
            ThreatLevel.MEDIUM: 240,          # 4 hours
            ThreatLevel.HIGH: 60,             # 1 hour
            ThreatLevel.CRITICAL: 15,         # 15 minutes
            ThreatLevel.CATASTROPHIC: 5       # 5 minutes
        }
        return response_times.get(self, 15)


class SecurityEventType(Enum):
    """
    Comprehensive security event classification system.
    
    Covers all major security domains with detailed categorization
    for advanced threat detection and compliance reporting.
    """
    # Authentication and Authorization Events
    LOGIN_SUCCESS = ("auth.login.success", "Successful user authentication")
    LOGIN_FAILURE = ("auth.login.failure", "Failed authentication attempt")
    LOGIN_BRUTE_FORCE = ("auth.login.brute_force", "Brute force login attempt detected")
    LOGOUT = ("auth.logout", "User logout event")
    SESSION_TIMEOUT = ("auth.session.timeout", "User session timeout")
    PASSWORD_CHANGE = ("auth.password.change", "Password modification")
    ACCOUNT_LOCKOUT = ("auth.account.lockout", "Account locked due to policy violation")
    PRIVILEGE_ESCALATION = ("auth.privilege.escalation", "Privilege escalation attempt")
    UNAUTHORIZED_ACCESS = ("auth.access.unauthorized", "Unauthorized resource access")
    TOKEN_VALIDATION_FAILURE = ("auth.token.validation_failure", "Authentication token validation failed")
    
    # Network Security Events
    NETWORK_INTRUSION = ("network.intrusion", "Network intrusion attempt")
    FIREWALL_BLOCK = ("network.firewall.block", "Firewall blocked connection")
    DDoS_ATTACK = ("network.ddos", "Distributed Denial of Service attack")
    PORT_SCAN = ("network.port_scan", "Port scanning activity detected")
    SUSPICIOUS_TRAFFIC = ("network.traffic.suspicious", "Suspicious network traffic pattern")
    DNS_POISONING = ("network.dns.poisoning", "DNS poisoning attempt")
    MAN_IN_MIDDLE = ("network.mitm", "Man-in-the-middle attack detected")
    BANDWIDTH_ANOMALY = ("network.bandwidth.anomaly", "Unusual bandwidth consumption")
    
    # Data Security Events
    DATA_BREACH = ("data.breach", "Data breach incident")
    DATA_EXFILTRATION = ("data.exfiltration", "Unauthorized data export")
    DATA_CORRUPTION = ("data.corruption", "Data integrity violation")
    ENCRYPTION_FAILURE = ("data.encryption.failure", "Data encryption failure")
    BACKUP_FAILURE = ("data.backup.failure", "Data backup operation failed")
    DATA_CLASSIFICATION_VIOLATION = ("data.classification.violation", "Data classification policy violation")
    PII_EXPOSURE = ("data.pii.exposure", "Personal identifiable information exposure")
    FINANCIAL_DATA_ACCESS = ("data.financial.access", "Financial data access event")
    
    # Application Security Events
    SQL_INJECTION = ("app.sql_injection", "SQL injection attempt")
    XSS_ATTACK = ("app.xss", "Cross-site scripting attack")
    CSRF_ATTACK = ("app.csrf", "Cross-site request forgery attack")
    BUFFER_OVERFLOW = ("app.buffer_overflow", "Buffer overflow attempt")
    CODE_INJECTION = ("app.code_injection", "Code injection attempt")
    PATH_TRAVERSAL = ("app.path_traversal", "Path traversal attack")
    COMMAND_INJECTION = ("app.command_injection", "Command injection attempt")
    DESERIALIZATION_ATTACK = ("app.deserialization", "Unsafe deserialization attempt")
    API_ABUSE = ("app.api.abuse", "API abuse or misuse")
    
    # System Security Events
    MALWARE_DETECTION = ("system.malware", "Malware detection")
    VIRUS_DETECTION = ("system.virus", "Virus detection")
    ROOTKIT_DETECTION = ("system.rootkit", "Rootkit detection")
    SYSTEM_COMPROMISE = ("system.compromise", "System compromise detected")
    FILE_INTEGRITY_VIOLATION = ("system.file_integrity", "File integrity check failed")
    REGISTRY_MODIFICATION = ("system.registry.modification", "System registry modification")
    SERVICE_MANIPULATION = ("system.service.manipulation", "System service manipulation")
    PROCESS_ANOMALY = ("system.process.anomaly", "Abnormal process behavior")
    
    # Compliance and Governance Events
    COMPLIANCE_VIOLATION = ("compliance.violation", "Regulatory compliance violation")
    AUDIT_FAILURE = ("compliance.audit.failure", "Audit trail failure")
    POLICY_VIOLATION = ("compliance.policy.violation", "Security policy violation")
    DATA_RETENTION_VIOLATION = ("compliance.retention.violation", "Data retention policy violation")
    PRIVACY_VIOLATION = ("compliance.privacy.violation", "Privacy policy violation")
    
    # Incident Response Events
    INCIDENT_DECLARED = ("incident.declared", "Security incident declared")
    INCIDENT_ESCALATED = ("incident.escalated", "Security incident escalated")
    INCIDENT_RESOLVED = ("incident.resolved", "Security incident resolved")
    FORENSIC_ANALYSIS = ("incident.forensics", "Forensic analysis initiated")
    CONTAINMENT_ACTION = ("incident.containment", "Incident containment action")
    
    # Threat Intelligence Events
    IOC_MATCH = ("threat.ioc.match", "Indicator of Compromise match")
    THREAT_FEED_UPDATE = ("threat.feed.update", "Threat intelligence feed update")
    BLACKLIST_MATCH = ("threat.blacklist.match", "Blacklist entry match")
    REPUTATION_VIOLATION = ("threat.reputation.violation", "IP/Domain reputation violation")
    BEHAVIORAL_ANOMALY = ("threat.behavior.anomaly", "Behavioral anomaly detected")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    @classmethod
    def from_code(cls, code: str) -> 'SecurityEventType':
        """Convert code string to SecurityEventType."""
        for event_type in cls:
            if event_type.code == code:
                return event_type
        return cls.COMPLIANCE_VIOLATION  # Default fallback
    
    def get_default_threat_level(self) -> ThreatLevel:
        """Get default threat level for this event type."""
        high_risk_events = [
            self.DATA_BREACH, self.SYSTEM_COMPROMISE, self.MALWARE_DETECTION,
            self.PRIVILEGE_ESCALATION, self.DDoS_ATTACK, self.SQL_INJECTION
        ]
        critical_events = [
            self.DATA_EXFILTRATION, self.ROOTKIT_DETECTION, self.NETWORK_INTRUSION
        ]
        
        if self in critical_events:
            return ThreatLevel.CRITICAL
        elif self in high_risk_events:
            return ThreatLevel.HIGH
        elif "failure" in self.code or "violation" in self.code:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class ComplianceFramework(Enum):
    """Compliance framework classifications."""
    SOX = ("SOX", "Sarbanes-Oxley Act")
    GDPR = ("GDPR", "General Data Protection Regulation")
    HIPAA = ("HIPAA", "Health Insurance Portability and Accountability Act")
    PCI_DSS = ("PCI-DSS", "Payment Card Industry Data Security Standard")
    ISO27001 = ("ISO27001", "ISO/IEC 27001 Information Security Management")
    NIST_CSF = ("NIST-CSF", "NIST Cybersecurity Framework")
    CIS_CONTROLS = ("CIS", "Center for Internet Security Controls")
    COBIT = ("COBIT", "Control Objectives for Information Technologies")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class SecurityEventSeverity(Enum):
    """Security event severity classification."""
    DEBUG = ("DEBUG", "Debug-level security information")
    INFO = ("INFO", "Informational security event")
    NOTICE = ("NOTICE", "Notable security event")
    WARNING = ("WARNING", "Warning-level security event")
    ERROR = ("ERROR", "Error-level security event")
    CRITICAL = ("CRITICAL", "Critical security event")
    ALERT = ("ALERT", "Alert-level security event")
    EMERGENCY = ("EMERGENCY", "Emergency-level security event")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class SecurityContext:
    """
    Comprehensive security context for events.
    
    Captures environmental and contextual information
    for advanced threat analysis and forensic investigation.
    """
    # User context
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    user_groups: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    authentication_method: Optional[str] = None
    
    # Network context
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Geographic context
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    
    # System context
    hostname: Optional[str] = None
    operating_system: Optional[str] = None
    process_id: Optional[int] = None
    process_name: Optional[str] = None
    parent_process_id: Optional[int] = None
    
    # Application context
    application_name: Optional[str] = None
    application_version: Optional[str] = None
    module_name: Optional[str] = None
    function_name: Optional[str] = None
    
    # Request context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    request_method: Optional[str] = None
    request_uri: Optional[str] = None
    response_code: Optional[int] = None
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if not self.hostname:
            self.hostname = socket.gethostname()
        
        if not self.operating_system:
            self.operating_system = platform.platform()
        
        if not self.process_id:
            self.process_id = os.getpid()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_context": {
                "user_id": self.user_id,
                "username": self.username,
                "user_roles": self.user_roles,
                "user_groups": self.user_groups,
                "session_id": self.session_id,
                "authentication_method": self.authentication_method
            },
            "network_context": {
                "source_ip": self.source_ip,
                "destination_ip": self.destination_ip,
                "source_port": self.source_port,
                "destination_port": self.destination_port,
                "protocol": self.protocol,
                "user_agent": self.user_agent
            },
            "geographic_context": {
                "country": self.country,
                "region": self.region,
                "city": self.city,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": self.timezone
            },
            "system_context": {
                "hostname": self.hostname,
                "operating_system": self.operating_system,
                "process_id": self.process_id,
                "process_name": self.process_name,
                "parent_process_id": self.parent_process_id
            },
            "application_context": {
                "application_name": self.application_name,
                "application_version": self.application_version,
                "module_name": self.module_name,
                "function_name": self.function_name
            },
            "request_context": {
                "request_id": self.request_id,
                "correlation_id": self.correlation_id,
                "request_method": self.request_method,
                "request_uri": self.request_uri,
                "response_code": self.response_code
            }
        }


@dataclass
class SecurityEvent:
    """
    Comprehensive security event representation.
    
    Captures detailed security event information with rich context,
    threat assessment, and compliance metadata for enterprise security operations.
    """
    # Core event identification
    event_id: str = field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:16]}")
    event_type: SecurityEventType = SecurityEventType.COMPLIANCE_VIOLATION
    event_category: str = "security"
    event_subcategory: str = "general"
    
    # Temporal properties
    timestamp: float = field(default_factory=time.time)
    event_duration_ms: Optional[float] = None
    detection_timestamp: Optional[float] = None
    
    # Event description and metadata
    title: str = "Security Event"
    description: str = "A security event has occurred"
    detailed_description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Threat assessment
    threat_level: ThreatLevel = ThreatLevel.LOW
    threat_score: float = 0.0
    risk_score: float = 0.0
    confidence_score: float = 0.0
    false_positive_probability: float = 0.0
    
    # Security context
    security_context: Optional[SecurityContext] = None
    
    # Event payload and evidence
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Compliance and regulatory
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    regulatory_impact: bool = False
    data_classification: str = "internal"
    
    # Investigation and response
    investigation_required: bool = False
    automated_response_triggered: bool = False
    response_actions: List[str] = field(default_factory=list)
    escalation_path: List[str] = field(default_factory=list)
    
    # Correlation and relationships
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)
    related_event_ids: List[str] = field(default_factory=list)
    incident_id: Optional[str] = None
    
    # Status and workflow
    status: str = "open"
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO
    priority: int = 5  # 1=highest, 10=lowest
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    
    # Forensic information
    chain_of_custody: List[Dict[str, Any]] = field(default_factory=list)
    forensic_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    
    # Performance and metrics
    processing_time_ms: Optional[float] = None
    detection_accuracy: Optional[float] = None
    
    # Additional metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        # Set detection timestamp if not provided
        if self.detection_timestamp is None:
            self.detection_timestamp = self.timestamp
        
        # Initialize security context if not provided
        if self.security_context is None:
            self.security_context = SecurityContext()
        
        # Set default threat level based on event type
        if self.threat_level == ThreatLevel.LOW:
            self.threat_level = self.event_type.get_default_threat_level()
        
        # Set threat score based on threat level
        self.threat_score = self.threat_level.score
        
        # Add automatic tags
        self._add_automatic_tags()
        
        # Generate forensic hash
        self._generate_forensic_hash()
    
    def _add_automatic_tags(self) -> None:
        """Add automatic tags based on event properties."""
        self.tags.add(f"type:{self.event_type.code}")
        self.tags.add(f"threat_level:{self.threat_level.name.lower()}")
        self.tags.add(f"severity:{self.severity.code.lower()}")
        
        if self.security_context and self.security_context.source_ip:
            self.tags.add("network_event")
        
        if self.investigation_required:
            self.tags.add("investigation_required")
        
        if self.regulatory_impact:
            self.tags.add("regulatory_impact")
        
        for framework in self.compliance_frameworks:
            self.tags.add(f"compliance:{framework.code.lower()}")
    
    def _generate_forensic_hash(self) -> None:
        """Generate forensic hash for event integrity."""
        content = f"{self.event_id}:{self.timestamp}:{self.event_type.code}:{self.description}"
        self.forensic_hash = hashlib.sha256(content.encode()).hexdigest()
    
    def escalate_threat_level(self) -> None:
        """Escalate the threat level and update related fields."""
        self.threat_level = self.threat_level.escalate()
        self.threat_score = self.threat_level.score
        self.investigation_required = True
        
        # Update tags
        self.tags.discard(f"threat_level:{self.threat_level.name.lower()}")
        self.tags.add(f"threat_level:{self.threat_level.name.lower()}")
    
    def add_related_event(self, event_id: str) -> None:
        """Add a related event ID."""
        if event_id not in self.related_event_ids:
            self.related_event_ids.append(event_id)
    
    def add_child_event(self, event_id: str) -> None:
        """Add a child event ID."""
        if event_id not in self.child_event_ids:
            self.child_event_ids.append(event_id)
    
    def add_evidence(self, key: str, value: Any) -> None:
        """Add evidence to the event."""
        self.evidence[key] = value
        self.tags.add("has_evidence")
    
    def add_artifact(self, artifact_path: str) -> None:
        """Add an artifact file path."""
        if artifact_path not in self.artifacts:
            self.artifacts.append(artifact_path)
            self.tags.add("has_artifacts")
    
    def trigger_automated_response(self, action: str) -> None:
        """Trigger an automated response action."""
        self.automated_response_triggered = True
        if action not in self.response_actions:
            self.response_actions.append(action)
        self.tags.add("automated_response")
    
    def get_age_seconds(self) -> float:
        """Get the age of the event in seconds."""
        return time.time() - self.timestamp
    
    def is_within_sla(self) -> bool:
        """Check if event is still within SLA response time."""
        required_response_minutes = self.threat_level.get_response_time_sla()
        age_minutes = self.get_age_seconds() / 60
        return age_minutes <= required_response_minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            # Core identification
            "event_id": self.event_id,
            "event_type": self.event_type.code,
            "event_type_description": self.event_type.description,
            "event_category": self.event_category,
            "event_subcategory": self.event_subcategory,
            
            # Temporal information
            "timestamp": self.timestamp,
            "formatted_timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "event_duration_ms": self.event_duration_ms,
            "detection_timestamp": self.detection_timestamp,
            "age_seconds": self.get_age_seconds(),
            
            # Event description
            "title": self.title,
            "description": self.description,
            "detailed_description": self.detailed_description,
            "tags": list(self.tags),
            "labels": self.labels,
            
            # Threat assessment
            "threat_level": self.threat_level.name,
            "threat_level_score": self.threat_level.score,
            "threat_score": self.threat_score,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "false_positive_probability": self.false_positive_probability,
            
            # Security context
            "security_context": self.security_context.to_dict() if self.security_context else None,
            
            # Event data
            "payload": self.payload,
            "evidence": self.evidence,
            "artifacts": self.artifacts,
            
            # Compliance
            "compliance_frameworks": [f.code for f in self.compliance_frameworks],
            "regulatory_impact": self.regulatory_impact,
            "data_classification": self.data_classification,
            
            # Investigation and response
            "investigation_required": self.investigation_required,
            "automated_response_triggered": self.automated_response_triggered,
            "response_actions": self.response_actions,
            "escalation_path": self.escalation_path,
            
            # Relationships
            "parent_event_id": self.parent_event_id,
            "child_event_ids": self.child_event_ids,
            "related_event_ids": self.related_event_ids,
            "incident_id": self.incident_id,
            
            # Status and workflow
            "status": self.status,
            "severity": self.severity.code,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "resolution": self.resolution,
            
            # Forensics
            "chain_of_custody": self.chain_of_custody,
            "forensic_hash": self.forensic_hash,
            "digital_signature": self.digital_signature,
            
            # Performance
            "processing_time_ms": self.processing_time_ms,
            "detection_accuracy": self.detection_accuracy,
            
            # SLA compliance
            "within_sla": self.is_within_sla(),
            "sla_response_time_minutes": self.threat_level.get_response_time_sla(),
            
            # Additional fields
            "custom_fields": self.custom_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create SecurityEvent from dictionary."""
        # Extract security context if present
        security_context = None
        if data.get("security_context"):
            ctx_data = data["security_context"]
            security_context = SecurityContext(
                user_id=ctx_data.get("user_context", {}).get("user_id"),
                username=ctx_data.get("user_context", {}).get("username"),
                user_roles=ctx_data.get("user_context", {}).get("user_roles", []),
                source_ip=ctx_data.get("network_context", {}).get("source_ip"),
                hostname=ctx_data.get("system_context", {}).get("hostname"),
                # Add other context fields as needed
            )
        
        return cls(
            event_id=data.get("event_id", f"sec_{uuid.uuid4().hex[:16]}"),
            event_type=SecurityEventType.from_code(data.get("event_type", "compliance.violation")),
            title=data.get("title", "Security Event"),
            description=data.get("description", "A security event has occurred"),
            threat_level=ThreatLevel.from_score(data.get("threat_level_score", 1)),
            security_context=security_context,
            payload=data.get("payload", {}),
            evidence=data.get("evidence", {}),
            tags=set(data.get("tags", [])),
            compliance_frameworks=set([ComplianceFramework(f) for f in data.get("compliance_frameworks", [])]),
            # Add other fields as needed
        )


class SecurityEventStore:
    """
    Enterprise-grade security event storage system.
    
    Provides high-performance storage, indexing, and retrieval of security events
    with advanced search capabilities and compliance-ready audit trails.
    """
    
    def __init__(self,
                 max_events: int = 1000000,
                 enable_encryption: bool = True,
                 enable_redis: bool = True,
                 redis_config: Optional[Dict[str, Any]] = None,
                 database_path: str = "security_events.db"):
        """
        Initialize the security event store.
        
        Args:
            max_events: Maximum number of events to store in memory
            enable_encryption: Whether to encrypt stored events
            enable_redis: Whether to use Redis for distributed storage
            redis_config: Redis configuration parameters
            database_path: Path to SQLite database for persistence
        """
        self.max_events = max_events
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        self.database_path = database_path
        
        # In-memory storage with indexing
        self.events: Dict[str, SecurityEvent] = {}
        self.events_by_type: Dict[SecurityEventType, List[str]] = defaultdict(list)
        self.events_by_threat_level: Dict[ThreatLevel, List[str]] = defaultdict(list)
        self.events_by_user: Dict[str, List[str]] = defaultdict(list)
        self.events_by_ip: Dict[str, List[str]] = defaultdict(list)
        self.events_by_status: Dict[str, List[str]] = defaultdict(list)
        self.events_by_time: List[Tuple[float, str]] = []  # (timestamp, event_id)
        self.events_by_tags: Dict[str, List[str]] = defaultdict(list)
        self.events_by_incident: Dict[str, List[str]] = defaultdict(list)
        
        # Performance optimization
        self.event_cache: Dict[str, SecurityEvent] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_max_size = 10000
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_threat_level": defaultdict(int),
            "events_by_type": defaultdict(int),
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_operations": 0,
            "query_operations": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        
        # Encryption setup
        self.encryption_key = None
        if self.enable_encryption:
            self._setup_encryption()
        
        # Redis setup
        self.redis_client = None
        if enable_redis and REDIS_AVAILABLE:
            self._setup_redis(redis_config)
        
        # Database setup
        self._setup_database()
        
        # Background maintenance
        self._maintenance_thread = None
        self._shutdown_event = threading.Event()
        self._start_maintenance_thread()
    
    def store_event(self, event: SecurityEvent) -> str:
        """Store a security event with full indexing."""
        with self._write_lock:
            event_id = event.event_id
            
            # Store in memory
            self.events[event_id] = event
            
            # Update indexes
            self._update_indexes(event, event_id)
            
            # Cache the event
            self._cache_event(event_id, event)
            
            # Store in Redis if enabled
            if self.redis_client:
                try:
                    event_data = json.dumps(event.to_dict(), default=str)
                    if self.enable_encryption:
                        event_data = self._encrypt_data(event_data)
                    self.redis_client.hset("security_events", event_id, event_data)
                    self.redis_client.expire(f"security_events:{event_id}", 2592000)  # 30 days
                except Exception as e:
                    logger.error(f"Redis storage failed for event {event_id}: {e}")
            
            # Store in database
            self._store_event_db(event)
            
            # Update statistics
            self.stats["total_events"] += 1
            self.stats["events_by_threat_level"][event.threat_level] += 1
            self.stats["events_by_type"][event.event_type] += 1
            self.stats["storage_operations"] += 1
            
            # Enforce storage limits
            self._enforce_limits()
            
            return event_id
    
    def get_event(self, event_id: str) -> Optional[SecurityEvent]:
        """Retrieve a security event by ID."""
        with self._lock:
            # Check cache first
            if event_id in self.event_cache:
                self.cache_access_times[event_id] = time.time()
                self.stats["cache_hits"] += 1
                return self.event_cache[event_id]
            
            # Check memory storage
            if event_id in self.events:
                event = self.events[event_id]
                self._cache_event(event_id, event)
                return event
            
            # Check Redis
            if self.redis_client:
                try:
                    event_data = self.redis_client.hget("security_events", event_id)
                    if event_data:
                        if self.enable_encryption:
                            event_data = self._decrypt_data(event_data)
                        event_dict = json.loads(event_data)
                        event = SecurityEvent.from_dict(event_dict)
                        self._cache_event(event_id, event)
                        return event
                except Exception as e:
                    logger.error(f"Redis retrieval failed for event {event_id}: {e}")
            
            # Check database
            event = self._get_event_db(event_id)
            if event:
                self._cache_event(event_id, event)
                return event
            
            self.stats["cache_misses"] += 1
            return None
    
    def search_events(self, criteria: Dict[str, Any], limit: int = 100) -> List[SecurityEvent]:
        """Advanced event search with multiple criteria."""
        with self._lock:
            self.stats["query_operations"] += 1
            
            # Start with appropriate index
            candidate_event_ids = self._get_candidate_events(criteria)
            
            # Apply filters
            results = []
            for event_id in candidate_event_ids:
                event = self.get_event(event_id)
                if event and self._matches_criteria(event, criteria):
                    results.append(event)
                    if len(results) >= limit:
                        break
            
            return results
    
    def get_events_by_threat_level(self, threat_level: ThreatLevel, limit: int = 100) -> List[SecurityEvent]:
        """Get events by threat level."""
        event_ids = self.events_by_threat_level[threat_level][-limit:]
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_type(self, event_type: SecurityEventType, limit: int = 100) -> List[SecurityEvent]:
        """Get events by type."""
        event_ids = self.events_by_type[event_type][-limit:]
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[SecurityEvent]:
        """Get events for a specific user."""
        event_ids = self.events_by_user[user_id][-limit:]
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_events_by_ip(self, ip_address: str, limit: int = 100) -> List[SecurityEvent]:
        """Get events from a specific IP address."""
        event_ids = self.events_by_ip[ip_address][-limit:]
        return [event for event in [self.get_event(eid) for eid in event_ids] if event]
    
    def get_recent_events(self, minutes: int = 60, limit: int = 100) -> List[SecurityEvent]:
        """Get recent events within specified time window."""
        cutoff_time = time.time() - (minutes * 60)
        recent_event_ids = []
        
        for timestamp, event_id in reversed(self.events_by_time):
            if timestamp < cutoff_time:
                break
            recent_event_ids.append(event_id)
            if len(recent_event_ids) >= limit:
                break
        
        return [event for event in [self.get_event(eid) for eid in recent_event_ids] if event]
    
    def get_high_risk_events(self, hours: int = 24) -> List[SecurityEvent]:
        """Get high-risk events from recent time period."""
        cutoff_time = time.time() - (hours * 3600)
        high_risk_events = []
        
        for threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.CATASTROPHIC]:
            for event_id in self.events_by_threat_level[threat_level]:
                event = self.get_event(event_id)
                if event and event.timestamp >= cutoff_time:
                    high_risk_events.append(event)
        
        return sorted(high_risk_events, key=lambda e: e.timestamp, reverse=True)
    
    def _update_indexes(self, event: SecurityEvent, event_id: str) -> None:
        """Update all index structures for an event."""
        # Type index
        self.events_by_type[event.event_type].append(event_id)
        
        # Threat level index
        self.events_by_threat_level[event.threat_level].append(event_id)
        
        # User index
        if event.security_context and event.security_context.user_id:
            self.events_by_user[event.security_context.user_id].append(event_id)
        
        # IP index
        if event.security_context and event.security_context.source_ip:
            self.events_by_ip[event.security_context.source_ip].append(event_id)
        
        # Status index
        self.events_by_status[event.status].append(event_id)
        
        # Time index
        self.events_by_time.append((event.timestamp, event_id))
        
        # Tags index
        for tag in event.tags:
            self.events_by_tags[tag].append(event_id)
        
        # Incident index
        if event.incident_id:
            self.events_by_incident[event.incident_id].append(event_id)
    
    def _cache_event(self, event_id: str, event: SecurityEvent) -> None:
        """Cache event with LRU eviction."""
        self.event_cache[event_id] = event
        self.cache_access_times[event_id] = time.time()
        
        # Enforce cache size limit
        if len(self.event_cache) > self.cache_max_size:
            lru_id = min(self.cache_access_times.keys(), 
                        key=lambda k: self.cache_access_times[k])
            del self.event_cache[lru_id]
            del self.cache_access_times[lru_id]
    
    def _get_candidate_events(self, criteria: Dict[str, Any]) -> List[str]:
        """Get candidate event IDs based on primary search criteria."""
        if "threat_level" in criteria:
            threat_level = ThreatLevel[criteria["threat_level"].upper()]
            return self.events_by_threat_level[threat_level]
        elif "event_type" in criteria:
            event_type = SecurityEventType.from_code(criteria["event_type"])
            return self.events_by_type[event_type]
        elif "user_id" in criteria:
            return self.events_by_user[criteria["user_id"]]
        elif "source_ip" in criteria:
            return self.events_by_ip[criteria["source_ip"]]
        elif "tag" in criteria:
            return self.events_by_tags[criteria["tag"]]
        else:
            # Return all events if no specific index criteria
            return list(self.events.keys())
    
    def _matches_criteria(self, event: SecurityEvent, criteria: Dict[str, Any]) -> bool:
        """Check if event matches search criteria."""
        for key, value in criteria.items():
            if key == "start_time" and event.timestamp < value:
                return False
            elif key == "end_time" and event.timestamp > value:
                return False
            elif key == "min_threat_score" and event.threat_score < value:
                return False
            elif key == "status" and event.status != value:
                return False
            elif key == "investigation_required" and event.investigation_required != value:
                return False
            elif key == "tags" and not any(tag in event.tags for tag in value):
                return False
        
        return True
    
    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data."""
        try:
            # Generate or load encryption key
            key_file = "security_events.key"
            if os.path.exists(key_file):
                with open(key_file, "rb") as f:
                    self.encryption_key = f.read()
            else:
                password = os.environ.get("SECURITY_ENCRYPTION_PASSWORD", "mars-quantum-security-2025").encode()
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = key
                
                # Save key securely
                with open(key_file, "wb") as f:
                    f.write(self.encryption_key)
                os.chmod(key_file, 0o600)  # Restrict permissions
                
        except Exception as e:
            logger.error(f"Encryption setup failed: {e}")
            self.enable_encryption = False
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.encryption_key:
            return data
        try:
            f = Fernet(self.encryption_key)
            return f.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.encryption_key:
            return encrypted_data
        try:
            f = Fernet(self.encryption_key)
            return f.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def _setup_redis(self, redis_config: Optional[Dict[str, Any]]) -> None:
        """Setup Redis connection."""
        try:
            config = redis_config or {"host": "localhost", "port": 6379, "db": 2}
            self.redis_client = redis.Redis(**config, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for SecurityEventStore")
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}")
            self.redis_client = None
    
    def _setup_database(self) -> None:
        """Setup SQLite database for persistence."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    threat_level TEXT,
                    threat_score REAL,
                    title TEXT,
                    description TEXT,
                    user_id TEXT,
                    source_ip TEXT,
                    status TEXT,
                    investigation_required BOOLEAN,
                    incident_id TEXT,
                    event_data TEXT,
                    created_at REAL DEFAULT (datetime('now'))
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON security_events (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON security_events (event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_threat_level ON security_events (threat_level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON security_events (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_ip ON security_events (source_ip)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON security_events (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incident_id ON security_events (incident_id)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def _store_event_db(self, event: SecurityEvent) -> None:
        """Store event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            event_data = json.dumps(event.to_dict(), default=str)
            if self.enable_encryption:
                event_data = self._encrypt_data(event_data)
            
            cursor.execute("""
                INSERT OR REPLACE INTO security_events 
                (event_id, timestamp, event_type, threat_level, threat_score, 
                 title, description, user_id, source_ip, status, 
                 investigation_required, incident_id, event_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.event_type.code,
                event.threat_level.name,
                event.threat_score,
                event.title,
                event.description,
                event.security_context.user_id if event.security_context else None,
                event.security_context.source_ip if event.security_context else None,
                event.status,
                event.investigation_required,
                event.incident_id,
                event_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database storage failed for event {event.event_id}: {e}")
    
    def _get_event_db(self, event_id: str) -> Optional[SecurityEvent]:
        """Retrieve event from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT event_data FROM security_events WHERE event_id = ?", (event_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                event_data = result[0]
                if self.enable_encryption:
                    event_data = self._decrypt_data(event_data)
                event_dict = json.loads(event_data)
                return SecurityEvent.from_dict(event_dict)
                
        except Exception as e:
            logger.error(f"Database retrieval failed for event {event_id}: {e}")
        
        return None
    
    def _enforce_limits(self) -> None:
        """Enforce storage limits by removing old events."""
        if len(self.events) <= self.max_events:
            return
        
        # Remove oldest events
        events_to_remove = len(self.events) - self.max_events
        oldest_events = sorted(self.events_by_time)[:events_to_remove]
        
        for timestamp, event_id in oldest_events:
            self._remove_event(event_id)
    
    def _remove_event(self, event_id: str) -> None:
        """Remove event from all storage structures."""
        if event_id not in self.events:
            return
        
        event = self.events[event_id]
        
        # Remove from memory
        del self.events[event_id]
        
        # Remove from cache
        if event_id in self.event_cache:
            del self.event_cache[event_id]
        if event_id in self.cache_access_times:
            del self.cache_access_times[event_id]
        
        # Remove from indexes
        self._remove_from_indexes(event, event_id)
    
    def _remove_from_indexes(self, event: SecurityEvent, event_id: str) -> None:
        """Remove event from all index structures."""
        # Helper function to safely remove from index
        def safe_remove(index_list, event_id):
            if event_id in index_list:
                index_list.remove(event_id)
        
        # Remove from all indexes
        safe_remove(self.events_by_type[event.event_type], event_id)
        safe_remove(self.events_by_threat_level[event.threat_level], event_id)
        safe_remove(self.events_by_status[event.status], event_id)
        
        if event.security_context:
            if event.security_context.user_id:
                safe_remove(self.events_by_user[event.security_context.user_id], event_id)
            if event.security_context.source_ip:
                safe_remove(self.events_by_ip[event.security_context.source_ip], event_id)
        
        for tag in event.tags:
            safe_remove(self.events_by_tags[tag], event_id)
        
        if event.incident_id:
            safe_remove(self.events_by_incident[event.incident_id], event_id)
        
        # Remove from time index
        self.events_by_time = [(t, eid) for t, eid in self.events_by_time if eid != event_id]
    
    def _start_maintenance_thread(self) -> None:
        """Start background maintenance thread."""
        def maintenance_worker():
            while not self._shutdown_event.wait(3600):  # Run every hour
                try:
                    # Cleanup old events
                    self._cleanup_old_events()
                    
                    # Optimize indexes
                    self._optimize_indexes()
                    
                except Exception as e:
                    logger.error(f"Maintenance thread error: {e}")
        
        self._maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_thread.start()
    
    def _cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        retention_days = 90  # Default retention period
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        old_events = []
        for timestamp, event_id in self.events_by_time:
            if timestamp < cutoff_time:
                old_events.append(event_id)
            else:
                break  # Events are sorted by time
        
        for event_id in old_events:
            self._remove_event(event_id)
        
        if old_events:
            logger.info(f"Cleaned up {len(old_events)} old security events")
    
    def _optimize_indexes(self) -> None:
        """Optimize index structures for performance."""
        # Remove empty index entries
        for index_dict in [self.events_by_type, self.events_by_threat_level, 
                          self.events_by_user, self.events_by_ip, self.events_by_status,
                          self.events_by_tags, self.events_by_incident]:
            empty_keys = [k for k, v in index_dict.items() if not v]
            for k in empty_keys:
                del index_dict[k]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the event store."""
        with self._lock:
            return {
                "storage_stats": {
                    "total_events": len(self.events),
                    "max_events": self.max_events,
                    "cache_size": len(self.event_cache),
                    "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                },
                "threat_level_distribution": dict(self.stats["events_by_threat_level"]),
                "event_type_distribution": dict(self.stats["events_by_type"]),
                "performance_stats": self.stats,
                "configuration": {
                    "encryption_enabled": self.enable_encryption,
                    "redis_enabled": self.redis_client is not None,
                    "database_path": self.database_path
                }
            }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the event store."""
        self._shutdown_event.set()
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass


class SecurityEventLogger:
    """
    Enterprise security event logging system with advanced features.
    
    Provides comprehensive security event logging with threat correlation,
    automated response triggers, and compliance reporting capabilities.
    """
    
    def __init__(self,
                 event_store: Optional[SecurityEventStore] = None,
                 enable_geoip: bool = True,
                 geoip_database_path: Optional[str] = None,
                 enable_threat_correlation: bool = True,
                 correlation_window_seconds: int = 300,
                 auto_escalation_enabled: bool = True):
        """
        Initialize the security event logger.
        
        Args:
            event_store: Optional external event store
            enable_geoip: Whether to enable GeoIP lookups
            geoip_database_path: Path to GeoIP database
            enable_threat_correlation: Whether to enable threat correlation
            correlation_window_seconds: Time window for correlating events
            auto_escalation_enabled: Whether to auto-escalate related threats
        """
        self.event_store = event_store or SecurityEventStore()
        self.enable_geoip = enable_geoip and GEOIP_AVAILABLE
        self.geoip_database_path = geoip_database_path
        self.enable_threat_correlation = enable_threat_correlation
        self.correlation_window_seconds = correlation_window_seconds
        self.auto_escalation_enabled = auto_escalation_enabled
        
        # GeoIP setup
        self.geoip_reader = None
        if self.enable_geoip:
            self._setup_geoip()
        
        # Threat correlation
        self.threat_patterns: Dict[str, List[str]] = {}
        self.correlation_cache: Dict[str, List[str]] = {}
        
        # Event processors
        self.event_processors: List[Callable[[SecurityEvent], None]] = []
        
        # Response handlers
        self.response_handlers: Dict[ThreatLevel, List[Callable[[SecurityEvent], None]]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_correlated": 0,
            "threats_escalated": 0,
            "automated_responses": 0,
            "geoip_lookups": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup default processors
        self._setup_default_processors()
    
    def log_security_event(self,
                          event_type: SecurityEventType,
                          description: str,
                          threat_level: Optional[ThreatLevel] = None,
                          security_context: Optional[SecurityContext] = None,
                          payload: Optional[Dict[str, Any]] = None,
                          evidence: Optional[Dict[str, Any]] = None,
                          **kwargs) -> SecurityEvent:
        """
        Log a comprehensive security event.
        
        Args:
            event_type: Type of security event
            description: Event description
            threat_level: Optional threat level (auto-determined if not provided)
            security_context: Security context information
            payload: Event payload data
            evidence: Evidence data
            **kwargs: Additional event properties
            
        Returns:
            SecurityEvent: The created and stored security event
        """
        with self._lock:
            # Create security context if not provided
            if security_context is None:
                security_context = self._create_default_context()
            
            # Enhance context with GeoIP if enabled
            if self.enable_geoip and security_context.source_ip:
                self._enhance_with_geoip(security_context)
            
            # Determine threat level if not provided
            if threat_level is None:
                threat_level = event_type.get_default_threat_level()
            
            # Create the security event
            event = SecurityEvent(
                event_type=event_type,
                description=description,
                threat_level=threat_level,
                security_context=security_context,
                payload=payload or {},
                evidence=evidence or {},
                **kwargs
            )
            
            # Process the event through processors
            for processor in self.event_processors:
                try:
                    processor(event)
                except Exception as e:
                    logger.error(f"Event processor failed: {e}")
            
            # Perform threat correlation
            if self.enable_threat_correlation:
                self._correlate_threats(event)
            
            # Store the event
            event_id = self.event_store.store_event(event)
            
            # Trigger automated responses
            self._trigger_automated_responses(event)
            
            # Update statistics
            self.stats["events_logged"] += 1
            
            logger.info(f"Security event logged: {event_id} - {event_type.code} - {threat_level.name}")
            
            return event
    
    def create_security_event(self,
                            event_type: str,
                            description: str,
                            threat_level: ThreatLevel,
                            source_ip: Optional[str] = None,
                            user_id: Optional[str] = None,
                            payload: Optional[Dict] = None,
                            request_id: Optional[str] = None) -> SecurityEvent:
        """
        Create a security event (legacy method for backward compatibility).
        
        This method maintains compatibility with the original simple interface
        while providing access to the enhanced logging capabilities.
        """
        # Convert string event_type to SecurityEventType
        if isinstance(event_type, str):
            # Try to find matching event type by code or create a generic one
            security_event_type = SecurityEventType.COMPLIANCE_VIOLATION
            for event_enum in SecurityEventType:
                if event_enum.code == event_type or event_enum.name.lower() == event_type.lower():
                    security_event_type = event_enum
                    break
        else:
            security_event_type = event_type
        
        # Create security context
        security_context = SecurityContext(
            source_ip=source_ip,
            user_id=user_id,
            username=user_id or os.environ.get("USER_LOGIN", "Shriram-2005"),
            request_id=request_id
        )
        
        # Log the event using the comprehensive method
        return self.log_security_event(
            event_type=security_event_type,
            description=description,
            threat_level=threat_level,
            security_context=security_context,
            payload=payload
        )
    
    def log_authentication_event(self,
                                success: bool,
                                user_id: str,
                                source_ip: str,
                                authentication_method: str = "password",
                                additional_context: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log authentication-related security events."""
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE
        threat_level = ThreatLevel.INFORMATIONAL if success else ThreatLevel.MEDIUM
        
        security_context = SecurityContext(
            user_id=user_id,
            username=user_id,
            source_ip=source_ip,
            authentication_method=authentication_method
        )
        
        description = f"User {user_id} authentication {'succeeded' if success else 'failed'} from {source_ip}"
        
        return self.log_security_event(
            event_type=event_type,
            description=description,
            threat_level=threat_level,
            security_context=security_context,
            payload=additional_context or {}
        )
    
    def log_data_access_event(self,
                             user_id: str,
                             resource: str,
                             action: str,
                             success: bool,
                             data_classification: str = "internal",
                             additional_context: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log data access security events."""
        event_type = SecurityEventType.UNAUTHORIZED_ACCESS if not success else SecurityEventType.LOGIN_SUCCESS
        threat_level = ThreatLevel.HIGH if not success else ThreatLevel.LOW
        
        if data_classification in ["confidential", "secret", "top_secret"]:
            threat_level = threat_level.escalate()
        
        security_context = SecurityContext(user_id=user_id, username=user_id)
        
        description = f"User {user_id} {action} access to {resource} - {'Success' if success else 'Denied'}"
        
        event = self.log_security_event(
            event_type=event_type,
            description=description,
            threat_level=threat_level,
            security_context=security_context,
            payload=additional_context or {},
            data_classification=data_classification
        )
        
        # Add compliance frameworks based on data classification
        if data_classification in ["pii", "personal"]:
            event.compliance_frameworks.add(ComplianceFramework.GDPR)
        if data_classification in ["financial", "payment"]:
            event.compliance_frameworks.add(ComplianceFramework.PCI_DSS)
        if data_classification in ["healthcare", "medical"]:
            event.compliance_frameworks.add(ComplianceFramework.HIPAA)
        
        return event
    
    def log_network_event(self,
                         event_type: SecurityEventType,
                         source_ip: str,
                         destination_ip: Optional[str] = None,
                         port: Optional[int] = None,
                         protocol: str = "TCP",
                         additional_context: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log network-related security events."""
        security_context = SecurityContext(
            source_ip=source_ip,
            destination_ip=destination_ip,
            destination_port=port,
            protocol=protocol
        )
        
        description = f"Network event from {source_ip}"
        if destination_ip:
            description += f" to {destination_ip}"
        if port:
            description += f":{port}"
        
        return self.log_security_event(
            event_type=event_type,
            description=description,
            security_context=security_context,
            payload=additional_context or {}
        )
    
    def log_application_security_event(self,
                                     event_type: SecurityEventType,
                                     application_name: str,
                                     user_id: Optional[str] = None,
                                     request_details: Optional[Dict[str, Any]] = None,
                                     additional_context: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log application security events."""
        security_context = SecurityContext(
            user_id=user_id,
            application_name=application_name
        )
        
        if request_details:
            security_context.request_method = request_details.get("method")
            security_context.request_uri = request_details.get("uri")
            security_context.user_agent = request_details.get("user_agent")
        
        description = f"Application security event in {application_name}"
        if user_id:
            description += f" for user {user_id}"
        
        return self.log_security_event(
            event_type=event_type,
            description=description,
            security_context=security_context,
            payload=additional_context or {}
        )
    
    def _create_default_context(self) -> SecurityContext:
        """Create default security context."""
        return SecurityContext(
            username=os.environ.get("USER_LOGIN", "Shriram-2005"),
            hostname=socket.gethostname(),
            process_id=os.getpid()
        )
    
    def _enhance_with_geoip(self, security_context: SecurityContext) -> None:
        """Enhance security context with GeoIP information."""
        if not self.geoip_reader or not security_context.source_ip:
            return
        
        try:
            response = self.geoip_reader.city(security_context.source_ip)
            security_context.country = response.country.name
            security_context.region = response.subdivisions.most_specific.name
            security_context.city = response.city.name
            security_context.latitude = float(response.location.latitude) if response.location.latitude else None
            security_context.longitude = float(response.location.longitude) if response.location.longitude else None
            security_context.timezone = response.location.time_zone
            
            self.stats["geoip_lookups"] += 1
            
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {security_context.source_ip}: {e}")
    
    def _setup_geoip(self) -> None:
        """Setup GeoIP database reader."""
        if not GEOIP_AVAILABLE:
            return
        
        try:
            # Try to find GeoIP database
            possible_paths = [
                self.geoip_database_path,
                "/usr/share/GeoIP/GeoLite2-City.mmdb",
                "/var/lib/GeoIP/GeoLite2-City.mmdb",
                "GeoLite2-City.mmdb"
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    self.geoip_reader = geoip2.database.Reader(path)
                    logger.info(f"GeoIP database loaded from {path}")
                    return
            
            logger.warning("GeoIP database not found, geographic context will be limited")
            self.enable_geoip = False
            
        except Exception as e:
            logger.error(f"GeoIP setup failed: {e}")
            self.enable_geoip = False
    
    def _correlate_threats(self, event: SecurityEvent) -> None:
        """Perform threat correlation analysis."""
        try:
            correlation_key = f"{event.event_type.code}:{event.security_context.source_ip if event.security_context else 'unknown'}"
            current_time = time.time()
            
            # Check for related events in the correlation window
            recent_events = self.event_store.get_recent_events(
                minutes=self.correlation_window_seconds // 60
            )
            
            related_events = []
            for recent_event in recent_events:
                if (recent_event.event_id != event.event_id and
                    self._are_events_related(event, recent_event)):
                    related_events.append(recent_event.event_id)
            
            if related_events:
                # Update event with correlations
                event.related_event_ids.extend(related_events)
                
                # Auto-escalate if multiple related threats
                if len(related_events) >= 3 and self.auto_escalation_enabled:
                    event.escalate_threat_level()
                    self.stats["threats_escalated"] += 1
                
                self.stats["events_correlated"] += 1
                
        except Exception as e:
            logger.error(f"Threat correlation failed: {e}")
    
    def _are_events_related(self, event1: SecurityEvent, event2: SecurityEvent) -> bool:
        """Determine if two events are related."""
        # Same source IP
        if (event1.security_context and event2.security_context and
            event1.security_context.source_ip == event2.security_context.source_ip):
            return True
        
        # Same user
        if (event1.security_context and event2.security_context and
            event1.security_context.user_id == event2.security_context.user_id):
            return True
        
        # Similar event types
        if event1.event_type.code.split('.')[0] == event2.event_type.code.split('.')[0]:
            return True
        
        return False
    
    def _trigger_automated_responses(self, event: SecurityEvent) -> None:
        """Trigger automated responses based on threat level."""
        try:
            handlers = self.response_handlers.get(event.threat_level, [])
            for handler in handlers:
                try:
                    handler(event)
                    event.trigger_automated_response(handler.__name__)
                except Exception as e:
                    logger.error(f"Response handler {handler.__name__} failed: {e}")
            
            if handlers:
                self.stats["automated_responses"] += 1
                
        except Exception as e:
            logger.error(f"Automated response trigger failed: {e}")
    
    def _setup_default_processors(self) -> None:
        """Setup default event processors."""
        self.add_event_processor(self._enrich_with_system_info)
        self.add_event_processor(self._detect_anomalies)
    
    def _enrich_with_system_info(self, event: SecurityEvent) -> None:
        """Enrich event with system information."""
        if PSUTIL_AVAILABLE and event.security_context:
            try:
                # Add system performance metrics
                event.custom_fields["cpu_percent"] = psutil.cpu_percent()
                event.custom_fields["memory_percent"] = psutil.virtual_memory().percent
                event.custom_fields["disk_usage"] = psutil.disk_usage('/').percent
                
                # Add process information if available
                if event.security_context.process_id:
                    try:
                        process = psutil.Process(event.security_context.process_id)
                        event.custom_fields["process_cpu_percent"] = process.cpu_percent()
                        event.custom_fields["process_memory_mb"] = process.memory_info().rss / 1024 / 1024
                        event.security_context.process_name = process.name()
                        event.security_context.parent_process_id = process.ppid()
                    except psutil.NoSuchProcess:
                        pass
                        
            except Exception as e:
                logger.debug(f"System info enrichment failed: {e}")
    
    def _detect_anomalies(self, event: SecurityEvent) -> None:
        """Detect anomalous patterns in events."""
        try:
            # Simple anomaly detection based on recent event patterns
            if event.security_context and event.security_context.source_ip:
                recent_events = self.event_store.get_events_by_ip(
                    event.security_context.source_ip, limit=10
                )
                
                # Check for rapid event generation
                if len(recent_events) >= 5:
                    time_window = recent_events[-1].timestamp - recent_events[0].timestamp
                    if time_window < 60:  # 5+ events in 1 minute
                        event.tags.add("rapid_events")
                        if event.threat_level.score < ThreatLevel.HIGH.score:
                            event.escalate_threat_level()
                
                # Check for geographic anomalies
                if len(recent_events) >= 2:
                    countries = set()
                    for recent_event in recent_events:
                        if (recent_event.security_context and 
                            recent_event.security_context.country):
                            countries.add(recent_event.security_context.country)
                    
                    if len(countries) > 2:  # Events from multiple countries
                        event.tags.add("geographic_anomaly")
                        event.investigation_required = True
                        
        except Exception as e:
            logger.debug(f"Anomaly detection failed: {e}")
    
    def add_event_processor(self, processor: Callable[[SecurityEvent], None]) -> None:
        """Add a custom event processor."""
        self.event_processors.append(processor)
    
    def add_response_handler(self, threat_level: ThreatLevel, 
                           handler: Callable[[SecurityEvent], None]) -> None:
        """Add a response handler for specific threat levels."""
        self.response_handlers[threat_level].append(handler)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the security logger."""
        store_stats = self.event_store.get_comprehensive_stats()
        
        return {
            "logger_stats": self.stats,
            "store_stats": store_stats,
            "configuration": {
                "geoip_enabled": self.enable_geoip,
                "threat_correlation_enabled": self.enable_threat_correlation,
                "auto_escalation_enabled": self.auto_escalation_enabled,
                "correlation_window_seconds": self.correlation_window_seconds
            }
        }


# Legacy compatibility function
def _create_security_event(event_type: str, description: str, 
                         threat_level: ThreatLevel, source_ip: str = None,
                         user_id: str = None, payload: Dict = None,
                         request_id: str = None) -> SecurityEvent:
    """
    Legacy security event creation function for backward compatibility.
    
    This function maintains the original interface while providing access
    to the enhanced security event logging capabilities.
    """
    # Create a global security logger instance if not exists
    global _global_security_logger
    if '_global_security_logger' not in globals():
        _global_security_logger = SecurityEventLogger()
    
    return _global_security_logger.create_security_event(
        event_type=event_type,
        description=description,
        threat_level=threat_level,
        source_ip=source_ip,
        user_id=user_id,
        payload=payload,
        request_id=request_id
    )


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the security event logging system
    logger_instance = SecurityEventLogger()
    
    # Example 1: Log a basic security event
    event1 = logger_instance.log_security_event(
        event_type=SecurityEventType.LOGIN_FAILURE,
        description="Failed login attempt from suspicious IP",
        threat_level=ThreatLevel.MEDIUM,
        security_context=SecurityContext(
            user_id="test_user",
            source_ip="192.168.1.100"
        )
    )
    
    # Example 2: Log an authentication event
    event2 = logger_instance.log_authentication_event(
        success=False,
        user_id="admin",
        source_ip="10.0.0.1",
        authentication_method="password"
    )
    
    # Example 3: Log a data access event
    event3 = logger_instance.log_data_access_event(
        user_id="user123",
        resource="/sensitive/financial_data.csv",
        action="read",
        success=False,
        data_classification="confidential"
    )
    
    # Example 4: Log a network security event
    event4 = logger_instance.log_network_event(
        event_type=SecurityEventType.PORT_SCAN,
        source_ip="192.168.1.200",
        destination_ip="10.0.0.10",
        port=22,
        protocol="TCP"
    )
    
    # Print comprehensive statistics
    stats = logger_instance.get_comprehensive_stats()
    print(f"Security Events Logged: {stats['logger_stats']['events_logged']}")
    print(f"Threat Correlations: {stats['logger_stats']['events_correlated']}")
    print(f"Total Events Stored: {stats['store_stats']['storage_stats']['total_events']}")
    
    # Demonstrate event retrieval
    recent_events = logger_instance.event_store.get_recent_events(minutes=60)
    print(f"Recent Events (last hour): {len(recent_events)}")
    
    high_risk_events = logger_instance.event_store.get_high_risk_events(hours=24)
    print(f"High Risk Events (last 24h): {len(high_risk_events)}")
    
    print("Enterprise Security Event Logging System initialized successfully!")