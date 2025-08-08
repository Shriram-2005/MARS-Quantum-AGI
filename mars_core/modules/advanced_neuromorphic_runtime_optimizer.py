"""
MARS Advanced Neuromorphic Runtime Optimizer
============================================

A sophisticated brain-inspired computational resource optimization system that implements
neuromorphic computing principles for intelligent task scheduling and resource allocation.

This module provides advanced capabilities including:
- Brain-inspired neural network optimization algorithms
- Hebbian learning for adaptive resource allocation
- Real-time system performance monitoring and optimization
- Multi-objective optimization with dynamic target switching
- Neuromorphic processing unit integration and management
- Quantum-classical hybrid execution strategies
- Self-adapting computational workload distribution

The system uses biological neural network principles such as:
- Hebbian plasticity for learning optimal resource allocations
- Spike-timing dependent plasticity for temporal optimization
- Neural homeostasis for system stability maintenance
- Distributed processing inspired by cortical architecture

"""

import numpy as np
import threading
import time
import logging
import uuid
import heapq
import os
import json
import random
import math
import platform
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from datetime import datetime
from collections import deque
import psutil
import hashlib

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARS.Neuromorphic")

# Module metadata
__version__ = "2.0.0"
__author__ = "Shriram-2005"
__license__ = "MIT"
__description__ = "Advanced Neuromorphic Runtime Optimizer with Brain-Inspired Algorithms"

class OptimizationTarget(Enum):
    """
    Enumeration of optimization targets for the neuromorphic system.
    
    These targets define the primary optimization objectives that the
    neuromorphic optimizer can focus on during runtime optimization.
    """
    THROUGHPUT = auto()      # Maximize throughput/operations per second
    LATENCY = auto()         # Minimize response latency and wait times
    MEMORY = auto()          # Minimize memory usage and allocation overhead
    ENERGY = auto()          # Minimize energy consumption across all resources
    RELIABILITY = auto()     # Maximize system reliability and fault tolerance
    ADAPTABILITY = auto()    # Maximize adaptation to changing workload conditions
    BALANCED = auto()        # Balance optimization across all available metrics
    CUSTOM = auto()          # Custom user-defined optimization target


class ExecutionMode(Enum):
    """
    Enumeration of execution modes for computational tasks.
    
    These modes determine how tasks are executed across the available
    computational resources in the neuromorphic system.
    """
    SEQUENTIAL = auto()      # Sequential execution (one task at a time)
    PARALLEL = auto()        # Parallel execution (multiple tasks simultaneously)
    DISTRIBUTED = auto()     # Distributed execution across multiple nodes
    ASYNCHRONOUS = auto()    # Asynchronous execution with event-driven completion
    SPECULATIVE = auto()     # Speculative execution with rollback capability
    QUANTUM = auto()         # Quantum-accelerated execution on QPUs
    NEUROMORPHIC = auto()    # Neuromorphic execution using spiking neural networks
    HYBRID = auto()          # Hybrid execution combining multiple modes


class ProcessingUnit(Enum):
    """
    Enumeration of processing unit types available for computation.
    
    These represent the different types of computational hardware that
    can be utilized by the neuromorphic optimization system.
    """
    CPU = auto()             # Central Processing Unit (general-purpose)
    GPU = auto()             # Graphics Processing Unit (parallel computation)
    TPU = auto()             # Tensor Processing Unit (AI/ML acceleration)
    QPU = auto()             # Quantum Processing Unit (quantum computation)
    NPU = auto()             # Neuromorphic Processing Unit (brain-inspired)
    FPGA = auto()            # Field-Programmable Gate Array (reconfigurable)
    ASIC = auto()            # Application-Specific Integrated Circuit (specialized)
    HYBRID = auto()          # Hybrid processing unit (multiple technologies)


class ResourceState(Enum):
    """
    Enumeration of possible states for computational resources.
    
    These states represent the current operational condition and
    availability of resources in the system.
    """
    IDLE = auto()            # Resource is idle and available for allocation
    BUSY = auto()            # Resource is busy executing tasks
    OVERLOADED = auto()      # Resource is overloaded beyond optimal capacity
    DEGRADED = auto()        # Resource is operating with reduced performance
    FAILING = auto()         # Resource is experiencing failures or errors
    OFFLINE = auto()         # Resource is offline and unavailable
    RESERVED = auto()        # Resource is reserved for specific tasks
    INITIALIZING = auto()    # Resource is initializing and not yet ready

@dataclass
class ComputationalResource:
    """
    Represents a computational resource in the neuromorphic system.
    
    This class encapsulates a computational resource with multi-dimensional capacity
    and load tracking. Resources can have multiple features (computational, memory,
    bandwidth, etc.) with independent capacity and utilization metrics.
    
    Attributes:
        resource_id: Unique identifier for the resource
        resource_type: Type of processing unit (CPU, GPU, TPU, etc.)
        capacity: Dictionary mapping feature names to maximum capacity values
        current_load: Dictionary mapping feature names to current load values
        state: Current operational state of the resource
        location: Optional physical or logical location identifier
        capabilities: Set of supported capabilities/features
        metadata: Additional metadata and configuration parameters
        last_updated: Timestamp of the last status update
    
    Example:
        >>> resource = ComputationalResource(
        ...     resource_type=ProcessingUnit.GPU,
        ...     capacity={"computational": 1000.0, "memory": 16.0},
        ...     capabilities={"tensor_ops", "parallel_compute"}
        ... )
        >>> resource.update_load("computational", 250.0)
        >>> print(f"GPU utilization: {resource.get_utilization('computational'):.2f}")
    """
    
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ProcessingUnit = ProcessingUnit.CPU
    capacity: Dict[str, float] = field(default_factory=dict)  # Feature -> capacity mapping
    current_load: Dict[str, float] = field(default_factory=dict)  # Feature -> load mapping
    state: ResourceState = ResourceState.IDLE
    location: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self) -> None:
        """
        Validates resource parameters and initializes default values.
        
        Raises:
            ValueError: If capacity or load values are invalid
            TypeError: If parameters have incorrect types
        """
        try:
            # Validate resource_type
            if not isinstance(self.resource_type, ProcessingUnit):
                raise TypeError(f"resource_type must be ProcessingUnit enum, got {type(self.resource_type)}")
            
            # Validate state
            if not isinstance(self.state, ResourceState):
                raise TypeError(f"state must be ResourceState enum, got {type(self.state)}")
            
            # Validate and initialize capacity
            if not isinstance(self.capacity, dict):
                raise TypeError(f"capacity must be a dictionary, got {type(self.capacity)}")
            
            for feature, cap in self.capacity.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Capacity feature names must be non-empty strings, got {feature}")
                if not isinstance(cap, (int, float)) or cap < 0:
                    raise ValueError(f"Capacity values must be non-negative numbers, got {cap} for {feature}")
            
            # Validate and initialize current_load
            if not isinstance(self.current_load, dict):
                raise TypeError(f"current_load must be a dictionary, got {type(self.current_load)}")
            
            for feature, load in self.current_load.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Load feature names must be non-empty strings, got {feature}")
                if not isinstance(load, (int, float)) or load < 0:
                    raise ValueError(f"Load values must be non-negative numbers, got {load} for {feature}")
            
            # Initialize missing current_load entries
            for feature in self.capacity:
                if feature not in self.current_load:
                    self.current_load[feature] = 0.0
            
            # Validate capabilities
            if not isinstance(self.capabilities, set):
                self.capabilities = set(self.capabilities) if self.capabilities else set()
            
            # Initialize metadata
            if self.metadata is None:
                self.metadata = {}
            
            # Add creation timestamp
            if 'created_at' not in self.metadata:
                self.metadata['created_at'] = time.time()
            
            # Set default capacities for common features if not provided
            if not self.capacity:
                default_capacities = {
                    ProcessingUnit.CPU: {"computational": 100.0, "memory": 8.0, "threads": 8.0},
                    ProcessingUnit.GPU: {"computational": 1000.0, "memory": 16.0, "cuda_cores": 2048.0},
                    ProcessingUnit.TPU: {"computational": 2000.0, "memory": 32.0, "tensor_ops": 1000.0},
                    ProcessingUnit.QPU: {"computational": 100.0, "qubits": 50.0, "coherence_time": 100.0},
                    ProcessingUnit.NPU: {"computational": 500.0, "synapses": 10000.0, "neurons": 1000.0}
                }
                
                if self.resource_type in default_capacities:
                    self.capacity = default_capacities[self.resource_type].copy()
                    # Initialize corresponding load values
                    for feature in self.capacity:
                        if feature not in self.current_load:
                            self.current_load[feature] = 0.0
            
            logger.debug(f"Initialized computational resource {self.resource_id}: "
                        f"{self.resource_type.name} with {len(self.capacity)} features")
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize ComputationalResource: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ComputationalResource initialization: {e}")
            raise RuntimeError(f"Resource initialization failed: {e}") from e
    
    def get_utilization(self, feature: str = "computational") -> float:
        """
        Get the utilization ratio for a specific feature.
        
        Args:
            feature: The feature name to check utilization for
        
        Returns:
            float: Utilization ratio (0.0 to 1.0+)
        
        Raises:
            ValueError: If feature name is invalid
        
        Example:
            >>> resource.get_utilization("memory")
            0.75  # 75% memory utilization
        """
        try:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError(f"Feature name must be a non-empty string, got {feature}")
            
            if feature in self.capacity and self.capacity[feature] > 0:
                utilization = self.current_load.get(feature, 0) / self.capacity[feature]
                logger.debug(f"Resource {self.resource_id} {feature} utilization: {utilization:.3f}")
                return utilization
            
            logger.warning(f"Feature '{feature}' not found in resource {self.resource_id} capacity")
            return 0.0
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error calculating utilization for feature '{feature}': {e}")
            return 0.0
    
    def get_available_capacity(self, feature: str = "computational") -> float:
        """
        Get the available capacity for a specific feature.
        
        Args:
            feature: The feature name to check available capacity for
        
        Returns:
            float: Available capacity (capacity - current_load)
        
        Raises:
            ValueError: If feature name is invalid
        
        Example:
            >>> resource.get_available_capacity("computational")
            250.0  # 250 units of computational capacity available
        """
        try:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError(f"Feature name must be a non-empty string, got {feature}")
            
            if feature in self.capacity:
                available = self.capacity[feature] - self.current_load.get(feature, 0)
                logger.debug(f"Resource {self.resource_id} {feature} available capacity: {available:.2f}")
                return max(0.0, available)  # Ensure non-negative
            
            logger.warning(f"Feature '{feature}' not found in resource {self.resource_id} capacity")
            return 0.0
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error calculating available capacity for feature '{feature}': {e}")
            return 0.0
    
    def update_load(self, feature: str, load_delta: float) -> bool:
        """
        Update the load for a specific feature.
        
        This method updates the current load for a given feature by adding the
        load_delta value. It includes capacity checking and automatic state updates.
        
        Args:
            feature: The feature name to update load for
            load_delta: The change in load (can be positive or negative)
        
        Returns:
            bool: True if update was successful, False if it would exceed capacity
        
        Raises:
            ValueError: If feature name is invalid or load_delta is not a number
            TypeError: If parameters have incorrect types
        
        Example:
            >>> resource = ComputationalResource(capacity={"cpu": 100.0})
            >>> success = resource.update_load("cpu", 25.0)  # Add 25 units of load
            >>> if success:
            ...     print(f"Load updated. Current utilization: {resource.get_utilization('cpu'):.2f}")
        """
        try:
            # Validate inputs
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError(f"Feature name must be a non-empty string, got {feature}")
            
            if not isinstance(load_delta, (int, float)):
                raise TypeError(f"load_delta must be a number, got {type(load_delta)}")
            
            # Check if feature exists in capacity
            if feature not in self.capacity:
                logger.warning(f"Feature '{feature}' not found in resource {self.resource_id} capacity")
                return False
            
            # Calculate new load
            current = self.current_load.get(feature, 0.0)
            new_load = max(0.0, current + load_delta)
            
            # Check capacity constraints (allow 10% overload for flexibility)
            max_allowed_load = self.capacity[feature] * 1.1
            if new_load > max_allowed_load:
                logger.warning(f"Load update rejected for resource {self.resource_id}: "
                             f"new load {new_load:.2f} exceeds max allowed {max_allowed_load:.2f} "
                             f"for feature '{feature}'")
                return False
            
            # Update the load
            old_load = self.current_load.get(feature, 0.0)
            self.current_load[feature] = new_load
            self.last_updated = time.time()
            
            # Update resource state based on overall utilization
            self._update_state_from_utilization()
            
            logger.debug(f"Updated load for resource {self.resource_id} feature '{feature}': "
                        f"{old_load:.2f} -> {new_load:.2f} (delta: {load_delta:+.2f})")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to update load for resource {self.resource_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating load for resource {self.resource_id}: {e}")
            return False
    
    def _update_state_from_utilization(self) -> None:
        """
        Update resource state based on current utilization levels.
        
        This internal method calculates the maximum utilization across all features
        and updates the resource state accordingly.
        """
        try:
            if not self.capacity:
                self.state = ResourceState.IDLE
                return
            
            # Calculate maximum utilization across all features
            max_utilization = 0.0
            for feature in self.capacity:
                if self.capacity[feature] > 0:
                    utilization = self.get_utilization(feature)
                    max_utilization = max(max_utilization, utilization)
            
            # Update state based on maximum utilization
            old_state = self.state
            if max_utilization > 1.0:  # Over capacity
                self.state = ResourceState.OVERLOADED
            elif max_utilization > 0.9:  # High utilization
                self.state = ResourceState.BUSY
            elif max_utilization > 0.1:  # Some utilization
                self.state = ResourceState.BUSY
            else:  # Low or no utilization
                self.state = ResourceState.IDLE
            
            if old_state != self.state:
                logger.debug(f"Resource {self.resource_id} state changed: "
                           f"{old_state.name} -> {self.state.name} "
                           f"(max utilization: {max_utilization:.3f})")
            
        except Exception as e:
            logger.error(f"Error updating state for resource {self.resource_id}: {e}")
            # Keep current state on error
    
    def is_overloaded(self, threshold: float = 0.9) -> bool:
        """
        Check if the resource is overloaded beyond a threshold.
        
        Args:
            threshold: Utilization threshold (0.0 to 1.0) above which resource is overloaded
        
        Returns:
            bool: True if any feature exceeds the threshold, False otherwise
        
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        
        Example:
            >>> if resource.is_overloaded(0.85):
            ...     print("Resource is overloaded!")
        """
        try:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
            
            for feature in self.capacity:
                if self.get_utilization(feature) > threshold:
                    logger.debug(f"Resource {self.resource_id} is overloaded on feature '{feature}': "
                               f"{self.get_utilization(feature):.3f} > {threshold:.3f}")
                    return True
            
            return False
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error checking overload status for resource {self.resource_id}: {e}")
            return False
    
    def can_handle_task(self, requirements: Dict[str, float]) -> bool:
        """
        Check if the resource can handle a task with given requirements.
        
        Args:
            requirements: Dictionary of feature requirements
        
        Returns:
            bool: True if resource can handle the task, False otherwise
        
        Raises:
            ValueError: If requirements are invalid
        
        Example:
            >>> task_reqs = {"computational": 50.0, "memory": 4.0}
            >>> if resource.can_handle_task(task_reqs):
            ...     print("Resource can handle this task")
        """
        try:
            if not isinstance(requirements, dict):
                raise ValueError(f"Requirements must be a dictionary, got {type(requirements)}")
            
            # Check each requirement
            for feature, required in requirements.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Requirement feature names must be non-empty strings, got {feature}")
                
                if not isinstance(required, (int, float)) or required < 0:
                    raise ValueError(f"Requirement values must be non-negative numbers, got {required}")
                
                # Check if we have this feature
                if feature not in self.capacity:
                    logger.debug(f"Resource {self.resource_id} lacks feature '{feature}'")
                    return False
                
                # Check if we have enough available capacity
                available = self.get_available_capacity(feature)
                if available < required:
                    logger.debug(f"Resource {self.resource_id} insufficient capacity for '{feature}': "
                               f"required {required:.2f}, available {available:.2f}")
                    return False
            
            # Check if resource is in a usable state
            if self.state in {ResourceState.OFFLINE, ResourceState.FAILING}:
                logger.debug(f"Resource {self.resource_id} not available due to state: {self.state.name}")
                return False
            
            logger.debug(f"Resource {self.resource_id} can handle task requirements")
            return True
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error checking task capability for resource {self.resource_id}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert resource to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing all resource information
        
        Example:
            >>> resource_dict = resource.to_dict()
            >>> print(f"Resource type: {resource_dict['resource_type']}")
        """
        try:
            utilization_metrics = {}
            performance_metrics = {}
            
            # Calculate utilization for each feature
            for feature in self.capacity:
                utilization_metrics[feature] = self.get_utilization(feature)
            
            # Calculate performance metrics
            total_capacity = sum(self.capacity.values())
            total_load = sum(self.current_load.values())
            overall_utilization = total_load / total_capacity if total_capacity > 0 else 0.0
            
            # Calculate efficiency (work done per unit capacity)
            efficiency = total_load / max(total_capacity, 1.0)
            
            # Calculate availability score
            availability_score = 1.0
            if self.state == ResourceState.OFFLINE:
                availability_score = 0.0
            elif self.state == ResourceState.FAILING:
                availability_score = 0.2
            elif self.state == ResourceState.DEGRADED:
                availability_score = 0.6
            elif self.state == ResourceState.OVERLOADED:
                availability_score = 0.8
            
            performance_metrics = {
                "overall_utilization": overall_utilization,
                "efficiency": efficiency,
                "availability_score": availability_score,
                "total_capacity": total_capacity,
                "total_load": total_load,
                "uptime": time.time() - self.metadata.get('created_at', time.time())
            }
            
            resource_dict = {
                "resource_id": self.resource_id,
                "resource_type": self.resource_type.name,
                "capacity": self.capacity.copy(),
                "current_load": self.current_load.copy(),
                "state": self.state.name,
                "location": self.location,
                "capabilities": list(self.capabilities),
                "metadata": self.metadata.copy(),
                "last_updated": self.last_updated,
                "utilization": utilization_metrics,
                "performance": performance_metrics
            }
            
            logger.debug(f"Generated dictionary representation for resource {self.resource_id}")
            return resource_dict
            
        except Exception as e:
            logger.error(f"Error converting resource {self.resource_id} to dictionary: {e}")
            # Return minimal representation on error
            return {
                "resource_id": self.resource_id,
                "resource_type": self.resource_type.name,
                "state": self.state.name,
                "error": str(e)
            }
    
    def reset_load(self, feature: Optional[str] = None) -> None:
        """
        Reset load for a specific feature or all features.
        
        Args:
            feature: Specific feature to reset, or None to reset all features
        
        Raises:
            ValueError: If feature name is invalid
        
        Example:
            >>> resource.reset_load("memory")  # Reset memory load only
            >>> resource.reset_load()         # Reset all loads
        """
        try:
            if feature is not None:
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Feature name must be a non-empty string, got {feature}")
                
                if feature in self.current_load:
                    old_load = self.current_load[feature]
                    self.current_load[feature] = 0.0
                    logger.debug(f"Reset load for resource {self.resource_id} feature '{feature}': "
                               f"{old_load:.2f} -> 0.0")
                else:
                    logger.warning(f"Feature '{feature}' not found in resource {self.resource_id}")
            else:
                # Reset all loads
                old_loads = self.current_load.copy()
                for feat in self.current_load:
                    self.current_load[feat] = 0.0
                logger.debug(f"Reset all loads for resource {self.resource_id}: {old_loads}")
            
            # Update timestamp and state
            self.last_updated = time.time()
            self._update_state_from_utilization()
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error resetting load for resource {self.resource_id}: {e}")
            raise RuntimeError(f"Load reset failed: {e}") from e
    
    def get_resource_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health assessment of the resource.
        
        Returns:
            Dict[str, Any]: Health metrics and assessment
        
        Example:
            >>> health = resource.get_resource_health()
            >>> print(f"Health score: {health['overall_health']:.2f}")
        """
        try:
            # Calculate utilization health (prefer moderate utilization)
            utilization_scores = []
            for feature in self.capacity:
                util = self.get_utilization(feature)
                # Optimal utilization is around 0.7, penalize both under and over utilization
                if util <= 0.7:
                    score = util / 0.7  # Linear up to optimal
                else:
                    score = max(0.0, 1.0 - (util - 0.7) / 0.3)  # Linear down from optimal
                utilization_scores.append(score)
            
            avg_utilization_health = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
            
            # Calculate state health
            state_health_map = {
                ResourceState.IDLE: 0.9,        # Slightly lower for underutilization
                ResourceState.BUSY: 1.0,        # Optimal
                ResourceState.OVERLOADED: 0.6,  # Suboptimal but functional
                ResourceState.DEGRADED: 0.4,    # Poor performance
                ResourceState.FAILING: 0.1,     # Critical
                ResourceState.OFFLINE: 0.0,     # Non-functional
                ResourceState.RESERVED: 0.8,    # Good but not optimal
                ResourceState.INITIALIZING: 0.5  # Temporarily unavailable
            }
            state_health = state_health_map.get(self.state, 0.5)
            
            # Calculate temporal health (based on time since last update)
            time_since_update = time.time() - self.last_updated
            if time_since_update < 60:  # Fresh data
                temporal_health = 1.0
            elif time_since_update < 300:  # Somewhat stale
                temporal_health = 0.8
            elif time_since_update < 900:  # Stale
                temporal_health = 0.6
            else:  # Very stale
                temporal_health = 0.3
            
            # Overall health (weighted average)
            overall_health = (
                avg_utilization_health * 0.4 +
                state_health * 0.4 +
                temporal_health * 0.2
            )
            
            health_assessment = {
                "overall_health": overall_health,
                "utilization_health": avg_utilization_health,
                "state_health": state_health,
                "temporal_health": temporal_health,
                "health_grade": self._get_health_grade(overall_health),
                "recommendations": self._get_health_recommendations(overall_health, avg_utilization_health, state_health),
                "last_assessed": time.time()
            }
            
            logger.debug(f"Generated health assessment for resource {self.resource_id}: "
                        f"overall={overall_health:.3f}, grade={health_assessment['health_grade']}")
            
            return health_assessment
            
        except Exception as e:
            logger.error(f"Error generating health assessment for resource {self.resource_id}: {e}")
            return {
                "overall_health": 0.0,
                "health_grade": "UNKNOWN",
                "error": str(e),
                "last_assessed": time.time()
            }
    
    def _get_health_grade(self, health_score: float) -> str:
        """Convert health score to letter grade."""
        if health_score >= 0.9:
            return "A"
        elif health_score >= 0.8:
            return "B"
        elif health_score >= 0.7:
            return "C"
        elif health_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _get_health_recommendations(self, overall: float, utilization: float, state: float) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if overall < 0.6:
            recommendations.append("Resource health is critical - immediate attention required")
        
        if utilization < 0.3:
            recommendations.append("Resource is underutilized - consider workload rebalancing")
        elif utilization > 0.8:
            recommendations.append("Resource may be overutilized - consider load reduction")
        
        if state < 0.5:
            recommendations.append("Resource state indicates operational issues - check logs")
        
        if self.state == ResourceState.OVERLOADED:
            recommendations.append("Resource is overloaded - redistribute tasks or increase capacity")
        
        if self.state in {ResourceState.FAILING, ResourceState.DEGRADED}:
            recommendations.append("Resource experiencing performance issues - maintenance required")
        
        if not recommendations:
            recommendations.append("Resource operating within normal parameters")
        
        return recommendations

@dataclass
class ComputationalTask:
    """
    Represents a computational task to be executed in the neuromorphic system.
    
    This class encapsulates all information about a computational task including
    its requirements, dependencies, execution parameters, and runtime state.
    Tasks can have complex dependency graphs and resource requirements.
    
    Attributes:
        task_id: Unique identifier for the task
        name: Human-readable name for the task
        requirements: Dictionary mapping feature names to required capacity
        dependencies: List of task IDs that must complete before this task
        priority: Task priority (0.0 to 1.0, higher is more important)
        execution_mode: How the task should be executed
        preferred_resource_types: Preferred types of processing units
        creation_time: Timestamp when task was created
        deadline: Optional deadline for task completion
        function: Optional callable function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        result: Result of task execution (None if not completed)
        error: Exception raised during execution (None if no error)
        status: Current status of the task
        assigned_resource: ID of the resource assigned to execute this task
        start_time: Timestamp when execution started
        completion_time: Timestamp when execution completed
        progress: Current progress (0.0 to 1.0)
        metadata: Additional metadata and configuration
    
    Example:
        >>> task = ComputationalTask(
        ...     name="matrix_multiplication",
        ...     requirements={"computational": 100.0, "memory": 4.0},
        ...     priority=0.8,
        ...     deadline=time.time() + 300  # 5 minutes from now
        ... )
        >>> print(f"Task urgency: {task.get_urgency():.2f}")
    """
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_task"
    requirements: Dict[str, float] = field(default_factory=dict)  # Feature -> requirement mapping
    dependencies: List[str] = field(default_factory=list)  # List of task_ids this task depends on
    priority: float = 0.5  # 0 to 1, higher is more important
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    preferred_resource_types: List[ProcessingUnit] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    function: Optional[Callable] = None
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[Exception] = None
    status: str = "pending"  # pending, assigned, running, completed, failed, cancelled
    assigned_resource: Optional[str] = None  # resource_id
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    progress: float = 0.0  # 0 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Validates task parameters after initialization.
        
        Raises:
            ValueError: If any parameters are outside valid ranges
            TypeError: If parameters are of incorrect types
        """
        try:
            # Validate task name
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError(f"Task name must be a non-empty string, got {self.name}")
            
            # Validate priority
            if not isinstance(self.priority, (int, float)) or not 0.0 <= self.priority <= 1.0:
                raise ValueError(f"Priority must be between 0.0 and 1.0, got {self.priority}")
            
            # Validate progress
            if not isinstance(self.progress, (int, float)) or not 0.0 <= self.progress <= 1.0:
                raise ValueError(f"Progress must be between 0.0 and 1.0, got {self.progress}")
            
            # Validate execution_mode
            if not isinstance(self.execution_mode, ExecutionMode):
                raise TypeError(f"execution_mode must be ExecutionMode enum, got {type(self.execution_mode)}")
            
            # Validate requirements
            if not isinstance(self.requirements, dict):
                raise TypeError(f"requirements must be a dictionary, got {type(self.requirements)}")
            
            for feature, requirement in self.requirements.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Requirement feature names must be non-empty strings, got {feature}")
                if not isinstance(requirement, (int, float)) or requirement < 0:
                    raise ValueError(f"Requirement values must be non-negative numbers, got {requirement}")
            
            # Validate dependencies
            if not isinstance(self.dependencies, list):
                raise TypeError(f"dependencies must be a list, got {type(self.dependencies)}")
            
            for dep in self.dependencies:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(f"Dependency IDs must be non-empty strings, got {dep}")
            
            # Validate preferred_resource_types
            if not isinstance(self.preferred_resource_types, list):
                raise TypeError(f"preferred_resource_types must be a list, got {type(self.preferred_resource_types)}")
            
            for resource_type in self.preferred_resource_types:
                if not isinstance(resource_type, ProcessingUnit):
                    raise TypeError(f"preferred_resource_types must contain ProcessingUnit enums, got {type(resource_type)}")
            
            # Validate status
            valid_statuses = {"pending", "assigned", "running", "completed", "failed", "cancelled"}
            if self.status not in valid_statuses:
                raise ValueError(f"Status must be one of {valid_statuses}, got {self.status}")
            
            # Validate deadline
            if self.deadline is not None:
                if not isinstance(self.deadline, (int, float)):
                    raise TypeError(f"deadline must be a number or None, got {type(self.deadline)}")
                if self.deadline < self.creation_time:
                    logger.warning(f"Task {self.task_id} deadline {self.deadline} is before creation time {self.creation_time}")
            
            # Validate time fields
            if self.start_time is not None and not isinstance(self.start_time, (int, float)):
                raise TypeError(f"start_time must be a number or None, got {type(self.start_time)}")
            
            if self.completion_time is not None and not isinstance(self.completion_time, (int, float)):
                raise TypeError(f"completion_time must be a number or None, got {type(self.completion_time)}")
            
            # Initialize metadata if needed
            if self.metadata is None:
                self.metadata = {}
            
            # Add creation timestamp to metadata
            if 'created_at' not in self.metadata:
                self.metadata['created_at'] = self.creation_time
            
            logger.debug(f"Initialized computational task {self.task_id}: {self.name} "
                        f"with {len(self.requirements)} requirements and {len(self.dependencies)} dependencies")
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize ComputationalTask: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ComputationalTask initialization: {e}")
            raise RuntimeError(f"Task initialization failed: {e}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing all task information
        
        Example:
            >>> task_dict = task.to_dict()
            >>> print(f"Task status: {task_dict['status']}")
        """
        try:
            # Calculate additional metrics
            age = time.time() - self.creation_time
            execution_time = None
            if self.start_time and self.completion_time:
                execution_time = self.completion_time - self.start_time
            elif self.start_time:
                execution_time = time.time() - self.start_time
            
            # Determine if task is overdue
            is_overdue = False
            if self.deadline and time.time() > self.deadline and self.status not in {"completed", "cancelled"}:
                is_overdue = True
            
            task_dict = {
                "task_id": self.task_id,
                "name": self.name,
                "requirements": self.requirements.copy(),
                "dependencies": self.dependencies.copy(),
                "priority": self.priority,
                "execution_mode": self.execution_mode.name,
                "preferred_resource_types": [rt.name for rt in self.preferred_resource_types],
                "creation_time": self.creation_time,
                "deadline": self.deadline,
                "status": self.status,
                "assigned_resource": self.assigned_resource,
                "start_time": self.start_time,
                "completion_time": self.completion_time,
                "progress": self.progress,
                "has_result": self.result is not None,
                "has_error": self.error is not None,
                "has_function": self.function is not None,
                "metadata": self.metadata.copy(),
                "metrics": {
                    "age": age,
                    "execution_time": execution_time,
                    "is_overdue": is_overdue,
                    "urgency": self.get_urgency(),
                    "estimated_remaining_time": self._estimate_remaining_time()
                }
            }
            
            logger.debug(f"Generated dictionary representation for task {self.task_id}")
            return task_dict
            
        except Exception as e:
            logger.error(f"Error converting task {self.task_id} to dictionary: {e}")
            # Return minimal representation on error
            return {
                "task_id": self.task_id,
                "name": self.name,
                "status": self.status,
                "error": str(e)
            }
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """
        Check if the task is ready to execute based on dependencies.
        
        Args:
            completed_tasks: Set of task IDs that have been completed
        
        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        
        Raises:
            TypeError: If completed_tasks is not a set
        
        Example:
            >>> completed = {"task_1", "task_2"}
            >>> if task.is_ready(completed):
            ...     print("Task is ready to execute")
        """
        try:
            if not isinstance(completed_tasks, set):
                raise TypeError(f"completed_tasks must be a set, got {type(completed_tasks)}")
            
            # Check if task is in a state where it can be executed
            if self.status not in {"pending", "assigned"}:
                logger.debug(f"Task {self.task_id} is not ready: status is {self.status}")
                return False
            
            # Check all dependencies
            missing_deps = []
            for dep in self.dependencies:
                if dep not in completed_tasks:
                    missing_deps.append(dep)
            
            is_ready = len(missing_deps) == 0
            
            if not is_ready:
                logger.debug(f"Task {self.task_id} is not ready: missing dependencies {missing_deps}")
            else:
                logger.debug(f"Task {self.task_id} is ready for execution")
            
            return is_ready
            
        except TypeError:
            raise
        except Exception as e:
            logger.error(f"Error checking readiness for task {self.task_id}: {e}")
            return False
    
    def get_urgency(self) -> float:
        """
        Calculate the urgency of the task based on priority and deadline.
        
        Returns:
            float: Urgency score (0.0 to 1.0+)
        
        Example:
            >>> urgency = task.get_urgency()
            >>> if urgency > 0.9:
            ...     print("High priority task!")
        """
        try:
            now = time.time()
            
            # Base urgency is the priority
            urgency = self.priority
            
            # If there's a deadline, factor it in
            if self.deadline:
                time_left = self.deadline - now
                if time_left <= 0:
                    # Past deadline, maximum urgency
                    urgency = 1.0
                else:
                    # Scale urgency based on how close we are to deadline
                    total_time = self.deadline - self.creation_time
                    if total_time > 0:
                        time_factor = 1.0 - (time_left / total_time)
                        urgency = max(urgency, time_factor)
            
            # Factor in how long the task has been waiting
            age = now - self.creation_time
            if age > 3600:  # If waiting more than 1 hour
                age_factor = min(0.3, age / 86400)  # Up to 30% boost for waiting a day
                urgency = min(1.0, urgency + age_factor)
            
            logger.debug(f"Task {self.task_id} urgency: {urgency:.3f}")
            return urgency
            
        except Exception as e:
            logger.error(f"Error calculating urgency for task {self.task_id}: {e}")
            return self.priority  # Fallback to base priority
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """
        Estimate remaining execution time based on progress.
        
        Returns:
            Optional[float]: Estimated remaining time in seconds, or None if unknown
        """
        try:
            if self.status == "completed":
                return 0.0
            
            if self.status not in {"running", "assigned"} or not self.start_time:
                return None
            
            elapsed = time.time() - self.start_time
            
            if self.progress <= 0:
                return None  # Can't estimate without progress
            
            # Estimate total time based on current progress
            estimated_total = elapsed / self.progress
            remaining = estimated_total - elapsed
            
            return max(0.0, remaining)
            
        except Exception as e:
            logger.error(f"Error estimating remaining time for task {self.task_id}: {e}")
            return None
    
    def update_progress(self, progress: float) -> None:
        """
        Update the task progress.
        
        Args:
            progress: New progress value (0.0 to 1.0)
        
        Raises:
            ValueError: If progress is not between 0.0 and 1.0
        
        Example:
            >>> task.update_progress(0.5)  # 50% complete
        """
        try:
            if not isinstance(progress, (int, float)):
                raise TypeError(f"Progress must be a number, got {type(progress)}")
            
            if not 0.0 <= progress <= 1.0:
                raise ValueError(f"Progress must be between 0.0 and 1.0, got {progress}")
            
            old_progress = self.progress
            self.progress = progress
            
            # Auto-complete if progress reaches 100%
            if progress >= 1.0 and self.status == "running":
                self.status = "completed"
                self.completion_time = time.time()
                logger.info(f"Task {self.task_id} auto-completed due to 100% progress")
            
            logger.debug(f"Updated progress for task {self.task_id}: {old_progress:.3f} -> {progress:.3f}")
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to update progress for task {self.task_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating progress for task {self.task_id}: {e}")
            raise RuntimeError(f"Progress update failed: {e}") from e
    
    def start_execution(self, resource_id: str) -> None:
        """
        Mark the task as started and assign it to a resource.
        
        Args:
            resource_id: ID of the resource executing the task
        
        Raises:
            ValueError: If task is not in a startable state
        
        Example:
            >>> task.start_execution("resource_123")
        """
        try:
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError(f"resource_id must be a non-empty string, got {resource_id}")
            
            if self.status not in {"pending", "assigned"}:
                raise ValueError(f"Cannot start task in status '{self.status}', must be 'pending' or 'assigned'")
            
            self.status = "running"
            self.assigned_resource = resource_id
            self.start_time = time.time()
            
            logger.info(f"Started execution of task {self.task_id} on resource {resource_id}")
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error starting execution for task {self.task_id}: {e}")
            raise RuntimeError(f"Failed to start task execution: {e}") from e
    
    def complete_execution(self, result: Any = None, error: Optional[Exception] = None) -> None:
        """
        Mark the task as completed or failed.
        
        Args:
            result: Result of the task execution
            error: Exception if task failed
        
        Example:
            >>> task.complete_execution(result="Success!")
            >>> # Or for failed task:
            >>> task.complete_execution(error=Exception("Task failed"))
        """
        try:
            self.completion_time = time.time()
            
            if error is not None:
                self.status = "failed"
                self.error = error
                logger.error(f"Task {self.task_id} failed: {error}")
            else:
                self.status = "completed"
                self.result = result
                self.progress = 1.0
                logger.info(f"Task {self.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error completing task {self.task_id}: {e}")
            # Try to mark as failed
            self.status = "failed"
            self.error = Exception(f"Completion error: {e}")
    
    def cancel(self, reason: str = "No reason provided") -> None:
        """
        Cancel the task execution.
        
        Args:
            reason: Reason for cancellation
        
        Example:
            >>> task.cancel("Resource unavailable")
        """
        try:
            if self.status in {"completed", "failed"}:
                logger.warning(f"Cannot cancel task {self.task_id}: already in final state {self.status}")
                return
            
            self.status = "cancelled"
            self.completion_time = time.time()
            
            if 'cancellation_reason' not in self.metadata:
                self.metadata['cancellation_reason'] = reason
            
            logger.info(f"Cancelled task {self.task_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Error cancelling task {self.task_id}: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive execution summary.
        
        Returns:
            Dict[str, Any]: Summary of task execution metrics
        
        Example:
            >>> summary = task.get_execution_summary()
            >>> print(f"Execution time: {summary['execution_time']:.2f} seconds")
        """
        try:
            now = time.time()
            
            # Calculate timing metrics
            age = now - self.creation_time
            wait_time = None
            execution_time = None
            
            if self.start_time:
                wait_time = self.start_time - self.creation_time
                if self.completion_time:
                    execution_time = self.completion_time - self.start_time
                elif self.status == "running":
                    execution_time = now - self.start_time
            
            # Calculate efficiency metrics
            efficiency_score = 0.0
            if execution_time and wait_time is not None:
                total_time = wait_time + execution_time
                efficiency_score = execution_time / total_time if total_time > 0 else 0.0
            
            summary = {
                "task_id": self.task_id,
                "name": self.name,
                "status": self.status,
                "progress": self.progress,
                "priority": self.priority,
                "urgency": self.get_urgency(),
                "timing": {
                    "age": age,
                    "wait_time": wait_time,
                    "execution_time": execution_time,
                    "estimated_remaining": self._estimate_remaining_time()
                },
                "resource": {
                    "assigned_resource": self.assigned_resource,
                    "preferred_types": [rt.name for rt in self.preferred_resource_types]
                },
                "dependencies": {
                    "count": len(self.dependencies),
                    "dependencies": self.dependencies.copy()
                },
                "requirements": self.requirements.copy(),
                "performance": {
                    "efficiency_score": efficiency_score,
                    "is_overdue": self.deadline and now > self.deadline,
                    "deadline_pressure": self._calculate_deadline_pressure()
                },
                "flags": {
                    "has_result": self.result is not None,
                    "has_error": self.error is not None,
                    "has_function": self.function is not None,
                    "has_deadline": self.deadline is not None
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating execution summary for task {self.task_id}: {e}")
            return {
                "task_id": self.task_id,
                "name": self.name,
                "status": self.status,
                "error": str(e)
            }
    
    def _calculate_deadline_pressure(self) -> float:
        """Calculate pressure from approaching deadline (0.0 to 1.0)."""
        try:
            if not self.deadline:
                return 0.0
            
            now = time.time()
            total_time = self.deadline - self.creation_time
            time_left = self.deadline - now
            
            if total_time <= 0:
                return 1.0  # Immediate deadline
            
            if time_left <= 0:
                return 1.0  # Past deadline
            
            # Pressure increases as deadline approaches
            pressure = 1.0 - (time_left / total_time)
            return max(0.0, min(1.0, pressure))
            
        except Exception:
            return 0.0
    
    def estimate_duration(self) -> float:
        """Estimate the duration of the task in seconds"""
        # Default implementation uses computational requirement as a proxy
        computational_req = self.requirements.get("computational", 1.0)
        # Convert to seconds based on a baseline of 1.0 = 1 second
        return max(0.1, computational_req)
    
    def execute(self) -> Any:
        """Execute the task function"""
        if not self.function:
            raise ValueError("No function specified for task execution")
            
        self.status = "running"
        self.start_time = time.time()
        
        try:
            self.result = self.function(*self.args, **self.kwargs)
            self.status = "completed"
        except Exception as e:
            self.error = e
            self.status = "failed"
        
        self.completion_time = time.time()
        self.progress = 1.0
        
        return self.result

@dataclass
class OptimizationMetrics:
    """
    Comprehensive metrics tracked for optimization decisions in the neuromorphic system.
    
    This class encapsulates all the key performance indicators and metrics that
    the neuromorphic optimizer uses to make resource allocation and task scheduling
    decisions. It provides methods for calculating optimization scores and comparing
    different configurations.
    
    Attributes:
        throughput: Operations completed per second
        latency: Average response time in seconds
        memory_usage: Memory usage in megabytes
        energy_consumption: Energy consumption in arbitrary units
        reliability: Reliability score (0.0 to 1.0, higher is better)
        adaptability: Adaptability score (0.0 to 1.0, higher is better)
        cost: Cost in arbitrary units (lower is better)
        qos: Quality of Service score (0.0 to 1.0, higher is better)
        timestamp: When these metrics were recorded
        sample_count: Number of samples these metrics represent
    
    Example:
        >>> metrics = OptimizationMetrics(
        ...     throughput=150.0,
        ...     latency=0.05,
        ...     memory_usage=512.0,
        ...     reliability=0.95
        ... )
        >>> score = metrics.calculate_score(OptimizationTarget.THROUGHPUT)
        >>> print(f"Throughput score: {score:.2f}")
    """
    
    throughput: float = 0.0  # Operations per second
    latency: float = 0.0  # Average response time in seconds
    memory_usage: float = 0.0  # Memory usage in MB
    energy_consumption: float = 0.0  # Energy consumption in arbitrary units
    reliability: float = 1.0  # Reliability score (0-1)
    adaptability: float = 0.5  # Adaptability score (0-1)
    cost: float = 0.0  # Cost in arbitrary units
    qos: float = 1.0  # Quality of Service (0-1)
    timestamp: float = field(default_factory=time.time)
    sample_count: int = 1
    
    def __post_init__(self) -> None:
        """
        Validates metrics parameters after initialization.
        
        Raises:
            ValueError: If any metrics are outside valid ranges
            TypeError: If parameters have incorrect types
        """
        try:
            # Validate throughput
            if not isinstance(self.throughput, (int, float)) or self.throughput < 0:
                raise ValueError(f"Throughput must be non-negative, got {self.throughput}")
            
            # Validate latency
            if not isinstance(self.latency, (int, float)) or self.latency < 0:
                raise ValueError(f"Latency must be non-negative, got {self.latency}")
            
            # Validate memory usage
            if not isinstance(self.memory_usage, (int, float)) or self.memory_usage < 0:
                raise ValueError(f"Memory usage must be non-negative, got {self.memory_usage}")
            
            # Validate energy consumption
            if not isinstance(self.energy_consumption, (int, float)) or self.energy_consumption < 0:
                raise ValueError(f"Energy consumption must be non-negative, got {self.energy_consumption}")
            
            # Validate reliability (0-1 range)
            if not isinstance(self.reliability, (int, float)) or not 0.0 <= self.reliability <= 1.0:
                raise ValueError(f"Reliability must be between 0.0 and 1.0, got {self.reliability}")
            
            # Validate adaptability (0-1 range)
            if not isinstance(self.adaptability, (int, float)) or not 0.0 <= self.adaptability <= 1.0:
                raise ValueError(f"Adaptability must be between 0.0 and 1.0, got {self.adaptability}")
            
            # Validate cost
            if not isinstance(self.cost, (int, float)) or self.cost < 0:
                raise ValueError(f"Cost must be non-negative, got {self.cost}")
            
            # Validate QoS (0-1 range)
            if not isinstance(self.qos, (int, float)) or not 0.0 <= self.qos <= 1.0:
                raise ValueError(f"QoS must be between 0.0 and 1.0, got {self.qos}")
            
            # Validate sample count
            if not isinstance(self.sample_count, int) or self.sample_count < 1:
                raise ValueError(f"Sample count must be a positive integer, got {self.sample_count}")
            
            logger.debug(f"Initialized optimization metrics with {self.sample_count} samples")
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize OptimizationMetrics: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OptimizationMetrics initialization: {e}")
            raise RuntimeError(f"Metrics initialization failed: {e}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing all metrics and metadata
        
        Example:
            >>> metrics_dict = metrics.to_dict()
            >>> print(f"Throughput: {metrics_dict['throughput']} ops/s")
        """
        try:
            metrics_dict = {
                "throughput": self.throughput,
                "latency": self.latency,
                "memory_usage": self.memory_usage,
                "energy_consumption": self.energy_consumption,
                "reliability": self.reliability,
                "adaptability": self.adaptability,
                "cost": self.cost,
                "qos": self.qos,
                "timestamp": self.timestamp,
                "sample_count": self.sample_count,
                "metadata": {
                    "age": time.time() - self.timestamp,
                    "efficiency": self._calculate_efficiency(),
                    "performance_index": self._calculate_performance_index(),
                    "sustainability_score": self._calculate_sustainability_score()
                }
            }
            
            logger.debug(f"Generated dictionary representation for optimization metrics")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error converting metrics to dictionary: {e}")
            return {
                "error": str(e),
                "timestamp": self.timestamp
            }
    
    def calculate_score(self, target: OptimizationTarget, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate optimization score based on target and optional weights.
        
        Args:
            target: The optimization target to score against
            weights: Optional weights for different metrics (for custom optimization)
        
        Returns:
            float: Optimization score (higher is better)
        
        Raises:
            ValueError: If target is invalid or weights are malformed
        
        Example:
            >>> score = metrics.calculate_score(OptimizationTarget.THROUGHPUT)
            >>> # Custom weights example:
            >>> custom_weights = {"throughput": 0.6, "latency": 0.4}
            >>> custom_score = metrics.calculate_score(OptimizationTarget.CUSTOM, custom_weights)
        """
        try:
            if not isinstance(target, OptimizationTarget):
                raise ValueError(f"target must be OptimizationTarget enum, got {type(target)}")
            
            if target == OptimizationTarget.THROUGHPUT:
                # Higher throughput is better
                score = self.throughput
                
            elif target == OptimizationTarget.LATENCY:
                # Lower latency is better, so invert (with minimum threshold to avoid division by zero)
                score = 1000.0 / max(0.001, self.latency)
                
            elif target == OptimizationTarget.MEMORY:
                # Lower memory usage is better, so invert
                score = 10000.0 / max(1.0, self.memory_usage)
                
            elif target == OptimizationTarget.ENERGY:
                # Lower energy consumption is better, so invert
                score = 1000.0 / max(0.001, self.energy_consumption)
                
            elif target == OptimizationTarget.RELIABILITY:
                # Higher reliability is better
                score = self.reliability * 100.0
                
            elif target == OptimizationTarget.ADAPTABILITY:
                # Higher adaptability is better
                score = self.adaptability * 100.0
                
            elif target == OptimizationTarget.BALANCED:
                # Balanced score across all metrics (weighted combination)
                throughput_norm = min(100.0, self.throughput / 10.0)  # Normalize throughput
                latency_norm = min(100.0, 1000.0 / max(0.001, self.latency))  # Invert and normalize
                memory_norm = min(100.0, 10000.0 / max(1.0, self.memory_usage))  # Invert and normalize
                energy_norm = min(100.0, 1000.0 / max(0.001, self.energy_consumption))  # Invert and normalize
                reliability_norm = self.reliability * 100.0
                adaptability_norm = self.adaptability * 100.0
                qos_norm = self.qos * 100.0
                
                # Weighted average with emphasis on key metrics
                score = (
                    throughput_norm * 0.2 +
                    latency_norm * 0.2 +
                    memory_norm * 0.15 +
                    energy_norm * 0.15 +
                    reliability_norm * 0.15 +
                    adaptability_norm * 0.1 +
                    qos_norm * 0.05
                )
                
            elif target == OptimizationTarget.CUSTOM:
                # Custom scoring with user-provided weights
                if not weights:
                    raise ValueError("Custom optimization target requires weights parameter")
                
                if not isinstance(weights, dict):
                    raise ValueError("Weights must be a dictionary")
                
                score = 0.0
                total_weight = 0.0
                
                metric_values = {
                    "throughput": self.throughput,
                    "latency": 1000.0 / max(0.001, self.latency),  # Inverted
                    "memory": 10000.0 / max(1.0, self.memory_usage),  # Inverted
                    "energy": 1000.0 / max(0.001, self.energy_consumption),  # Inverted
                    "reliability": self.reliability * 100.0,
                    "adaptability": self.adaptability * 100.0,
                    "qos": self.qos * 100.0,
                    "cost": 1000.0 / max(0.001, self.cost)  # Inverted (lower cost is better)
                }
                
                for metric, weight in weights.items():
                    if metric not in metric_values:
                        logger.warning(f"Unknown metric '{metric}' in weights, skipping")
                        continue
                    
                    if not isinstance(weight, (int, float)) or weight < 0:
                        raise ValueError(f"Weight for '{metric}' must be non-negative, got {weight}")
                    
                    score += metric_values[metric] * weight
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    score /= total_weight
                else:
                    raise ValueError("Total weight must be positive")
            
            else:
                # Fallback: simple balanced score
                score = (self.throughput / 10.0 + 1000.0 / max(0.001, self.latency)) / 2.0
            
            logger.debug(f"Calculated optimization score for {target.name}: {score:.3f}")
            return max(0.0, score)  # Ensure non-negative
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0  # Safe fallback
    
    def _calculate_efficiency(self) -> float:
        """Calculate overall efficiency metric."""
        try:
            if self.energy_consumption <= 0:
                return 0.0
            
            # Efficiency = work done per unit energy
            work_done = self.throughput * self.reliability * self.qos
            efficiency = work_done / self.energy_consumption
            return efficiency
            
        except Exception:
            return 0.0
    
    def _calculate_performance_index(self) -> float:
        """Calculate overall performance index."""
        try:
            # Combine throughput and latency for performance
            if self.latency <= 0:
                return self.throughput
            
            # Performance = throughput / latency (operations per second per second of latency)
            performance = self.throughput / self.latency
            return performance
            
        except Exception:
            return 0.0
    
    def _calculate_sustainability_score(self) -> float:
        """Calculate sustainability score based on resource usage."""
        try:
            # Lower resource usage is more sustainable
            memory_factor = 1.0 / max(1.0, self.memory_usage / 1000.0)  # Normalize to GB
            energy_factor = 1.0 / max(0.001, self.energy_consumption)
            cost_factor = 1.0 / max(0.001, self.cost)
            
            # Combine factors with reliability and adaptability
            sustainability = (
                memory_factor * 0.3 +
                energy_factor * 0.3 +
                cost_factor * 0.2 +
                self.reliability * 0.1 +
                self.adaptability * 0.1
            )
            
            return min(1.0, sustainability)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def compare_with(self, other: 'OptimizationMetrics', target: OptimizationTarget) -> Dict[str, Any]:
        """
        Compare these metrics with another set of metrics.
        
        Args:
            other: Other OptimizationMetrics instance to compare with
            target: Optimization target for scoring comparison
        
        Returns:
            Dict[str, Any]: Comparison results and analysis
        
        Raises:
            TypeError: If other is not OptimizationMetrics instance
        
        Example:
            >>> comparison = metrics1.compare_with(metrics2, OptimizationTarget.THROUGHPUT)
            >>> if comparison['is_better']:
            ...     print("These metrics are better!")
        """
        try:
            if not isinstance(other, OptimizationMetrics):
                raise TypeError(f"other must be OptimizationMetrics instance, got {type(other)}")
            
            # Calculate scores for both
            self_score = self.calculate_score(target)
            other_score = other.calculate_score(target)
            
            # Calculate individual metric comparisons
            comparisons = {
                "throughput": {
                    "self": self.throughput,
                    "other": other.throughput,
                    "diff": self.throughput - other.throughput,
                    "pct_change": ((self.throughput - other.throughput) / max(0.001, other.throughput)) * 100
                },
                "latency": {
                    "self": self.latency,
                    "other": other.latency,
                    "diff": self.latency - other.latency,
                    "pct_change": ((self.latency - other.latency) / max(0.001, other.latency)) * 100
                },
                "memory_usage": {
                    "self": self.memory_usage,
                    "other": other.memory_usage,
                    "diff": self.memory_usage - other.memory_usage,
                    "pct_change": ((self.memory_usage - other.memory_usage) / max(0.001, other.memory_usage)) * 100
                },
                "energy_consumption": {
                    "self": self.energy_consumption,
                    "other": other.energy_consumption,
                    "diff": self.energy_consumption - other.energy_consumption,
                    "pct_change": ((self.energy_consumption - other.energy_consumption) / max(0.001, other.energy_consumption)) * 100
                },
                "reliability": {
                    "self": self.reliability,
                    "other": other.reliability,
                    "diff": self.reliability - other.reliability,
                    "pct_change": ((self.reliability - other.reliability) / max(0.001, other.reliability)) * 100
                }
            }
            
            # Determine which is better overall
            is_better = self_score > other_score
            score_improvement = self_score - other_score
            
            # Generate recommendations
            recommendations = []
            if comparisons["throughput"]["diff"] < 0:
                recommendations.append("Consider optimizing for higher throughput")
            if comparisons["latency"]["diff"] > 0:
                recommendations.append("Consider optimizing for lower latency")
            if comparisons["memory_usage"]["diff"] > 0:
                recommendations.append("Consider optimizing for lower memory usage")
            if comparisons["energy_consumption"]["diff"] > 0:
                recommendations.append("Consider optimizing for lower energy consumption")
            if comparisons["reliability"]["diff"] < 0:
                recommendations.append("Consider improving system reliability")
            
            comparison_result = {
                "target": target.name,
                "scores": {
                    "self": self_score,
                    "other": other_score,
                    "improvement": score_improvement,
                    "pct_improvement": (score_improvement / max(0.001, other_score)) * 100
                },
                "is_better": is_better,
                "metric_comparisons": comparisons,
                "recommendations": recommendations,
                "summary": f"{'Better' if is_better else 'Worse'} by {abs(score_improvement):.3f} points ({abs(score_improvement / max(0.001, other_score) * 100):.1f}%)"
            }
            
            logger.debug(f"Generated comparison analysis: {'better' if is_better else 'worse'} by {score_improvement:.3f}")
            return comparison_result
            
        except TypeError:
            raise
        except Exception as e:
            logger.error(f"Error comparing metrics: {e}")
            return {
                "error": str(e),
                "is_better": False,
                "scores": {"self": 0.0, "other": 0.0}
            }

class ResourceManager:
    """
    Manages computational resources in the neuromorphic system.
    
    This class provides comprehensive resource management capabilities including
    resource discovery, allocation, monitoring, and lifecycle management. It
    supports heterogeneous resource types and provides thread-safe operations.
    
    The ResourceManager automatically discovers system resources (CPU, GPU)
    and can manage simulated resources (QPU, NPU) for testing and development.
    It provides real-time monitoring and load balancing capabilities.
    
    Attributes:
        resources: Dictionary mapping resource IDs to ComputationalResource instances
        resource_types: Dictionary organizing resources by ProcessingUnit type
        _lock: Thread lock for ensuring thread-safe operations
        _monitoring_active: Flag indicating if resource monitoring is active
        _monitor_thread: Background thread for resource monitoring
        _monitor_interval: Interval in seconds between monitoring updates
    
    Example:
        >>> manager = ResourceManager()
        >>> manager.discover_system_resources()
        >>> resources = manager.get_available_resources({"computational": 10.0})
        >>> if resources:
        ...     success = manager.allocate_resource(resources[0], {"computational": 10.0})
    """
    
    def __init__(self, monitor_interval: float = 5.0):
        """
        Initialize the resource manager.
        
        Args:
            monitor_interval: Interval in seconds between resource monitoring updates
        
        Raises:
            ValueError: If monitor_interval is not positive
        """
        try:
            if not isinstance(monitor_interval, (int, float)) or monitor_interval <= 0:
                raise ValueError(f"monitor_interval must be positive, got {monitor_interval}")
            
            self.resources: Dict[str, ComputationalResource] = {}
            self.resource_types: Dict[ProcessingUnit, List[str]] = {pt: [] for pt in ProcessingUnit}
            self._lock = threading.RLock()
            self._monitoring_active = False
            self._monitor_thread: Optional[threading.Thread] = None
            self._monitor_interval = monitor_interval
            self._shutdown_event = threading.Event()
            
            logger.info(f"Initialized ResourceManager with monitoring interval {monitor_interval}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResourceManager: {e}")
            raise RuntimeError(f"ResourceManager initialization failed: {e}") from e
    
    def add_resource(self, resource: ComputationalResource) -> str:
        """
        Add a resource to the manager.
        
        Args:
            resource: ComputationalResource instance to add
        
        Returns:
            str: Resource ID of the added resource
        
        Raises:
            TypeError: If resource is not a ComputationalResource instance
            ValueError: If resource already exists
        
        Example:
            >>> resource = ComputationalResource(resource_type=ProcessingUnit.GPU)
            >>> resource_id = manager.add_resource(resource)
        """
        try:
            if not isinstance(resource, ComputationalResource):
                raise TypeError(f"resource must be ComputationalResource instance, got {type(resource)}")
            
            with self._lock:
                if resource.resource_id in self.resources:
                    raise ValueError(f"Resource {resource.resource_id} already exists")
                
                self.resources[resource.resource_id] = resource
                self.resource_types[resource.resource_type].append(resource.resource_id)
                
                logger.info(f"Added resource {resource.resource_id} of type {resource.resource_type.name}")
                return resource.resource_id
                
        except (TypeError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Error adding resource: {e}")
            raise RuntimeError(f"Failed to add resource: {e}") from e
    
    def remove_resource(self, resource_id: str) -> bool:
        """
        Remove a resource from the manager.
        
        Args:
            resource_id: ID of the resource to remove
        
        Returns:
            bool: True if resource was removed, False if not found
        
        Raises:
            ValueError: If resource_id is invalid
        
        Example:
            >>> success = manager.remove_resource("resource_123")
        """
        try:
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError(f"resource_id must be a non-empty string, got {resource_id}")
            
            with self._lock:
                if resource_id not in self.resources:
                    logger.warning(f"Resource {resource_id} not found for removal")
                    return False
                
                resource = self.resources[resource_id]
                del self.resources[resource_id]
                
                # Remove from resource types index
                if resource_id in self.resource_types[resource.resource_type]:
                    self.resource_types[resource.resource_type].remove(resource_id)
                
                logger.info(f"Removed resource {resource_id} of type {resource.resource_type.name}")
                return True
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error removing resource {resource_id}: {e}")
            return False
    
    def get_resource(self, resource_id: str) -> Optional[ComputationalResource]:
        """
        Get a specific resource by ID.
        
        Args:
            resource_id: ID of the resource to retrieve
        
        Returns:
            Optional[ComputationalResource]: The resource if found, None otherwise
        
        Raises:
            ValueError: If resource_id is invalid
        
        Example:
            >>> resource = manager.get_resource("cpu_001")
            >>> if resource:
            ...     print(f"Resource utilization: {resource.get_utilization():.2f}")
        """
        try:
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError(f"resource_id must be a non-empty string, got {resource_id}")
            
            with self._lock:
                resource = self.resources.get(resource_id)
                if resource:
                    logger.debug(f"Retrieved resource {resource_id}")
                else:
                    logger.debug(f"Resource {resource_id} not found")
                return resource
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving resource {resource_id}: {e}")
            return None
    
    def get_resources_by_type(self, resource_type: ProcessingUnit) -> List[ComputationalResource]:
        """
        Get all resources of a specific type.
        
        Args:
            resource_type: Type of processing unit to filter by
        
        Returns:
            List[ComputationalResource]: List of resources of the specified type
        
        Raises:
            TypeError: If resource_type is not a ProcessingUnit enum
        
        Example:
            >>> gpu_resources = manager.get_resources_by_type(ProcessingUnit.GPU)
            >>> print(f"Found {len(gpu_resources)} GPU resources")
        """
        try:
            if not isinstance(resource_type, ProcessingUnit):
                raise TypeError(f"resource_type must be ProcessingUnit enum, got {type(resource_type)}")
            
            with self._lock:
                resource_ids = self.resource_types.get(resource_type, [])
                resources = [self.resources[rid] for rid in resource_ids if rid in self.resources]
                
                logger.debug(f"Retrieved {len(resources)} resources of type {resource_type.name}")
                return resources
                
        except TypeError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving resources by type {resource_type}: {e}")
            return []
    
    def get_available_resources(self, requirements: Dict[str, float]) -> List[str]:
        """
        Get resources that can satisfy the given requirements.
        
        Args:
            requirements: Dictionary mapping feature names to required amounts
        
        Returns:
            List[str]: List of resource IDs that can satisfy the requirements
        
        Raises:
            TypeError: If requirements is not a dictionary
            ValueError: If requirements contain invalid values
        
        Example:
            >>> reqs = {"computational": 50.0, "memory": 2048.0}
            >>> available = manager.get_available_resources(reqs)
            >>> print(f"Found {len(available)} suitable resources")
        """
        try:
            if not isinstance(requirements, dict):
                raise TypeError(f"requirements must be a dictionary, got {type(requirements)}")
            
            # Validate requirements
            for feature, amount in requirements.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Requirement feature names must be non-empty strings, got {feature}")
                if not isinstance(amount, (int, float)) or amount < 0:
                    raise ValueError(f"Requirement amounts must be non-negative numbers, got {amount}")
            
            with self._lock:
                available_resources = []
                
                for resource_id, resource in self.resources.items():
                    try:
                        # Check if resource is in a usable state
                        if resource.state not in {ResourceState.IDLE, ResourceState.BUSY}:
                            continue
                        
                        # Check if it can satisfy all requirements
                        if resource.can_handle_task(requirements):
                            available_resources.append(resource_id)
                            
                    except Exception as e:
                        logger.warning(f"Error checking resource {resource_id} availability: {e}")
                        continue
                
                logger.debug(f"Found {len(available_resources)} resources available for requirements {requirements}")
                return available_resources
                
        except (TypeError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Error finding available resources: {e}")
            return []
    
    def allocate_resource(self, resource_id: str, requirements: Dict[str, float]) -> bool:
        """
        Allocate a resource for a task.
        
        Args:
            resource_id: ID of the resource to allocate
            requirements: Dictionary mapping feature names to required amounts
        
        Returns:
            bool: True if allocation was successful, False otherwise
        
        Raises:
            ValueError: If resource_id or requirements are invalid
            TypeError: If parameters have incorrect types
        
        Example:
            >>> requirements = {"computational": 25.0, "memory": 1024.0}
            >>> success = manager.allocate_resource("cpu_001", requirements)
            >>> if success:
            ...     print("Resource allocated successfully")
        """
        try:
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError(f"resource_id must be a non-empty string, got {resource_id}")
            
            if not isinstance(requirements, dict):
                raise TypeError(f"requirements must be a dictionary, got {type(requirements)}")
            
            # Validate requirements
            for feature, amount in requirements.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Requirement feature names must be non-empty strings, got {feature}")
                if not isinstance(amount, (int, float)) or amount < 0:
                    raise ValueError(f"Requirement amounts must be non-negative numbers, got {amount}")
            
            with self._lock:
                resource = self.resources.get(resource_id)
                if not resource:
                    logger.warning(f"Resource {resource_id} not found for allocation")
                    return False
                
                # Check if resource can satisfy requirements
                if not resource.can_handle_task(requirements):
                    logger.debug(f"Resource {resource_id} cannot satisfy requirements {requirements}")
                    return False
                
                # Update resource load for each requirement
                allocation_successful = True
                allocated_features = []
                
                try:
                    for feature, req_amount in requirements.items():
                        if resource.update_load(feature, req_amount):
                            allocated_features.append(feature)
                        else:
                            allocation_successful = False
                            break
                    
                    if not allocation_successful:
                        # Rollback partial allocations
                        for feature in allocated_features:
                            resource.update_load(feature, -requirements[feature])
                        
                        logger.warning(f"Failed to allocate resource {resource_id}: partial allocation rollback performed")
                        return False
                    
                    logger.info(f"Successfully allocated resource {resource_id} with requirements {requirements}")
                    return True
                    
                except Exception as e:
                    # Rollback any successful allocations
                    for feature in allocated_features:
                        try:
                            resource.update_load(feature, -requirements[feature])
                        except Exception as rollback_error:
                            logger.error(f"Error during allocation rollback: {rollback_error}")
                    
                    logger.error(f"Error during resource allocation: {e}")
                    return False
                
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error allocating resource {resource_id}: {e}")
            return False
    
    def release_resource(self, resource_id: str, requirements: Dict[str, float]) -> bool:
        """
        Release resources allocated for a task.
        
        Args:
            resource_id: ID of the resource to release
            requirements: Dictionary mapping feature names to amounts to release
        
        Returns:
            bool: True if release was successful, False otherwise
        
        Raises:
            ValueError: If resource_id or requirements are invalid
            TypeError: If parameters have incorrect types
        
        Example:
            >>> requirements = {"computational": 25.0, "memory": 1024.0}
            >>> success = manager.release_resource("cpu_001", requirements)
        """
        try:
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError(f"resource_id must be a non-empty string, got {resource_id}")
            
            if not isinstance(requirements, dict):
                raise TypeError(f"requirements must be a dictionary, got {type(requirements)}")
            
            # Validate requirements
            for feature, amount in requirements.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Requirement feature names must be non-empty strings, got {feature}")
                if not isinstance(amount, (int, float)) or amount < 0:
                    raise ValueError(f"Requirement amounts must be non-negative numbers, got {amount}")
            
            with self._lock:
                resource = self.resources.get(resource_id)
                if not resource:
                    logger.warning(f"Resource {resource_id} not found for release")
                    return False
                
                # Release resources by reducing load
                release_successful = True
                released_features = []
                
                try:
                    for feature, req_amount in requirements.items():
                        if resource.update_load(feature, -req_amount):
                            released_features.append(feature)
                        else:
                            release_successful = False
                            break
                    
                    if not release_successful:
                        # Rollback releases
                        for feature in released_features:
                            resource.update_load(feature, requirements[feature])
                        
                        logger.warning(f"Failed to release resource {resource_id}: partial release rollback performed")
                        return False
                    
                    logger.info(f"Successfully released resource {resource_id} with requirements {requirements}")
                    return True
                    
                except Exception as e:
                    # Rollback any successful releases
                    for feature in released_features:
                        try:
                            resource.update_load(feature, requirements[feature])
                        except Exception as rollback_error:
                            logger.error(f"Error during release rollback: {rollback_error}")
                    
                    logger.error(f"Error during resource release: {e}")
                    return False
                
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error releasing resource {resource_id}: {e}")
            return False
    
    def discover_system_resources(self) -> List[ComputationalResource]:
        """
        Discover and add resources based on the actual system capabilities.
        
        This method automatically detects available system resources including
        CPU cores, memory, GPU devices, and creates simulated quantum and
        neuromorphic processing units for demonstration purposes.
        
        Returns:
            List[ComputationalResource]: List of discovered resources
        
        Example:
            >>> resources = manager.discover_system_resources()
            >>> print(f"Discovered {len(resources)} system resources")
        """
        try:
            with self._lock:
                discovered_resources = []
                
                # Discover CPU resources
                try:
                    cpu_count = os.cpu_count() or 4
                    total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
                    
                    cpu_capacity = {
                        "computational": float(cpu_count * 100),  # Scale for better representation
                        "memory": total_memory,
                        "threads": float(cpu_count * 2),  # Hyperthreading assumption
                        "cache": float(cpu_count * 8),  # Estimated cache in MB
                        "bandwidth": 1000.0  # Estimated memory bandwidth
                    }
                    
                    cpu_resource = ComputationalResource(
                        resource_id="system_cpu_primary",
                        resource_type=ProcessingUnit.CPU,
                        capacity=cpu_capacity,
                        location="local_system",
                        capabilities={"cpu", "parallel", "vector", "x86_64", "multithreading"},
                        metadata={
                            "cpu_count": cpu_count,
                            "architecture": platform.machine(),
                            "system": platform.system(),
                            "discovered_at": time.time()
                        }
                    )
                    
                    self.add_resource(cpu_resource)
                    discovered_resources.append(cpu_resource)
                    logger.info(f"Discovered CPU resource with {cpu_count} cores and {total_memory:.0f} MB memory")
                    
                except Exception as e:
                    logger.error(f"Error discovering CPU resources: {e}")
                
                # Discover GPU resources
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        for i in range(gpu_count):
                            try:
                                gpu_props = torch.cuda.get_device_properties(i)
                                
                                gpu_capacity = {
                                    "computational": float(gpu_props.multi_processor_count * 128),  # Estimated compute units
                                    "memory": float(gpu_props.total_memory / (1024 * 1024)),  # Convert to MB
                                    "cores": float(gpu_props.multi_processor_count * 64),  # Estimated CUDA cores
                                    "tensor_ops": float(gpu_props.multi_processor_count * 256),  # Tensor operation capacity
                                    "bandwidth": 500.0  # Estimated memory bandwidth
                                }
                                
                                gpu_resource = ComputationalResource(
                                    resource_id=f"system_gpu_{i}",
                                    resource_type=ProcessingUnit.GPU,
                                    capacity=gpu_capacity,
                                    location="local_system",
                                    capabilities={"gpu", "cuda", "parallel", "tensor", "fp16", "fp32"},
                                    metadata={
                                        "device_name": gpu_props.name,
                                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                                        "multiprocessor_count": gpu_props.multi_processor_count,
                                        "total_memory": gpu_props.total_memory,
                                        "discovered_at": time.time()
                                    }
                                )
                                
                                self.add_resource(gpu_resource)
                                discovered_resources.append(gpu_resource)
                                logger.info(f"Discovered GPU resource: {gpu_props.name} with {gpu_props.total_memory // (1024*1024)} MB memory")
                                
                            except Exception as e:
                                logger.error(f"Error discovering GPU {i}: {e}")
                                
                except ImportError:
                    logger.info("PyTorch not available, skipping GPU discovery")
                except Exception as e:
                    logger.error(f"Error during GPU discovery: {e}")
                
                # Create simulated quantum processing unit
                try:
                    qpu_capacity = {
                        "qubits": 16.0,  # Simulated quantum bits
                        "quantum_memory": 32.0,  # Quantum memory capacity
                        "gate_ops": 1000.0,  # Quantum gate operations per second
                        "coherence_time": 100.0,  # Coherence time in microseconds
                        "fidelity": 0.99  # Gate fidelity
                    }
                    
                    qpu_resource = ComputationalResource(
                        resource_id="simulated_qpu_001",
                        resource_type=ProcessingUnit.QPU,
                        capacity=qpu_capacity,
                        location="simulated",
                        capabilities={"quantum", "superposition", "entanglement", "gate_model", "simulation"},
                        metadata={
                            "simulated": True,
                            "quantum_volume": 32,
                            "topology": "all_to_all",
                            "discovered_at": time.time()
                        }
                    )
                    
                    self.add_resource(qpu_resource)
                    discovered_resources.append(qpu_resource)
                    logger.info("Created simulated QPU resource with 16 qubits")
                    
                except Exception as e:
                    logger.error(f"Error creating simulated QPU: {e}")
                
                # Create simulated neuromorphic processing unit
                try:
                    npu_capacity = {
                        "neurons": 2048.0,  # Simulated neurons
                        "synapses": 16384.0,  # Simulated synapses
                        "spike_throughput": 2000.0,  # Spikes per second
                        "learning_rate": 100.0,  # Learning operations per second
                        "memory_capacity": 1024.0  # Synaptic memory in MB
                    }
                    
                    npu_resource = ComputationalResource(
                        resource_id="simulated_npu_001",
                        resource_type=ProcessingUnit.NPU,
                        capacity=npu_capacity,
                        location="simulated",
                        capabilities={"neuromorphic", "spiking", "learning", "plasticity", "stdp"},
                        metadata={
                            "simulated": True,
                            "neuron_model": "leaky_integrate_fire",
                            "plasticity_rules": ["STDP", "homeostasis"],
                            "discovered_at": time.time()
                        }
                    )
                    
                    self.add_resource(npu_resource)
                    discovered_resources.append(npu_resource)
                    logger.info("Created simulated NPU resource with 2048 neurons")
                    
                except Exception as e:
                    logger.error(f"Error creating simulated NPU: {e}")
                
                # Create simulated TPU for tensor operations
                try:
                    tpu_capacity = {
                        "computational": 2000.0,  # Matrix operations per second
                        "memory": 8192.0,  # High-bandwidth memory in MB
                        "tensor_ops": 4000.0,  # Tensor operations per second
                        "matrix_ops": 3000.0,  # Matrix multiplication ops per second
                        "bandwidth": 2000.0  # Memory bandwidth
                    }
                    
                    tpu_resource = ComputationalResource(
                        resource_id="simulated_tpu_001",
                        resource_type=ProcessingUnit.TPU,
                        capacity=tpu_capacity,
                        location="simulated",
                        capabilities={"tpu", "tensor", "matrix", "bfloat16", "systolic_array"},
                        metadata={
                            "simulated": True,
                            "version": "v4",
                            "systolic_array_size": "128x128",
                            "discovered_at": time.time()
                        }
                    )
                    
                    self.add_resource(tpu_resource)
                    discovered_resources.append(tpu_resource)
                    logger.info("Created simulated TPU resource with tensor acceleration")
                    
                except Exception as e:
                    logger.error(f"Error creating simulated TPU: {e}")
                
                logger.info(f"Resource discovery completed: {len(discovered_resources)} resources discovered/created")
                return discovered_resources
                
        except Exception as e:
            logger.error(f"Error during system resource discovery: {e}")
            return []
        """Get resources based on the actual system capabilities"""
        with self._lock:
            # Create resources based on system
            cpu_count = os.cpu_count() or 4
            
            # Create CPU resource
            cpu_capacity = {
                "computational": float(cpu_count),
                "memory": psutil.virtual_memory().total / (1024 * 1024),  # MB
                "threads": float(cpu_count * 2)
            }
            
            cpu_resource = ComputationalResource(
                resource_id="system_cpu",
                resource_type=ProcessingUnit.CPU,
                capacity=cpu_capacity,
                current_load={
                    "computational": 0.0,
                    "memory": 0.0,
                    "threads": 0.0
                },
                capabilities={"cpu", "parallel", "vector"}
            )
            
            # If we have a GPU, add it
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        gpu_props = torch.cuda.get_device_properties(i)
                        
                        gpu_resource = ComputationalResource(
                            resource_id=f"system_gpu_{i}",
                            resource_type=ProcessingUnit.GPU,
                            capacity={
                                "computational": float(gpu_props.multi_processor_count),
                                "memory": float(gpu_props.total_memory / (1024 * 1024)),  # MB
                                "cores": float(gpu_props.multi_processor_count * 64)  # Estimate
                            },
                            current_load={
                                "computational": 0.0,
                                "memory": 0.0,
                                "cores": 0.0
                            },
                            capabilities={"gpu", "cuda", "parallel", "tensor"}
                        )
                        
                        self.add_resource(gpu_resource)
            except ImportError:
                # No PyTorch, assume no GPU
                pass
                
            # Add CPU resource
            self.add_resource(cpu_resource)
            
            # Create a simulated QPU (Quantum Processing Unit) for demonstration
            qpu_resource = ComputationalResource(
                resource_id="simulated_qpu",
                resource_type=ProcessingUnit.QPU,
                capacity={
                    "qubits": 8.0,
                    "quantum_memory": 16.0
                },
                current_load={
                    "qubits": 0.0,
                    "quantum_memory": 0.0
                },
                capabilities={"quantum", "superposition", "entanglement"},
                metadata={"simulated": True}
            )
            
            self.add_resource(qpu_resource)
            
            # Create a simulated NPU (Neuromorphic Processing Unit)
            npu_resource = ComputationalResource(
                resource_id="simulated_npu",
                resource_type=ProcessingUnit.NPU,
                capacity={
                    "neurons": 1024.0,
                    "synapses": 8192.0,
                    "spike_throughput": 1000.0
                },
                current_load={
                    "neurons": 0.0,
                    "synapses": 0.0,
                    "spike_throughput": 0.0
                },
                capabilities={"neuromorphic", "spiking", "learning"},
                metadata={"simulated": True}
            )
            
            self.add_resource(npu_resource)
            
            return list(self.resources.values())
    
    def update_resource_loads(self) -> None:
        """Update resource loads based on system monitoring"""
        with self._lock:
            # Update CPU load
            cpu_resource = self.resources.get("system_cpu")
            if cpu_resource:
                cpu_percent = psutil.cpu_percent() / 100.0
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                cpu_resource.current_load["computational"] = cpu_resource.capacity["computational"] * cpu_percent
                cpu_resource.current_load["memory"] = cpu_resource.capacity["memory"] * memory_percent
                cpu_resource.current_load["threads"] = cpu_resource.current_load["computational"] * 2
                
                # Update state
                if cpu_percent > 0.9:
                    cpu_resource.state = ResourceState.OVERLOADED
                elif cpu_percent > 0.7:
                    cpu_resource.state = ResourceState.BUSY
                else:
                    cpu_resource.state = ResourceState.IDLE
            
            # Update GPU load if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        gpu_resource = self.resources.get(f"system_gpu_{i}")
                        if gpu_resource:
                            # This is a simplification; real GPU monitoring would use NVML or similar
                            # Here we just use a simple model based on time
                            t = time.time()
                            gpu_percent = 0.3 + 0.2 * math.sin(t / 60.0) + 0.1 * math.sin(t / 10.0)
                            gpu_percent = max(0, min(1, gpu_percent))
                            
                            gpu_resource.current_load["computational"] = gpu_resource.capacity["computational"] * gpu_percent
                            gpu_resource.current_load["memory"] = gpu_resource.capacity["memory"] * gpu_percent * 0.8
                            gpu_resource.current_load["cores"] = gpu_resource.current_load["computational"] * 64
                            
                            # Update state
                            if gpu_percent > 0.9:
                                gpu_resource.state = ResourceState.OVERLOADED
                            elif gpu_percent > 0.7:
                                gpu_resource.state = ResourceState.BUSY
                            else:
                                gpu_resource.state = ResourceState.IDLE
            except ImportError:
                pass
            
            # Simulated resources get random variations
            qpu_resource = self.resources.get("simulated_qpu")
            if qpu_resource:
                t = time.time()
                qpu_percent = 0.2 + 0.15 * math.sin(t / 45.0) + 0.05 * math.sin(t / 5.0)
                qpu_percent = max(0, min(1, qpu_percent))
                
                qpu_resource.current_load["qubits"] = qpu_resource.capacity["qubits"] * qpu_percent
                qpu_resource.current_load["quantum_memory"] = qpu_resource.capacity["quantum_memory"] * qpu_percent * 0.7
                
                # Update state
                if qpu_percent > 0.8:
                    qpu_resource.state = ResourceState.BUSY
                else:
                    qpu_resource.state = ResourceState.IDLE
            
            npu_resource = self.resources.get("simulated_npu")
            if npu_resource:
                t = time.time()
                npu_percent = 0.3 + 0.2 * math.sin(t / 30.0) + 0.1 * math.cos(t / 15.0)
                npu_percent = max(0, min(1, npu_percent))
                
                npu_resource.current_load["neurons"] = npu_resource.capacity["neurons"] * npu_percent
                npu_resource.current_load["synapses"] = npu_resource.capacity["synapses"] * npu_percent * 0.6
                npu_resource.current_load["spike_throughput"] = npu_resource.capacity["spike_throughput"] * npu_percent * 0.8
                
                # Update state
                if npu_percent > 0.85:
                    npu_resource.state = ResourceState.BUSY
                else:
                    npu_resource.state = ResourceState.IDLE

class TaskScheduler:
    """Schedules computational tasks on available resources"""
    
    def __init__(self, resource_manager: ResourceManager):
        """Initialize the task scheduler"""
        self.resource_manager = resource_manager
        self.tasks: Dict[str, ComputationalTask] = {}
        self.pending_tasks: List[str] = []
        self.running_tasks: Dict[str, Tuple[str, threading.Thread]] = {}  # task_id -> (resource_id, thread)
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Optimization metrics
        self.metrics = OptimizationMetrics()
        self.optimization_target = OptimizationTarget.BALANCED
        
        # Task history for performance analysis
        self.task_history: List[Dict[str, Any]] = []
        self.history_limit = 1000
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def submit_task(self, task: ComputationalTask) -> str:
        """Submit a task for scheduling"""
        with self._lock:
            self.tasks[task.task_id] = task
            self.pending_tasks.append(task.task_id)
            return task.task_id
    
    def get_task(self, task_id: str) -> Optional[ComputationalTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if possible"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
                
            if task.status == "pending":
                # Remove from pending queue
                if task_id in self.pending_tasks:
                    self.pending_tasks.remove(task_id)
                task.status = "cancelled"
                return True
                
            elif task.status == "assigned" or task.status == "running":
                # Mark for cancellation
                task.status = "cancelled"
                return True
                
            # Already completed, failed, or cancelled
            return False
    
    def set_optimization_target(self, target: OptimizationTarget) -> None:
        """Set the optimization target for the scheduler"""
        self.optimization_target = target
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task"""
        task = self.tasks.get(task_id)
        if task:
            return task.status
        return None
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = None) -> Any:
        """Get the result of a completed task"""
        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
            
        start_time = time.time()
        
        # Wait for completion if requested
        while wait and task.status not in ["completed", "failed", "cancelled"]:
            time.sleep(0.1)
            
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
        
        if task.status == "completed":
            return task.result
        elif task.status == "failed":
            raise task.error or RuntimeError(f"Task {task_id} failed without specific error")
        elif task.status == "cancelled":
            raise RuntimeError(f"Task {task_id} was cancelled")
        else:
            raise RuntimeError(f"Task {task_id} is not completed (status: {task.status})")
    
    def get_metrics(self) -> OptimizationMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self._lock:
            return {
                "tasks_total": len(self.tasks),
                "tasks_pending": len(self.pending_tasks),
                "tasks_running": len(self.running_tasks),
                "tasks_completed": len(self.completed_tasks),
                "tasks_failed": len(self.failed_tasks),
                "metrics": self.metrics.to_dict(),
                "optimization_target": self.optimization_target.name
            }
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while not self._shutdown:
            try:
                self._schedule_tasks()
                self._update_metrics()
                self._cleanup_completed_tasks()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
            
            # Sleep a bit before next iteration
            time.sleep(0.1)
    
    def _schedule_tasks(self) -> None:
        """Schedule pending tasks on available resources"""
        with self._lock:
            # Update resource information
            self.resource_manager.update_resource_loads()
            
            # Sort pending tasks by priority and readiness
            ready_tasks = []
            for task_id in list(self.pending_tasks):
                task = self.tasks.get(task_id)
                if task and task.is_ready(self.completed_tasks):
                    # Calculate effective priority based on urgency
                    effective_priority = task.get_urgency()
                    ready_tasks.append((effective_priority, task_id))
            
            # Sort by priority (highest first)
            ready_tasks.sort(reverse=True)
            
            # Try to schedule each ready task
            for _, task_id in ready_tasks:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                    
                # Find suitable resources
                suitable_resources = self._find_suitable_resources(task)
                
                if not suitable_resources:
                    # No suitable resource found, keep task pending
                    continue
                    
                # Choose the best resource
                best_resource = self._choose_best_resource(task, suitable_resources)
                
                if best_resource:
                    # Allocate the resource
                    success = self.resource_manager.allocate_resource(best_resource, task.requirements)
                    if success:
                        # Update task status
                        task.status = "assigned"
                        task.assigned_resource = best_resource
                        
                                                # Remove from pending tasks
                        if task_id in self.pending_tasks:
                            self.pending_tasks.remove(task_id)
                        
                        # Start task execution
                        self._start_task_execution(task, best_resource)
    
    def _find_suitable_resources(self, task: ComputationalTask) -> List[str]:
        """Find resources suitable for a task"""
        # Get resources that can satisfy the requirements
        available_resources = self.resource_manager.get_available_resources(task.requirements)
        
        # Filter by preferred resource types if specified
        if task.preferred_resource_types:
            filtered_resources = []
            for resource_id in available_resources:
                resource = self.resource_manager.get_resource(resource_id)
                if resource and resource.resource_type in task.preferred_resource_types:
                    filtered_resources.append(resource_id)
            
            if filtered_resources:
                return filtered_resources
        
        # If we didn't filter or no preferred resources available, return all suitable resources
        return available_resources
    
    def _choose_best_resource(self, task: ComputationalTask, resource_ids: List[str]) -> Optional[str]:
        """Choose the best resource for a task based on current optimization target"""
        if not resource_ids:
            return None
            
        # If there's only one resource, choose it
        if len(resource_ids) == 1:
            return resource_ids[0]
            
        # Calculate scores for each resource based on optimization target
        resource_scores = []
        for resource_id in resource_ids:
            resource = self.resource_manager.get_resource(resource_id)
            if not resource:
                continue
                
            # Calculate score based on optimization target
            if self.optimization_target == OptimizationTarget.THROUGHPUT:
                # Prefer resources with high computational capacity and low utilization
                computational_capacity = resource.capacity.get("computational", 0)
                utilization = resource.get_utilization("computational")
                score = computational_capacity * (1 - utilization)
                
            elif self.optimization_target == OptimizationTarget.LATENCY:
                # Prefer resources with low current load
                utilization = 0
                for feature in resource.current_load:
                    utilization = max(utilization, resource.get_utilization(feature))
                score = 1 - utilization
                
            elif self.optimization_target == OptimizationTarget.MEMORY:
                # Prefer resources with high available memory
                memory_capacity = resource.capacity.get("memory", 0)
                memory_utilization = resource.get_utilization("memory")
                score = memory_capacity * (1 - memory_utilization)
                
            elif self.optimization_target == OptimizationTarget.ENERGY:
                # Prefer more energy-efficient resources (simplified)
                if resource.resource_type == ProcessingUnit.GPU:
                    # GPUs tend to be more energy efficient for parallel tasks
                    score = 0.8 if task.execution_mode == ExecutionMode.PARALLEL else 0.3
                elif resource.resource_type == ProcessingUnit.TPU:
                    score = 0.9  # TPUs are very energy efficient for tensor operations
                elif resource.resource_type == ProcessingUnit.NPU:
                    score = 0.95  # Neuromorphic processors are extremely energy efficient
                elif resource.resource_type == ProcessingUnit.QPU:
                    score = 0.7 * (task.execution_mode == ExecutionMode.QUANTUM)
                else:  # CPU and others
                    score = 0.5
                    
                # Adjust for utilization - higher utilization is more energy efficient
                utilization = resource.get_utilization("computational")
                score *= (0.5 + 0.5 * utilization)  # Higher utilization is better for energy
                
            elif self.optimization_target == OptimizationTarget.RELIABILITY:
                # Prefer more reliable resources (simplified)
                if resource.state == ResourceState.DEGRADED:
                    score = 0.3
                elif resource.state == ResourceState.BUSY:
                    score = 0.7
                else:  # IDLE or other states
                    score = 1.0
                    
            else:  # BALANCED or other
                # Balanced approach
                computational_score = 1 - resource.get_utilization("computational")
                memory_score = 1 - resource.get_utilization("memory")
                # Simple average
                score = (computational_score + memory_score) / 2
            
            # Special case: if resource type matches preferred execution mode, boost score
            if task.execution_mode == ExecutionMode.PARALLEL and "parallel" in resource.capabilities:
                score *= 1.5
            elif task.execution_mode == ExecutionMode.QUANTUM and "quantum" in resource.capabilities:
                score *= 2.0
            elif task.execution_mode == ExecutionMode.NEUROMORPHIC and "neuromorphic" in resource.capabilities:
                score *= 2.0
            
            resource_scores.append((score, resource_id))
        
        # Sort by score (highest first)
        resource_scores.sort(reverse=True)
        
        # Return the best resource
        return resource_scores[0][1] if resource_scores else None
    
    def _start_task_execution(self, task: ComputationalTask, resource_id: str) -> None:
        """Start executing a task on a resource"""
        # Create thread for task execution
        thread = threading.Thread(
            target=self._execute_task,
            args=(task.task_id, resource_id),
            daemon=True
        )
        
        # Store in running tasks
        self.running_tasks[task.task_id] = (resource_id, thread)
        
        # Start execution
        thread.start()
    
    def _execute_task(self, task_id: str, resource_id: str) -> None:
        """Execute a task on a resource"""
        task = self.tasks.get(task_id)
        if not task:
            return
            
        task.status = "running"
        task.start_time = time.time()
        
        try:
            # Execute the task function
            if task.function:
                task.result = task.function(*task.args, **task.kwargs)
            else:
                # Simulate execution if no function provided
                self._simulate_task_execution(task)
                task.result = {"status": "simulated", "task_id": task_id}
            
            # Mark as completed
            with self._lock:
                task.status = "completed"
                task.completion_time = time.time()
                task.progress = 1.0
                self.completed_tasks.add(task_id)
                
                # Add to history
                self._add_to_history(task)
                
        except Exception as e:
            # Handle task failure
            with self._lock:
                task.status = "failed"
                task.completion_time = time.time()
                task.error = e
                self.failed_tasks.add(task_id)
                
                # Add to history
                self._add_to_history(task)
                
            logger.error(f"Task {task_id} failed: {str(e)}")
            
        finally:
            # Release resource
            self.resource_manager.release_resource(resource_id, task.requirements)
    
    def _simulate_task_execution(self, task: ComputationalTask) -> None:
        """Simulate task execution (for tasks without a function)"""
        # Calculate duration based on requirements
        duration = task.estimate_duration()
        
        # Report progress periodically
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time and task.status != "cancelled":
            # Update progress
            elapsed = time.time() - start_time
            progress = min(0.99, elapsed / duration)
            task.progress = progress
            
            # Sleep a bit
            time.sleep(min(0.1, (end_time - time.time()) / 10))
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        with self._lock:
            # Calculate throughput
            now = time.time()
            window = 60.0  # Use 1-minute window
            completed_in_window = sum(1 for task_id in self.completed_tasks
                                   if task_id in self.tasks and
                                   self.tasks[task_id].completion_time and
                                   now - self.tasks[task_id].completion_time < window)
            
            self.metrics.throughput = completed_in_window / window
            
            # Calculate average latency
            completed_tasks = [self.tasks[task_id] for task_id in self.completed_tasks
                             if task_id in self.tasks and
                             self.tasks[task_id].completion_time]
            
            if completed_tasks:
                total_latency = sum(task.completion_time - task.creation_time
                                  for task in completed_tasks)
                self.metrics.latency = total_latency / len(completed_tasks)
            
            # Memory usage (approximate based on resource utilization)
            total_memory = 0
            total_capacity = 0
            
            for resource in self.resource_manager.resources.values():
                if "memory" in resource.capacity:
                    memory_capacity = resource.capacity["memory"]
                    memory_used = resource.current_load.get("memory", 0)
                    total_memory += memory_used
                    total_capacity += memory_capacity
            
            self.metrics.memory_usage = total_memory
            
            # Energy consumption (simplified model)
            self.metrics.energy_consumption = 0
            
            for task_id, (resource_id, _) in self.running_tasks.items():
                task = self.tasks.get(task_id)
                resource = self.resource_manager.get_resource(resource_id)
                
                if task and resource:
                    # Different resource types have different energy profiles
                    if resource.resource_type == ProcessingUnit.CPU:
                        power_factor = 1.0
                    elif resource.resource_type == ProcessingUnit.GPU:
                        power_factor = 3.5
                    elif resource.resource_type == ProcessingUnit.TPU:
                        power_factor = 2.8
                    elif resource.resource_type == ProcessingUnit.QPU:
                        power_factor = 5.0
                    elif resource.resource_type == ProcessingUnit.NPU:
                        power_factor = 0.3  # Neuromorphic is very efficient
                    else:
                        power_factor = 1.0
                        
                    # Computational requirement affects energy use
                    computational_req = task.requirements.get("computational", 1.0)
                    
                    # Add to total energy consumption
                    self.metrics.energy_consumption += power_factor * computational_req
            
            # Reliability (based on task success rate)
            total_finished = len(self.completed_tasks) + len(self.failed_tasks)
            if total_finished > 0:
                self.metrics.reliability = len(self.completed_tasks) / total_finished
            
            # Adaptability (based on resource utilization balance)
            resource_utilizations = []
            for resource in self.resource_manager.resources.values():
                for feature in resource.capacity:
                    if feature in resource.current_load:
                        utilization = resource.current_load[feature] / resource.capacity[feature]
                        resource_utilizations.append(utilization)
            
            if resource_utilizations:
                # More balanced utilization means higher adaptability
                mean_util = sum(resource_utilizations) / len(resource_utilizations)
                variance = sum((u - mean_util) ** 2 for u in resource_utilizations) / len(resource_utilizations)
                # Convert variance to adaptability score (lower variance = higher adaptability)
                self.metrics.adaptability = max(0, min(1, 1.0 - variance * 4))
    
    def _cleanup_completed_tasks(self) -> None:
        """Clean up completed tasks from the running tasks list"""
        with self._lock:
            for task_id in list(self.running_tasks.keys()):
                task = self.tasks.get(task_id)
                if task and task.status in ["completed", "failed", "cancelled"]:
                    # Remove from running tasks
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
    
    def _add_to_history(self, task: ComputationalTask) -> None:
        """Add a completed task to the history"""
        # Create history entry
        history_entry = {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status,
            "creation_time": task.creation_time,
            "start_time": task.start_time,
            "completion_time": task.completion_time,
            "duration": task.completion_time - task.start_time if task.completion_time else None,
            "total_time": task.completion_time - task.creation_time if task.completion_time else None,
            "assigned_resource": task.assigned_resource,
            "requirements": task.requirements
        }
        
        # Add to history
        self.task_history.append(history_entry)
        
        # Trim history if needed
        if len(self.task_history) > self.history_limit:
            self.task_history = self.task_history[-self.history_limit:]
    
    def shutdown(self) -> None:
        """Shutdown the scheduler"""
        self._shutdown = True
        self.scheduler_thread.join(timeout=1.0)

class NeuralResourceOptimizer:
    """Advanced resource optimizer using neural network techniques"""
    
    def __init__(self, resource_manager: ResourceManager, task_scheduler: TaskScheduler):
        """Initialize the neural resource optimizer"""
        self.resource_manager = resource_manager
        self.task_scheduler = task_scheduler
        self.optimization_history = []
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.discount_factor = 0.95
        
        # Neural network parameters (simplified for demonstration)
        self.weights = {
            "throughput": 1.0,
            "latency": -1.0,  # Negative because lower latency is better
            "memory_usage": -0.5,  # Negative because lower memory usage is better
            "energy_consumption": -0.8,  # Negative because lower energy is better
            "reliability": 1.0,
            "adaptability": 0.7
        }
        
        # Current best configurations for different scenarios
        self.best_configs = {
            OptimizationTarget.THROUGHPUT: {},
            OptimizationTarget.LATENCY: {},
            OptimizationTarget.MEMORY: {},
            OptimizationTarget.ENERGY: {},
            OptimizationTarget.RELIABILITY: {},
            OptimizationTarget.ADAPTABILITY: {},
            OptimizationTarget.BALANCED: {}
        }
        
        # Spinning hebbian learning networks
        self.hebbian_networks = {target: self._create_hebbian_network() for target in OptimizationTarget}
        
        # Start optimization thread
        self.optimization_thread = None
        self._shutdown = False
    
    def _create_hebbian_network(self) -> Dict:
        """Create a simple hebbian network for learning resource allocations"""
        return {
            "weights": np.random.randn(6, 6) * 0.01,  # Small initial weights
            "activation": np.zeros(6),
            "learning_rate": 0.01,
            "decay_rate": 0.999,
            "stability": 0.5,
            "plasticity": 0.2
        }
    
    def start_optimization(self) -> None:
        """Start the optimization thread"""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self._shutdown = False
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
    
    def stop_optimization(self) -> None:
        """Stop the optimization thread"""
        self._shutdown = True
        if self.optimization_thread:
            self.optimization_thread.join(timeout=1.0)
    
    def _optimization_loop(self) -> None:
        """Main optimization loop"""
        while not self._shutdown:
            try:
                # Get current metrics
                metrics = self.task_scheduler.get_metrics()
                
                # Get current optimization target
                target = self.task_scheduler.optimization_target
                
                # Calculate reward based on metrics and weights
                reward = self._calculate_reward(metrics, target)
                
                # Update hebbian network
                self._update_hebbian_network(target, metrics, reward)
                
                # Use hebbian network to generate optimization suggestions
                suggestions = self._generate_optimization_suggestions(target)
                
                # Apply suggestions (if any)
                if suggestions:
                    self._apply_optimization_suggestions(suggestions)
                
                # Record optimization state
                self._record_optimization_state(metrics, target, reward, suggestions)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {str(e)}")
                
            # Sleep to avoid excessive CPU usage
            time.sleep(2.0)
    
    def _calculate_reward(self, metrics: OptimizationMetrics, target: OptimizationTarget) -> float:
        """Calculate reward based on current metrics and optimization target"""
        # Get metrics dictionary
        metrics_dict = metrics.to_dict()
        
        # Calculate weighted sum based on optimization target
        if target == OptimizationTarget.THROUGHPUT:
            return metrics_dict["throughput"]
            
        elif target == OptimizationTarget.LATENCY:
            # Lower latency is better, so use negative
            return 1.0 / max(0.001, metrics_dict["latency"])
            
        elif target == OptimizationTarget.MEMORY:
            # Lower memory usage is better, so use negative
            return 1.0 / max(0.001, metrics_dict["memory_usage"] / 1000)  # Normalize
            
        elif target == OptimizationTarget.ENERGY:
            # Lower energy consumption is better, so use negative
            return 1.0 / max(0.001, metrics_dict["energy_consumption"])
            
        elif target == OptimizationTarget.RELIABILITY:
            return metrics_dict["reliability"]
            
        elif target == OptimizationTarget.ADAPTABILITY:
            return metrics_dict["adaptability"]
            
        else:  # BALANCED or other
            # Weighted sum of all metrics
            reward = 0
            for metric, weight in self.weights.items():
                if metric in metrics_dict:
                    reward += metrics_dict[metric] * weight
            return reward
    
    def _update_hebbian_network(self, target: OptimizationTarget, 
                               metrics: OptimizationMetrics, reward: float) -> None:
        """Update hebbian network based on reward"""
        network = self.hebbian_networks[target]
        
        # Create input vector from metrics
        metrics_dict = metrics.to_dict()
        input_vector = np.array([
            metrics_dict["throughput"] / 10.0,  # Normalize
            1.0 / max(0.001, metrics_dict["latency"]),
            1.0 / max(0.001, metrics_dict["memory_usage"] / 1000),
            1.0 / max(0.001, metrics_dict["energy_consumption"]),
            metrics_dict["reliability"],
            metrics_dict["adaptability"]
        ])
        
        # Update network activation
        decay_rate = network["decay_rate"]
        network["activation"] = network["activation"] * decay_rate + input_vector * (1 - decay_rate)
        
        # Hebbian learning: neurons that fire together, wire together
        # Weight update proportional to pre and post-synaptic activity
        # dw_ij = lr * (reward * a_i * a_j - decay * w_ij)
        pre_post = np.outer(network["activation"], network["activation"])
        weight_decay = network["weights"] * 0.0001  # Small weight decay
        
        # Scale learning based on reward
        effective_lr = network["learning_rate"] * (1.0 + reward)
        
        # Update weights
        network["weights"] += effective_lr * (reward * pre_post - weight_decay)
    
    def _generate_optimization_suggestions(self, target: OptimizationTarget) -> Dict[str, Any]:
        """Generate optimization suggestions using the hebbian network"""
        network = self.hebbian_networks[target]
        
        # Generate suggestions based on network activations and weights
        suggestions = {}
        
        # Calculate suggestions for execution modes
        execution_mode_idx = np.argmax(np.sum(network["weights"][:, :4], axis=0))
        
        if execution_mode_idx == 0:
            suggestions["execution_mode"] = ExecutionMode.PARALLEL
        elif execution_mode_idx == 1:
            suggestions["execution_mode"] = ExecutionMode.SEQUENTIAL
        elif execution_mode_idx == 2:
            suggestions["execution_mode"] = ExecutionMode.DISTRIBUTED
        else:
            suggestions["execution_mode"] = ExecutionMode.ASYNCHRONOUS
        
        # Calculate suggestions for resource allocations
        resource_weights = np.sum(network["weights"], axis=1)
        
        # Normalize resource allocation weights
        total_weight = np.sum(np.abs(resource_weights))
        if total_weight > 0:
            normalized_weights = resource_weights / total_weight
        else:
            normalized_weights = np.ones_like(resource_weights) / len(resource_weights)
        
        # Resource allocation suggestions
        suggestions["resource_allocation"] = {
            "CPU": max(0.1, normalized_weights[0]),
            "GPU": max(0, normalized_weights[1]),
            "TPU": max(0, normalized_weights[2]),
            "NPU": max(0, normalized_weights[3]),
            "QPU": max(0, normalized_weights[4])
        }
        
        # Exploration: occasionally try random allocation
        if random.random() < self.exploration_rate:
            random_allocation = np.random.random(5)
            random_allocation = random_allocation / np.sum(random_allocation)
            
            suggestions["resource_allocation"] = {
                "CPU": random_allocation[0],
                "GPU": random_allocation[1],
                "TPU": random_allocation[2],
                "NPU": random_allocation[3],
                "QPU": random_allocation[4]
            }
        
        return suggestions
    
    def _apply_optimization_suggestions(self, suggestions: Dict[str, Any]) -> None:
        """Apply the optimization suggestions"""
        # This is a simplified implementation
        # In a real system, this would involve more complex resource management
        
        # Apply execution mode suggestion to new tasks
        if "execution_mode" in suggestions:
            # This could modify a default setting for new tasks
            pass
            
        # Apply resource allocation suggestions
        if "resource_allocation" in suggestions:
            allocation = suggestions["resource_allocation"]
            
            # Adjust resource allocation (simplified)
            # In a real system, this would involve reconfiguring resource schedulers
            
            # For demonstration, just log the suggested allocation
            logger.info(f"Suggested resource allocation: {allocation}")
    
    def _record_optimization_state(self, metrics: OptimizationMetrics, 
                                 target: OptimizationTarget, 
                                 reward: float,
                                 suggestions: Dict[str, Any]) -> None:
        """Record the current optimization state for analysis"""
        state = {
            "timestamp": time.time(),
            "metrics": metrics.to_dict(),
            "target": target.name,
            "reward": reward,
            "suggestions": {
                k: (v.name if hasattr(v, 'name') else v)
                for k, v in suggestions.items()
            }
        }
        
        self.optimization_history.append(state)
        
        # Limit history size
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
            
        # Update best configuration if this is the best reward so far for this target
        best_reward = self.best_configs[target].get("reward", float("-inf"))
        if reward > best_reward:
            self.best_configs[target] = {
                "reward": reward,
                "metrics": metrics.to_dict(),
                "suggestions": suggestions,
                "timestamp": time.time()
            }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get a report of the optimization results"""
        report = {
            "current_target": self.task_scheduler.optimization_target.name,
            "optimization_runs": len(self.optimization_history),
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "best_configs": {target.name: config for target, config in self.best_configs.items() 
                           if "reward" in config},
            "weights": self.weights
        }
        
        # Add recent history
        if self.optimization_history:
            recent_history = self.optimization_history[-10:]  # Last 10 entries
            report["recent_history"] = recent_history
            
            # Calculate improvement trends
            if len(self.optimization_history) >= 2:
                first_reward = self.optimization_history[0]["reward"]
                last_reward = self.optimization_history[-1]["reward"]
                reward_improvement = (last_reward - first_reward) / max(0.001, first_reward)
                report["reward_improvement"] = reward_improvement
                
        return report
        
    def optimize_for_workload(self, workload_type: str) -> Dict[str, Any]:
        """Optimize system for a specific workload type"""
        # This method would optimize the system for specific workload types
        # For demonstration, we'll just return a predefined configuration
        
        if workload_type == "compute_intensive":
            # For compute-intensive workloads
            target = OptimizationTarget.THROUGHPUT
            execution_mode = ExecutionMode.PARALLEL
            resource_allocation = {
                "CPU": 0.3,
                "GPU": 0.6,
                "TPU": 0.1,
                "NPU": 0.0,
                "QPU": 0.0
            }
            
        elif workload_type == "memory_intensive":
            # For memory-intensive workloads
            target = OptimizationTarget.MEMORY
            execution_mode = ExecutionMode.SEQUENTIAL
            resource_allocation = {
                "CPU": 0.7,
                "GPU": 0.2,
                "TPU": 0.0,
                "NPU": 0.1,
                "QPU": 0.0
            }
            
        elif workload_type == "quantum":
            # For quantum workloads
            target = OptimizationTarget.BALANCED
            execution_mode = ExecutionMode.QUANTUM
            resource_allocation = {
                "CPU": 0.2,
                "GPU": 0.0,
                "TPU": 0.0,
                "NPU": 0.0,
                "QPU": 0.8
            }
            
        elif workload_type == "neuromorphic":
            # For neuromorphic workloads
            target = OptimizationTarget.ENERGY
            execution_mode = ExecutionMode.NEUROMORPHIC
            resource_allocation = {
                "CPU": 0.1,
                "GPU": 0.0,
                "TPU": 0.0,
                "NPU": 0.9,
                "QPU": 0.0
            }
            
        else:  # "balanced" or unknown
            # Balanced workload
            target = OptimizationTarget.BALANCED
            execution_mode = ExecutionMode.HYBRID
            resource_allocation = {
                "CPU": 0.4,
                "GPU": 0.3,
                "TPU": 0.1,
                "NPU": 0.1,
                "QPU": 0.1
            }
            
        # Apply the configuration
        self.task_scheduler.set_optimization_target(target)
            
        return {
            "workload_type": workload_type,
            "optimization_target": target.name,
            "execution_mode": execution_mode.name,
            "resource_allocation": resource_allocation
        }
        
class NeuromorphicRuntimeSystem:
    """Main system that integrates all neuromorphic optimization components"""
    
    def __init__(self):
        """Initialize the neuromorphic runtime system"""
        self.resource_manager = ResourceManager()
        
        # Initialize system resources
        self.resource_manager.get_system_resources()
        
        # Create task scheduler
        self.task_scheduler = TaskScheduler(self.resource_manager)
        
        # Create neural optimizer
        self.neural_optimizer = NeuralResourceOptimizer(
            self.resource_manager,
            self.task_scheduler
        )
        
        # System status tracking
        self.start_time = time.time()
        self.status_history = []
        self.current_workload_type = "balanced"
        
        # Task templates for quick submission
        self.task_templates = self._create_task_templates()
    
    def _create_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined task templates"""
        return {
            "compute_basic": {
                "name": "compute_basic",
                "requirements": {"computational": 1.0, "memory": 100.0},
                "execution_mode": ExecutionMode.SEQUENTIAL,
                "preferred_resource_types": [ProcessingUnit.CPU]
            },
            "compute_intensive": {
                "name": "compute_intensive",
                "requirements": {"computational": 10.0, "memory": 500.0},
                "execution_mode": ExecutionMode.PARALLEL,
                "preferred_resource_types": [ProcessingUnit.GPU, ProcessingUnit.CPU]
            },
            "memory_intensive": {
                "name": "memory_intensive",
                "requirements": {"computational": 2.0, "memory": 2000.0},
                "execution_mode": ExecutionMode.SEQUENTIAL,
                "preferred_resource_types": [ProcessingUnit.CPU]
            },
            "gpu_compute": {
                "name": "gpu_compute",
                "requirements": {"computational": 8.0, "memory": 1000.0, "cores": 128.0},
                "execution_mode": ExecutionMode.PARALLEL,
                "preferred_resource_types": [ProcessingUnit.GPU]
            },
            "quantum_simulation": {
                "name": "quantum_simulation",
                "requirements": {"computational": 5.0, "memory": 500.0, "qubits": 4.0},
                "execution_mode": ExecutionMode.QUANTUM,
                "preferred_resource_types": [ProcessingUnit.QPU, ProcessingUnit.GPU]
            },
            "neural_simulation": {
                "name": "neural_simulation",
                "requirements": {"computational": 3.0, "memory": 800.0, "neurons": 512.0},
                "execution_mode": ExecutionMode.NEUROMORPHIC,
                "preferred_resource_types": [ProcessingUnit.NPU, ProcessingUnit.GPU]
            }
        }
    
    def start(self) -> None:
        """Start the system"""
        # Start neural optimizer
        self.neural_optimizer.start_optimization()
        
        logger.info("Neuromorphic Runtime System started")
    
    def stop(self) -> None:
        """Stop the system"""
        # Stop neural optimizer
        self.neural_optimizer.stop_optimization()
        
        logger.info("Neuromorphic Runtime System stopped")
    
    def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a task to the system"""
        # Create task from data
        task = ComputationalTask(
            name=task_data.get("name", "unnamed_task"),
            requirements=task_data.get("requirements", {"computational": 1.0}),
            dependencies=task_data.get("dependencies", []),
            priority=task_data.get("priority", 0.5),
            execution_mode=task_data.get("execution_mode", ExecutionMode.SEQUENTIAL),
            preferred_resource_types=task_data.get("preferred_resource_types", []),
            function=task_data.get("function"),
            args=task_data.get("args", []),
            kwargs=task_data.get("kwargs", {}),
            deadline=task_data.get("deadline"),
            metadata=task_data.get("metadata", {})
        )
        
        # Submit task to scheduler
        return self.task_scheduler.submit_task(task)
    
    def submit_task_from_template(self, template_name: str, **kwargs) -> str:
        """Submit a task using a predefined template"""
        if template_name not in self.task_templates:
            raise ValueError(f"Unknown task template: {template_name}")
            
        # Get template
        template = self.task_templates[template_name].copy()
        
        # Override with provided kwargs
        template.update(kwargs)
        
        # Submit task
        return self.submit_task(template)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        task = self.task_scheduler.get_task(task_id)
        if task:
            return task.to_dict()
        else:
            return {"error": "Task not found", "task_id": task_id}
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = None) -> Any:
        """Get task result"""
        try:
            return self.task_scheduler.get_task_result(task_id, wait, timeout)
        except Exception as e:
            return {"error": str(e), "task_id": task_id}
    
    def optimize_for_workload(self, workload_type: str) -> Dict[str, Any]:
        """Optimize the system for a specific workload type"""
        result = self.neural_optimizer.optimize_for_workload(workload_type)
        self.current_workload_type = workload_type
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        # Get scheduler stats
        scheduler_stats = self.task_scheduler.get_stats()
        
        # Get optimizer report
        optimizer_report = self.neural_optimizer.get_optimization_report()
        
        # Get resource stats
        resources = {}
        for resource_id, resource in self.resource_manager.resources.items():
            resources[resource_id] = resource.to_dict()
        
        status = {
            "timestamp": time.time(),
            "formatted_time": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_seconds": time.time() - self.start_time,
            "scheduler_stats": scheduler_stats,
            "optimization_target": self.task_scheduler.optimization_target.name,
            "current_workload_type": self.current_workload_type,
            "resources": resources,
            "metrics": self.task_scheduler.get_metrics().to_dict(),
        }
        
        # Add to history
        self.status_history.append({
            "timestamp": status["timestamp"],
            "metrics": status["metrics"],
            "scheduler_stats": {
                k: v for k, v in status["scheduler_stats"].items()
                if k not in ["metrics"]
            }
        })
        
        # Trim history
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]
        
        return status

# Example usage of the system
def run_example():
    print(f"Current Date/Time: 2025-07-24 14:31:54")
    print(f"User: Shriram-2005")
    
    print("\n===== Advanced Neuromorphic Runtime Optimizer Example =====")
    
    # Create the runtime system
    system = NeuromorphicRuntimeSystem()
    
    # Start the system
    system.start()
    
    print("\nSystem started. Initializing resources...")
    time.sleep(2)  # Give time for system initialization
    
    # Get initial system status
    status = system.get_system_status()
    
    print("\nSystem Status:")
    print(f"  Time: {status['formatted_time']}")
    print(f"  Uptime: {status['uptime_seconds']:.2f} seconds")
    print(f"  Current workload type: {status['current_workload_type']}")
    print(f"  Optimization target: {status['optimization_target']}")
    print("\nAvailable Resources:")
    
    for resource_id, resource in status['resources'].items():
        print(f"  - {resource['resource_type']} (ID: {resource_id})")
        print(f"    State: {resource['state']}")
        print(f"    Capabilities: {', '.join(resource['capabilities'])}")
    
    # Optimize for compute intensive workload
    print("\nOptimizing system for compute-intensive workload...")
    optimize_result = system.optimize_for_workload("compute_intensive")
    print(f"  Target: {optimize_result['optimization_target']}")
    print(f"  Execution Mode: {optimize_result['execution_mode']}")
    
    # Submit some tasks
    print("\nSubmitting tasks...")
    task_ids = []
    
    # Submit compute-intensive tasks
    for i in range(3):
        task_id = system.submit_task_from_template(
            "compute_intensive",
            name=f"compute_task_{i+1}"
        )
        task_ids.append(task_id)
        print(f"  Submitted compute-intensive task {i+1}: {task_id}")
    
    # Submit GPU compute task
    task_id = system.submit_task_from_template("gpu_compute")
    task_ids.append(task_id)
    print(f"  Submitted GPU compute task: {task_id}")
    
    # Submit quantum simulation task
    task_id = system.submit_task_from_template("quantum_simulation")
    task_ids.append(task_id)
    print(f"  Submitted quantum simulation task: {task_id}")
    
    # Submit neural simulation task
    task_id = system.submit_task_from_template("neural_simulation")
    task_ids.append(task_id)
    print(f"  Submitted neural simulation task: {task_id}")
    
    # Wait for tasks to be scheduled and processed
    print("\nWaiting for tasks to be processed...")
    time.sleep(5)
    
    # Check task statuses
    print("\nTask Statuses:")
    for task_id in task_ids:
        status = system.get_task_status(task_id)
        print(f"  Task {status['name']} (ID: {task_id}): {status['status']}")
        if status['assigned_resource']:
            print(f"    Assigned to: {status['assigned_resource']}")
        print(f"    Progress: {status['progress'] * 100:.1f}%")
    
    # Wait a bit longer
    time.sleep(3)
    
    # Get final system status
    status = system.get_system_status()
    
    print("\nFinal System Status:")
    print(f"  Tasks completed: {status['scheduler_stats']['tasks_completed']}")
    print(f"  Tasks pending: {status['scheduler_stats']['tasks_pending']}")
    print(f"  Tasks running: {status['scheduler_stats']['tasks_running']}")
    
    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in status['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Stop the system
    system.stop()
    
    print("\nSystem stopped.")
    print("\nAdvanced Neuromorphic Runtime Optimizer implementation complete!")
    print("The system successfully implements neuromorphic-inspired resource optimization.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    run_example()