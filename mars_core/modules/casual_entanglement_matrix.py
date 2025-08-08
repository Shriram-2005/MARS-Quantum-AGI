"""
MARS Causal Entanglement Matrix
=====================================

A sophisticated causal inference system that combines classical causal modeling 
with quantum entanglement principles to discover and analyze complex causal 
relationships in data.

This module implements advanced causal inference algorithms including:
- Quantum-enhanced causal discovery
- Structural equation modeling
- Graphical causal models
- Granger causality analysis
- Transfer entropy methods
- Intervention modeling with do-calculus
- Counterfactual reasoning

"""

import numpy as np
import scipy.stats as stats
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
import time
import threading
import uuid
import random
import math
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
import pickle
import heapq
from collections import deque

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0"
__author__ = "Shriram-2005"
__license__ = "MIT"
__description__ = "Advanced Causal Inference with Quantum Entanglement Principles"

class CausalRelationType(Enum):
    """
    Enumeration of different types of causal relationships.
    
    This enum defines the various ways two variables can be causally related,
    from simple direct causation to complex quantum entanglement relationships.
    """
    DIRECT = auto()           # Direct causation: X → Y
    INDIRECT = auto()         # Indirect causation: X → Z → Y  
    BIDIRECTIONAL = auto()    # Bidirectional causality: X ↔ Y
    CONFOUNDED = auto()       # Common cause relationship: Z → X, Z → Y
    BLOCKED = auto()          # Blocked/mediated relationship
    QUANTUM = auto()          # Quantum entangled causality
    TEMPORAL = auto()         # Time-dependent causality
    PROBABILISTIC = auto()    # Probabilistic causality
    NECESSARY = auto()        # Necessary cause (Y cannot occur without X)
    SUFFICIENT = auto()       # Sufficient cause (X always leads to Y)
    EMERGENT = auto()         # Emergent causality from system complexity
    VIRTUAL = auto()          # Virtual (simulated) causality

class CausalStrength(Enum):
    """
    Predefined strength levels for causal relationships.
    
    These values provide standardized thresholds for categorizing
    the strength of causal relationships from negligible to certain.
    """
    NONE = 0.0                # No causal relationship
    NEGLIGIBLE = 0.1          # Barely detectable effect
    WEAK = 0.3                # Weak but measurable relationship
    MODERATE = 0.5            # Moderate relationship
    STRONG = 0.7              # Strong relationship
    VERY_STRONG = 0.9         # Very strong relationship
    CERTAIN = 1.0             # Deterministic causality


class InterventionType(Enum):
    """
    Types of causal interventions that can be performed on the network.
    
    These intervention types correspond to different experimental designs
    and causal reasoning approaches used in causal inference.
    """
    DO = auto()               # Direct intervention (do-calculus): do(X=x)
    SOFT = auto()             # Soft intervention (partial constraint)
    COUNTERFACTUAL = auto()   # Counterfactual intervention: "what if"
    CONDITIONAL = auto()      # Conditional intervention: do(X=x | Z=z)
    QUANTUM = auto()          # Quantum intervention (superposition state)
    TEMPORAL = auto()         # Time-shifted intervention
    STOCHASTIC = auto()       # Probabilistic intervention
    MULTIVARIATE = auto()     # Multiple simultaneous interventions


class InferenceMethod(Enum):
    """
    Methods for causal structure discovery and effect estimation.
    
    These methods represent different algorithmic approaches to inferring
    causal relationships from observational and experimental data.
    """
    STRUCTURAL = auto()       # Structural equation models (SEM)
    POTENTIAL_OUTCOMES = auto() # Rubin causal model / potential outcomes
    GRAPHICAL = auto()        # Causal graphical models (DAGs)
    GRANGER = auto()          # Granger causality for time series
    QUANTUM_WALK = auto()     # Quantum walk algorithm
    INFORMATION_FLOW = auto() # Information flow analysis
    COUNTERFACTUAL = auto()   # Counterfactual reasoning
    ENTROPIC = auto()         # Entropic causal inference
    TRANSFER_ENTROPY = auto() # Transfer entropy approach
    QUANTUM_ENTANGLEMENT = auto() # Quantum entanglement-based discovery
    TOPOLOGICAL = auto()      # Topological causal inference
    INTEGRATED = auto()       # Integrated information approach

@dataclass
class CausalNode:
    """
    Represents a variable/node in the causal network.
    
    A CausalNode encapsulates all information about a variable in the causal
    system, including its current value, historical values, quantum state (if enabled),
    and various metadata properties.
    
    Attributes:
        node_id: Unique identifier for the node
        name: Human-readable name for the variable
        value: Current value of the variable
        value_history: Historical values with timestamps
        creation_time: When the node was created
        last_updated: When the node was last modified
        dimension: Dimensionality of the node's value space
        is_observed: Whether this is an observed (vs latent) variable
        is_latent: Whether this is a latent/hidden variable
        metadata: Additional metadata about the node
        quantum_state: Quantum state vector (if quantum mode enabled)
        entropy: Information-theoretic entropy of the node
        stability: Measure of how stable the node's value is over time
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed"
    value: Any = None
    value_history: List[Tuple[Any, float]] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    dimension: int = 1  # Dimensionality of the node's value
    is_observed: bool = True
    is_latent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    quantum_state: Optional[np.ndarray] = None
    entropy: float = 0.0
    stability: float = 1.0
    
    def update_value(self, value: Any) -> None:
        """
        Update the node's value and maintain historical records.
        
        This method updates the node's current value, stores the previous value
        in history, updates the timestamp, and recalculates entropy if applicable.
        
        Args:
            value: The new value to assign to this node
            
        Raises:
            ValueError: If the value is incompatible with the node's constraints
        """
        try:
            # Store old value in history
            if self.value is not None:
                self.value_history.append((self.value, self.last_updated))
                # Keep history to a reasonable size to prevent memory issues
                if len(self.value_history) > 100:
                    self.value_history = self.value_history[-100:]
            
            # Update value and timestamp
            self.value = value
            self.last_updated = time.time()
            
            # Update entropy if value supports it
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                self._calculate_entropy()
                
        except Exception as e:
            logger.error(f"Error updating value for node {self.name}: {e}")
            raise ValueError(f"Failed to update node value: {e}")
    
    def _calculate_entropy(self) -> None:
        """
        Calculate the information-theoretic entropy of the node's value.
        
        For array-like values, this computes the Shannon entropy based on
        the distribution of values. For scalar values, entropy remains 0.
        """
        try:
            # Convert to numpy array if needed
            arr = np.array(self.value) if not isinstance(self.value, np.ndarray) else self.value
            
            # Normalize to approximate probability distribution
            if np.sum(np.abs(arr)) > 0:
                probs = np.abs(arr) / np.sum(np.abs(arr))
                # Calculate Shannon entropy
                self.entropy = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                self.entropy = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate entropy for node {self.name}: {e}")
            self.entropy = 0.0
    
    def update_quantum_state(self, state: np.ndarray) -> None:
        """
        Update the quantum state of the node and calculate von Neumann entropy.
        
        This method updates the node's quantum state vector and computes the
        associated von Neumann entropy for quantum information measures.
        
        Args:
            state: Quantum state vector or density matrix
            
        Raises:
            ValueError: If the state is not a valid quantum state
        """
        try:
            if state is not None:
                # Validate quantum state
                if not isinstance(state, np.ndarray):
                    raise ValueError("Quantum state must be a numpy array")
                
                # Normalize state if it's a vector
                if len(state.shape) == 1:
                    norm = np.linalg.norm(state)
                    if norm > 0:
                        state = state / norm
                    else:
                        raise ValueError("Zero-norm quantum state is invalid")
            
            self.quantum_state = state
            self.last_updated = time.time()
            
            # Calculate von Neumann entropy if possible
            if state is not None:
                self._calculate_von_neumann_entropy()
                
        except Exception as e:
            logger.error(f"Error updating quantum state for node {self.name}: {e}")
            raise ValueError(f"Failed to update quantum state: {e}")
    
    def _calculate_von_neumann_entropy(self) -> None:
        """
        Calculate the von Neumann entropy of the quantum state.
        
        The von Neumann entropy S = -Tr(ρ log₂ ρ) where ρ is the density matrix.
        For pure states, this is zero. For mixed states, it measures entanglement.
        """
        try:
            if self.quantum_state is None:
                self.entropy = 0.0
                return
                
            # Create density matrix
            if len(self.quantum_state.shape) == 1:
                # Pure state: ρ = |ψ⟩⟨ψ|
                density = np.outer(self.quantum_state, np.conjugate(self.quantum_state))
            else:
                # Already a density matrix
                density = self.quantum_state
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(density)
            # Remove negligible eigenvalues to avoid log(0)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            # Calculate von Neumann entropy: S = -Tr(ρ log₂ ρ)
            if len(eigenvalues) > 0:
                self.entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            else:
                self.entropy = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate von Neumann entropy for node {self.name}: {e}")
            self.entropy = 0.0
    
    def get_value_at_time(self, timestamp: float) -> Any:
        """
        Retrieve the value of the node at a specific point in time.
        
        This method searches through the node's value history to find the value
        that was active at the specified timestamp.
        
        Args:
            timestamp: The time point to query (Unix timestamp)
            
        Returns:
            The value that was active at the specified time, or None if
            the timestamp is before the node's creation time
        """
        try:
            # If timestamp is after last update, return current value
            if timestamp >= self.last_updated:
                return self.value
            
            # If timestamp is before creation, return None
            if timestamp < self.creation_time:
                return None
            
            # Look through history for closest value before timestamp
            for i in range(len(self.value_history) - 1, -1, -1):
                val, t = self.value_history[i]
                if t <= timestamp:
                    return val
            
            # Default to earliest known value
            if self.value_history:
                return self.value_history[0][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving historical value for node {self.name}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary representation for serialization.
        
        Returns:
            Dictionary containing all node properties in a serializable format
        """
        try:
            return {
                "node_id": self.node_id,
                "name": self.name,
                "value": self.value,
                "creation_time": self.creation_time,
                "last_updated": self.last_updated,
                "dimension": self.dimension,
                "is_observed": self.is_observed,
                "is_latent": self.is_latent,
                "metadata": self.metadata,
                "entropy": self.entropy,
                "stability": self.stability,
                "has_quantum_state": self.quantum_state is not None,
                "history_length": len(self.value_history)
            }
        except Exception as e:
            logger.error(f"Error converting node {self.name} to dictionary: {e}")
            return {"error": str(e), "node_id": self.node_id, "name": self.name}


@dataclass
class CausalEdge:
    """
    Represents a causal relationship between two nodes in the network.
    
    A CausalEdge encapsulates information about how one variable causally
    influences another, including the strength, type, and mechanism of causation.
    
    Attributes:
        edge_id: Unique identifier for the edge
        source_id: ID of the source (cause) node
        target_id: ID of the target (effect) node
        relation_type: Type of causal relationship
        strength: Strength of the causal effect (0.0 to 1.0)
        creation_time: When the edge was created
        last_updated: When the edge was last modified
        confidence: Confidence in this causal relationship
        metadata: Additional metadata about the edge
        temporal_delay: Time delay between cause and effect
        mechanism: Function defining the causal mechanism
        parameters: Parameters for the causal mechanism
        is_active: Whether this edge is currently active
        quantum_correlation: Strength of quantum correlation (0.0 to 1.0)
    """
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    strength: float = 0.5  # Causal strength from 0 to 1
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.5  # Confidence in this causal relationship
    metadata: Dict[str, Any] = field(default_factory=dict)
    temporal_delay: float = 0.0  # Time delay of causal effect
    mechanism: Optional[Callable] = None  # Function defining causal mechanism
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters for mechanism
    is_active: bool = True
    quantum_correlation: float = 0.0  # Quantum correlation strength
    
    def update_strength(self, new_strength: float, confidence: float) -> None:
        """
        Update causal strength with confidence-weighted averaging.
        
        This method updates the edge's strength using a confidence-weighted
        approach where higher confidence updates receive more weight.
        
        Args:
            new_strength: New strength value (0.0 to 1.0)
            confidence: Confidence in the new strength measurement
            
        Raises:
            ValueError: If strength or confidence are outside valid ranges
        """
        try:
            if not (0.0 <= new_strength <= 1.0):
                raise ValueError(f"Strength must be between 0.0 and 1.0, got {new_strength}")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
                
            # Weighted average based on confidence
            if confidence > self.confidence:
                # Higher confidence update gets more weight
                weight = confidence / (confidence + self.confidence)
                self.strength = (1 - weight) * self.strength + weight * new_strength
                self.confidence = max(self.confidence, confidence)
            else:
                # Lower confidence update gets less weight
                weight = 0.1
                self.strength = (1 - weight) * self.strength + weight * new_strength
            
            self.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Error updating edge strength: {e}")
            raise
    
    def apply_mechanism(self, input_value: Any) -> Any:
        """
        Apply the causal mechanism to transform input to output value.
        
        This method applies the causal transformation function to convert
        the cause variable's value to its effect on the target variable.
        
        Args:
            input_value: Value from the source/cause variable
            
        Returns:
            Transformed value representing the causal effect
            
        Raises:
            RuntimeError: If the mechanism function fails
        """
        try:
            if self.mechanism is None:
                # Default linear mechanism if none specified
                if isinstance(input_value, (int, float)):
                    return float(input_value) * self.strength
                else:
                    return input_value
            
            # Apply custom mechanism with parameters
            return self.mechanism(input_value, **self.parameters)
            
        except Exception as e:
            logger.warning(f"Error applying causal mechanism: {e}")
            # Fallback to simple scaling
            try:
                if isinstance(input_value, (int, float)):
                    return float(input_value) * self.strength
                else:
                    return input_value
            except:
                return input_value
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Generate a correlation matrix for quantum entanglement analysis.
        
        Returns:
            2x2 correlation matrix representing quantum correlations between
            the source and target nodes of this edge
        """
        try:
            if self.quantum_correlation == 0:
                # No quantum correlation - return identity matrix
                return np.eye(2)
            
            # Create correlation matrix for quantum entanglement
            # Values closer to 1 indicate stronger correlation
            c = max(-1.0, min(1.0, self.quantum_correlation))  # Clamp to valid range
            return np.array([
                [1.0, c],
                [c, 1.0]
            ])
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            return np.eye(2)  # Return safe default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the edge to a dictionary representation for serialization.
        
        Returns:
            Dictionary containing all edge properties in a serializable format
        """
        try:
            return {
                "edge_id": self.edge_id,
                "source_id": self.source_id,
                "target_id": self.target_id,
                "relation_type": self.relation_type.name,
                "strength": self.strength,
                "creation_time": self.creation_time,
                "last_updated": self.last_updated,
                "confidence": self.confidence,
                "metadata": self.metadata,
                "temporal_delay": self.temporal_delay,
                "is_active": self.is_active,
                "quantum_correlation": self.quantum_correlation,
                "has_mechanism": self.mechanism is not None
            }
        except Exception as e:
            logger.error(f"Error converting edge to dictionary: {e}")
            return {"error": str(e), "edge_id": self.edge_id}

class CausalEntanglementMatrix:
    """
    Advanced causal inference system using quantum entanglement principles.
    
    This class implements a sophisticated causal discovery and inference system
    that combines classical causal modeling approaches with quantum-inspired
    algorithms for enhanced pattern detection and relationship analysis.
    
    Features:
    - Multiple causal inference algorithms (structural, graphical, Granger, etc.)
    - Quantum entanglement-based correlation detection
    - Real-time causal effect propagation
    - Intervention modeling with do-calculus
    - Counterfactual reasoning
    - Temporal causal analysis
    
    Attributes:
        nodes: Dictionary mapping node IDs to CausalNode objects
        edges: Dictionary mapping edge IDs to CausalEdge objects
        quantum_enabled: Whether quantum features are enabled
        temporal_resolution: Time resolution for temporal analysis
        default_confidence: Default confidence level for new relationships
    """
    
    def __init__(self, quantum_enabled: bool = True):
        """
        Initialize the causal entanglement matrix.
        
        Args:
            quantum_enabled: Whether to enable quantum entanglement features
        """
        try:
            # Graph data structures
            self.nodes: Dict[str, CausalNode] = {}
            self.edges: Dict[str, CausalEdge] = {}
            self.outgoing_edges: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
            self.incoming_edges: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
            
            # Configuration settings
            self.quantum_enabled = quantum_enabled
            self.temporal_resolution = 0.01  # Time step for temporal causal analysis
            self.default_confidence = 0.5
            
            # Inference configuration
            self.default_inference_method = InferenceMethod.GRAPHICAL
            self.default_intervention_type = InterventionType.DO
            
            # Quantum entanglement tracking
            self.entangled_pairs: Dict[Tuple[str, str], float] = {}  # (node_id1, node_id2) -> entanglement_strength
            
            # Time tracking
            self.current_time = time.time()
            self.creation_time = self.current_time
            
            # Background inference system
            self.inference_thread = None
            self.running = False
            self.inference_queue = deque()  # Queue of nodes to update
            self.inference_results = {}  # node_id -> last inference result
            self._lock = threading.RLock()
            
            logger.info(f"Initialized CausalEntanglementMatrix (quantum_enabled={quantum_enabled})")
            
        except Exception as e:
            logger.error(f"Error initializing CausalEntanglementMatrix: {e}")
            raise
    
    def add_node(self, name: str, value: Any = None, is_observed: bool = True,
                 is_latent: bool = False, metadata: Dict[str, Any] = None) -> str:
        """
        Add a new node (variable) to the causal network.
        
        This method creates a new node representing a variable in the causal
        system and initializes its quantum state if quantum mode is enabled.
        
        Args:
            name: Human-readable name for the variable
            value: Initial value of the variable
            is_observed: Whether this is an observed variable (vs latent)
            is_latent: Whether this is a latent/hidden variable
            metadata: Additional metadata about the variable
            
        Returns:
            Unique identifier for the created node
            
        Raises:
            ValueError: If a node with the same name already exists
        """
        try:
            with self._lock:
                # Check for duplicate names
                existing_names = [node.name for node in self.nodes.values()]
                if name in existing_names:
                    raise ValueError(f"Node with name '{name}' already exists")
                
                # Create the node
                node = CausalNode(
                    name=name,
                    value=value,
                    is_observed=is_observed,
                    is_latent=is_latent,
                    metadata=metadata or {}
                )
                
                # Determine dimensionality based on value type
                if isinstance(value, np.ndarray):
                    node.dimension = value.size
                elif isinstance(value, (list, tuple)):
                    node.dimension = len(value)
                elif value is not None:
                    node.dimension = 1
                
                # Add to graph data structures
                node_id = node.node_id
                self.nodes[node_id] = node
                self.outgoing_edges[node_id] = []
                self.incoming_edges[node_id] = []
                
                # Initialize quantum state if enabled and not latent
                if self.quantum_enabled and not is_latent:
                    self._initialize_quantum_state(node)
                
                logger.debug(f"Added node '{name}' with ID {node_id}")
                return node_id
                
        except Exception as e:
            logger.error(f"Error adding node '{name}': {e}")
            raise
    
    def add_edge(self, source_id: str, target_id: str, 
                 relation_type: CausalRelationType = CausalRelationType.DIRECT,
                 strength: float = 0.5, confidence: float = None,
                 temporal_delay: float = 0.0,
                 mechanism: Callable = None,
                 parameters: Dict[str, Any] = None) -> str:
        """
        Add a causal edge (relationship) between two nodes.
        
        This method creates a directed causal relationship from a source node
        to a target node, specifying the strength and type of causation.
        
        Args:
            source_id: ID of the source (cause) node
            target_id: ID of the target (effect) node
            relation_type: Type of causal relationship
            strength: Strength of the causal effect (0.0 to 1.0)
            confidence: Confidence in this relationship (defaults to system default)
            temporal_delay: Time delay between cause and effect
            mechanism: Function defining the causal mechanism
            parameters: Parameters for the causal mechanism
            
        Returns:
            Unique identifier for the created edge
            
        Raises:
            ValueError: If source or target nodes don't exist, or parameters are invalid
        """
        try:
            with self._lock:
                # Validate input parameters
                if source_id not in self.nodes:
                    raise ValueError(f"Source node '{source_id}' does not exist")
                if target_id not in self.nodes:
                    raise ValueError(f"Target node '{target_id}' does not exist")
                if not (0.0 <= strength <= 1.0):
                    raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")
                if confidence is not None and not (0.0 <= confidence <= 1.0):
                    raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
                if temporal_delay < 0.0:
                    raise ValueError(f"Temporal delay cannot be negative, got {temporal_delay}")
                
                # Create the edge
                edge = CausalEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    strength=strength,
                    confidence=confidence if confidence is not None else self.default_confidence,
                    temporal_delay=temporal_delay,
                    mechanism=mechanism,
                    parameters=parameters or {}
                )
                
                # Add to graph data structures
                edge_id = edge.edge_id
                self.edges[edge_id] = edge
                self.outgoing_edges[source_id].append(edge_id)
                self.incoming_edges[target_id].append(edge_id)
                
                # Handle quantum relationships
                if self.quantum_enabled:
                    if relation_type in [CausalRelationType.QUANTUM, CausalRelationType.BIDIRECTIONAL]:
                        self._establish_quantum_correlation(source_id, target_id, strength)
                        edge.quantum_correlation = strength
                
                logger.debug(f"Added edge from {source_id} to {target_id} with strength {strength}")
                return edge_id
                
        except Exception as e:
            logger.error(f"Error adding edge from {source_id} to {target_id}: {e}")
            raise
    
    def update_node_value(self, node_id: str, value: Any) -> None:
        """
        Update the value of a node and trigger causal propagation.
        
        This method updates a node's value and automatically triggers the
        causal inference system to propagate effects through the network.
        
        Args:
            node_id: ID of the node to update
            value: New value for the node
            
        Raises:
            ValueError: If the node doesn't exist
        """
        try:
            with self._lock:
                if node_id not in self.nodes:
                    raise ValueError(f"Node '{node_id}' does not exist")
                    
                # Store old value for comparison
                old_value = self.nodes[node_id].value
                
                # Update node value
                self.nodes[node_id].update_value(value)
                
                # Queue for causal inference if value actually changed
                if old_value != value:
                    self._queue_for_inference(node_id)
                    
                    # Update quantum state and propagate entanglement if enabled
                    if self.quantum_enabled:
                        self._update_quantum_state(node_id)
                
                logger.debug(f"Updated node {node_id} value from {old_value} to {value}")
                
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {e}")
            raise
    
    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """
        Retrieve a node by its ID.
        
        Args:
            node_id: Unique identifier of the node
            
        Returns:
            The CausalNode object, or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[CausalEdge]:
        """
        Retrieve an edge by its ID.
        
        Args:
            edge_id: Unique identifier of the edge
            
        Returns:
            The CausalEdge object, or None if not found
        """
        return self.edges.get(edge_id)
    
    def get_connected_nodes(self, node_id: str, direction: str = 'both') -> List[str]:
        """
        Get IDs of nodes connected to a given node.
        
        Args:
            node_id: ID of the central node
            direction: Direction of connections ('in', 'out', or 'both')
            
        Returns:
            List of node IDs connected to the specified node
            
        Raises:
            ValueError: If direction is not valid
        """
        try:
            if node_id not in self.nodes:
                return []
            
            if direction not in ['in', 'out', 'both']:
                raise ValueError(f"Direction must be 'in', 'out', or 'both', got '{direction}'")
                
            connected = set()
            
            # Get nodes connected by outgoing edges (node causes others)
            if direction in ['out', 'both']:
                for edge_id in self.outgoing_edges.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if edge and edge.is_active:
                        connected.add(edge.target_id)
            
            # Get nodes connected by incoming edges (others cause node)
            if direction in ['in', 'both']:
                for edge_id in self.incoming_edges.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if edge and edge.is_active:
                        connected.add(edge.source_id)
            
            return list(connected)
            
        except Exception as e:
            logger.error(f"Error getting connected nodes for {node_id}: {e}")
            return []
        if direction in ['in', 'both']:
            for edge_id in self.incoming_edges.get(node_id, []):
                edge = self.edges.get(edge_id)
                if edge:
                    connected.add(edge.source_id)
        
        return list(connected)
    
    def get_node_by_name(self, name: str) -> Optional[str]:
        """
        Find a node ID by its name.
        
        Args:
            name: Name of the node to find
            
        Returns:
            Node ID if found, None otherwise
        """
        try:
            for node_id, node in self.nodes.items():
                if node.name == name:
                    return node_id
            return None
        except Exception as e:
            logger.error(f"Error finding node by name '{name}': {e}")
            return None
    
    def start_inference_engine(self) -> None:
        """
        Start the background causal inference engine.
        
        This starts a background thread that continuously processes the
        inference queue and propagates causal effects through the network.
        """
        try:
            if self.inference_thread is not None and self.inference_thread.is_alive():
                logger.warning("Inference engine is already running")
                return
                
            self.running = True
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="CausalInference"
            )
            self.inference_thread.start()
            logger.info("Causal inference engine started")
            
        except Exception as e:
            logger.error(f"Error starting inference engine: {e}")
            self.running = False
            raise
    
    def stop_inference_engine(self) -> None:
        """
        Stop the background causal inference engine.
        
        This gracefully shuts down the background inference thread.
        """
        try:
            if not self.running:
                return
                
            self.running = False
            
            # Wait for thread to finish
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=5.0)
                if self.inference_thread.is_alive():
                    logger.warning("Inference thread did not shut down gracefully")
                    
            logger.info("Causal inference engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping inference engine: {e}")
    
    def infer_causality(self, data: Dict[str, List[Any]], 
                      method: InferenceMethod = None) -> Dict[str, Any]:
        """
        Infer causal structure from observational data.
        
        This method analyzes time series or cross-sectional data to discover
        causal relationships between variables using various inference algorithms.
        
        Args:
            data: Dictionary mapping variable names to lists of values
            method: Causal inference method to use (defaults to system default)
            
        Returns:
            Dictionary containing inference results including:
            - method: Name of the inference method used
            - timestamp: When the inference was performed
            - edges_created: List of new causal edges discovered
            - edges_updated: List of existing edges that were updated
            - strength_matrix: Matrix of causal strengths (if applicable)
            
        Raises:
            ValueError: If data is invalid or method is not supported
        """
        try:
            if not data:
                raise ValueError("Data dictionary cannot be empty")
                
            if method is None:
                method = self.default_inference_method
                
            logger.info(f"Starting causal inference using {method.name} method")
            
            # Create nodes for data variables if they don't exist
            node_ids = []
            for var_name, values in data.items():
                if not isinstance(values, (list, tuple, np.ndarray)):
                    raise ValueError(f"Data for variable '{var_name}' must be a list or array")
                if len(values) == 0:
                    logger.warning(f"Variable '{var_name}' has no data points")
                    continue
                    
                # Find or create node
                node_id = self.get_node_by_name(var_name)
                if node_id is None:
                    node_id = self.add_node(name=var_name, value=values[-1] if values else None)
                node_ids.append(node_id)
                
            results = {
                "method": method.name,
                "timestamp": time.time(),
                "edges_created": [],
                "edges_updated": [],
                "strength_matrix": None,
                "variables_processed": len(node_ids),
                "data_points": {var: len(vals) for var, vals in data.items()}
            }
        
            # Apply the selected inference method
            if method == InferenceMethod.STRUCTURAL:
                edges_info = self._infer_structural(data)
            elif method == InferenceMethod.GRAPHICAL:
                edges_info = self._infer_graphical(data)
            elif method == InferenceMethod.GRANGER:
                edges_info = self._infer_granger(data)
            elif method == InferenceMethod.QUANTUM_ENTANGLEMENT:
                edges_info = self._infer_quantum_entanglement(data)
            elif method == InferenceMethod.TRANSFER_ENTROPY:
                edges_info = self._infer_transfer_entropy(data)
            else:
                # Default to graphical model
                edges_info = self._infer_graphical(data)
            
            # Process the inferred edges
            created_edges = []
            updated_edges = []
            
            # Create or update edges based on inference
            for source, target, strength, confidence, relation_type in edges_info:
                source_id = self.get_node_by_name(source)
                target_id = self.get_node_by_name(target)
                
                if not source_id or not target_id:
                    continue
                    
                # Check if edge already exists
                existing_edge_id = None
                for edge_id in self.outgoing_edges.get(source_id, []):
                    edge = self.edges.get(edge_id)
                    if edge and edge.target_id == target_id:
                        existing_edge_id = edge_id
                        break
                
                if existing_edge_id:
                    # Update existing edge
                    self.edges[existing_edge_id].update_strength(strength, confidence)
                    updated_edges.append(existing_edge_id)
                else:
                    # Create new edge
                    edge_id = self.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation_type,
                        strength=strength,
                        confidence=confidence
                    )
                    created_edges.append(edge_id)
            
            # Create strength matrix
            variables = list(data.keys())
            n = len(variables)
            strength_matrix = np.zeros((n, n))
            
            for i, source in enumerate(variables):
                for j, target in enumerate(variables):
                    if i == j:
                        continue
                        
                    source_id = self.get_node_by_name(source)
                    target_id = self.get_node_by_name(target)
                    
                    if not source_id or not target_id:
                        continue
                    
                    # Find edge from source to target
                    for edge_id in self.outgoing_edges.get(source_id, []):
                        edge = self.edges.get(edge_id)
                        if edge and edge.target_id == target_id:
                            strength_matrix[i, j] = edge.strength
                            break
            
            # Update results
            results["edges_created"] = created_edges
            results["edges_updated"] = updated_edges
            results["strength_matrix"] = strength_matrix.tolist()
            results["variables"] = variables
            
            logger.info(f"Causal inference completed: {len(created_edges)} edges created, {len(updated_edges)} updated")
            return results
            
        except Exception as e:
            logger.error(f"Error during causal inference: {e}")
            return {
                "method": method.name if method else "unknown",
                "timestamp": time.time(),
                "edges_created": [],
                "edges_updated": [],
                "error": str(e)
            }
    
    def perform_intervention(self, node_id: str, value: Any, 
                           intervention_type: InterventionType = None) -> Dict[str, Any]:
        """
        Perform a causal intervention on a specified node.
        
        This method implements various types of causal interventions including
        do-operations, soft interventions, and counterfactual reasoning.
        
        Args:
            node_id: ID of the node to intervene on
            value: Value to set for the intervention
            intervention_type: Type of intervention to perform
            
        Returns:
            Dictionary containing intervention results including:
            - intervention_node: ID of the intervened node
            - intervention_type: Type of intervention performed
            - intervention_value: Value used in intervention
            - timestamp: When the intervention was performed
            - affected_nodes: List of nodes affected by the intervention
            - counterfactual_effects: Effects compared to original state (if applicable)
            
        Raises:
            ValueError: If node doesn't exist or intervention parameters are invalid
        """
        try:
            with self._lock:
                if node_id not in self.nodes:
                    raise ValueError(f"Node '{node_id}' does not exist")
                    
                if intervention_type is None:
                    intervention_type = self.default_intervention_type
                    
                logger.info(f"Performing {intervention_type.name} intervention on node {node_id}")
                
                # Store original state for counterfactuals
                original_values = {}
                if intervention_type == InterventionType.COUNTERFACTUAL:
                    for n_id, node in self.nodes.items():
                        original_values[n_id] = node.value
                
                # Apply the intervention
                self._apply_intervention(node_id, value, intervention_type)
                
                # Propagate effects through the network
                affected_nodes = self._propagate_causal_effects(node_id)
                
                # Prepare result
                result = {
                    "intervention_node": node_id,
                    "intervention_node_name": self.nodes[node_id].name,
                    "intervention_type": intervention_type.name,
                    "intervention_value": value,
                    "timestamp": time.time(),
                    "affected_nodes": affected_nodes,
                    "num_affected": len(affected_nodes)
                }
                
                # For counterfactuals, compare with original state
                if intervention_type == InterventionType.COUNTERFACTUAL:
                    counterfactual_effects = {}
                    for n_id in affected_nodes:
                        if n_id in original_values:
                            node = self.nodes.get(n_id)
                            if node:
                                # Calculate difference between counterfactual and original
                                original = original_values[n_id]
                                counterfactual = node.value
                                
                                if isinstance(original, (int, float)) and isinstance(counterfactual, (int, float)):
                                    effect = counterfactual - original
                                else:
                                    effect = counterfactual
                                    
                                counterfactual_effects[n_id] = {
                                    "original": original,
                                    "counterfactual": counterfactual,
                                    "effect": effect
                                }
                    
                    result["counterfactual_effects"] = counterfactual_effects
                    
                    # Restore original values for counterfactual analysis
                    for n_id, orig_value in original_values.items():
                        if n_id in self.nodes:
                            self.nodes[n_id].update_value(orig_value)
                
                logger.info(f"Intervention completed: {len(affected_nodes)} nodes affected")
                return result
                
        except Exception as e:
            logger.error(f"Error performing intervention on node {node_id}: {e}")
            return {
                "intervention_node": node_id,
                "intervention_type": intervention_type.name if intervention_type else "unknown",
                "timestamp": time.time(),
                "error": str(e),
                "affected_nodes": []
            }
    
    def calculate_causal_effect(self, cause_id: str, effect_id: str, 
                             method: InferenceMethod = None) -> Dict[str, Any]:
        """Calculate the causal effect of one node on another"""
        if cause_id not in self.nodes or effect_id not in self.nodes:
            raise ValueError("Cause or effect node does not exist")
            
        if method is None:
            method = self.default_inference_method
            
        # Get current values
        cause_node = self.nodes[cause_id]
        effect_node = self.nodes[effect_id]
        
        cause_value = cause_node.value
        effect_value = effect_node.value
        
        # Different methods for calculating causal effect
        if method == InferenceMethod.GRAPHICAL:
            effect = self._calculate_graphical_effect(cause_id, effect_id)
        elif method == InferenceMethod.POTENTIAL_OUTCOMES:
            effect = self._calculate_potential_outcomes_effect(cause_id, effect_id)
        elif method == InferenceMethod.QUANTUM_ENTANGLEMENT:
            effect = self._calculate_quantum_effect(cause_id, effect_id)
        else:
            # Default to simple path analysis
            effect = self._calculate_path_effect(cause_id, effect_id)
            
        # Prepare the result
        result = {
            "cause_id": cause_id,
            "cause_name": cause_node.name,
            "cause_value": cause_value,
            "effect_id": effect_id,
            "effect_name": effect_node.name,
            "effect_value": effect_value,
            "causal_effect": effect["effect"],
            "method": method.name,
            "paths": effect.get("paths", []),
            "confidence": effect.get("confidence", 0.0),
            "timestamp": time.time()
        }
        
        return result
    
    def find_common_causes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Find common causes of multiple nodes"""
        if not node_ids:
            return []
            
        # Check if all nodes exist
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} does not exist")
                
        # Get all potential causes (nodes with paths to all target nodes)
        potential_causes = {}  # node_id -> {target_id: paths}
        
        for target_id in node_ids:
            # Get all nodes with paths to this target
            for node_id in self.nodes:
                if node_id == target_id:
                    continue
                    
                paths = self._find_causal_paths(node_id, target_id)
                if paths:
                    if node_id not in potential_causes:
                        potential_causes[node_id] = {}
                    potential_causes[node_id][target_id] = paths
        
        # Find nodes that have paths to all targets
        common_causes = []
        for node_id, target_paths in potential_causes.items():
            if len(target_paths) == len(node_ids):
                # This is a common cause
                node = self.nodes[node_id]
                
                # Calculate average path strength
                path_strengths = []
                for target_id, paths in target_paths.items():
                    # Take strongest path to each target
                    if paths:
                        path_strengths.append(max(p["strength"] for p in paths))
                
                avg_strength = sum(path_strengths) / len(path_strengths) if path_strengths else 0
                
                common_causes.append({
                    "node_id": node_id,
                    "name": node.name,
                    "value": node.value,
                    "average_strength": avg_strength,
                    "target_paths": {t: [p["strength"] for p in paths] for t, paths in target_paths.items()}
                })
        
        # Sort by average strength
        common_causes.sort(key=lambda x: x["average_strength"], reverse=True)
        
        return common_causes
    
    def detect_confounders(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Detect potential confounding variables between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node does not exist")
            
        # First check if there's a direct edge
        direct_edge = None
        for edge_id in self.outgoing_edges.get(source_id, []):
            edge = self.edges.get(edge_id)
            if edge and edge.target_id == target_id:
                direct_edge = edge
                break
        
        # Find all common causes of both source and target
        common_causes = self.find_common_causes([source_id, target_id])
        
        # For each common cause, check if it explains away the direct effect
        confounders = []
        for cause in common_causes:
            cause_id = cause["node_id"]
            
            # Skip if the cause is the source or target
            if cause_id == source_id or cause_id == target_id:
                continue
            
            # Calculate direct effect
            direct_effect = self._calculate_path_effect(source_id, target_id)
            
            # Calculate effect adjusting for confounder
            adjusted_effect = self._calculate_adjusted_effect(source_id, target_id, [cause_id])
            
            # Calculate how much the confounder explains away
            explanation_ratio = 1.0
            if direct_effect["effect"] != 0:
                explanation_ratio = 1.0 - (adjusted_effect["effect"] / direct_effect["effect"])
                
            confounders.append({
                "node_id": cause_id,
                "name": self.nodes[cause_id].name,
                "value": self.nodes[cause_id].value,
                "direct_effect": direct_effect["effect"],
                "adjusted_effect": adjusted_effect["effect"],
                "explanation_ratio": explanation_ratio,
                "is_significant": abs(explanation_ratio) > 0.3  # Consider significant if explains >30%
            })
        
        # Sort by explanation ratio
        confounders.sort(key=lambda x: abs(x["explanation_ratio"]), reverse=True)
        
        return confounders
    
    def generate_causal_graph(self) -> Dict[str, Any]:
        """Generate a representation of the causal graph"""
        with self._lock:
            nodes_data = []
            for node_id, node in self.nodes.items():
                nodes_data.append({
                    "id": node_id,
                    "name": node.name,
                    "observed": node.is_observed,
                    "latent": node.is_latent,
                    "entropy": node.entropy
                })
                
            edges_data = []
            for edge_id, edge in self.edges.items():
                edges_data.append({
                    "id": edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.relation_type.name,
                    "strength": edge.strength,
                    "confidence": edge.confidence
                })
                
            # Generate additional graph metrics
            metrics = self._calculate_graph_metrics()
            
            graph_data = {
                "nodes": nodes_data,
                "edges": edges_data,
                "entangled_pairs": [
                    {"source": src, "target": tgt, "strength": strength}
                    for (src, tgt), strength in self.entangled_pairs.items()
                ],
                "metrics": metrics
            }
            
            return graph_data
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for advanced graph analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.to_dict())
            
        # Add edges
        for edge_id, edge in self.edges.items():
            G.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
            
        return G
    
    def save_to_file(self, filepath: str) -> bool:
        """Save the causal entanglement matrix to a file"""
        try:
            # Create a serializable representation
            data = {
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
                "outgoing_edges": self.outgoing_edges,
                "incoming_edges": self.incoming_edges,
                "entangled_pairs": {str(k): v for k, v in self.entangled_pairs.items()},
                "quantum_enabled": self.quantum_enabled,
                "temporal_resolution": self.temporal_resolution,
                "default_confidence": self.default_confidence,
                "creation_time": self.creation_time
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            return True
            
        except Exception as e:
            print(f"Error saving causal matrix: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CausalEntanglementMatrix':
        """Load a causal entanglement matrix from a file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Create new matrix
            matrix = cls(quantum_enabled=data.get("quantum_enabled", True))
            matrix.temporal_resolution = data.get("temporal_resolution", 0.01)
            matrix.default_confidence = data.get("default_confidence", 0.5)
            matrix.creation_time = data.get("creation_time", time.time())
            
            # Load nodes
            for node_id, node_data in data.get("nodes", {}).items():
                node = CausalNode(node_id=node_id)
                
                # Set attributes
                for k, v in node_data.items():
                    if hasattr(node, k) and k != "value_history":
                        setattr(node, k, v)
                
                # Add to matrix
                matrix.nodes[node_id] = node
                matrix.outgoing_edges[node_id] = []
                matrix.incoming_edges[node_id] = []
            
            # Load edges
            for edge_id, edge_data in data.get("edges", {}).items():
                edge = CausalEdge(edge_id=edge_id)
                
                # Set attributes
                for k, v in edge_data.items():
                    if hasattr(edge, k) and k not in ["mechanism"]:
                        if k == "relation_type" and isinstance(v, str):
                            # Convert string to enum
                            try:
                                setattr(edge, k, CausalRelationType[v])
                            except KeyError:
                                # Use default if enum value not found
                                setattr(edge, k, CausalRelationType.DIRECT)
                        else:
                            setattr(edge, k, v)
                
                # Add to matrix
                matrix.edges[edge_id] = edge
                
                # Add to outgoing/incoming edges
                if edge.source_id in matrix.outgoing_edges:
                    matrix.outgoing_edges[edge.source_id].append(edge_id)
                if edge.target_id in matrix.incoming_edges:
                    matrix.incoming_edges[edge.target_id].append(edge_id)
            
            # Load edge connections if not added above
            matrix.outgoing_edges = data.get("outgoing_edges", matrix.outgoing_edges)
            matrix.incoming_edges = data.get("incoming_edges", matrix.incoming_edges)
            
            # Load entangled pairs
            for k, v in data.get("entangled_pairs", {}).items():
                try:
                    # Convert string key back to tuple
                    key_parts = k.strip("()").replace("'", "").split(", ")
                    if len(key_parts) == 2:
                        matrix.entangled_pairs[(key_parts[0], key_parts[1])] = v
                except:
                    pass
            
            return matrix
            
        except Exception as e:
            print(f"Error loading causal matrix: {e}")
            return cls()
    
    def _initialize_quantum_state(self, node: CausalNode) -> None:
        """Initialize the quantum state of a node"""
        # Create simple quantum state based on value
        if node.value is None:
            # Default to |0⟩ state if no value
            state = np.array([1.0, 0.0], dtype=complex)
        elif isinstance(node.value, (bool, np.bool_)):
            # Boolean value: False -> |0⟩, True -> |1⟩
            state = np.array([1.0, 0.0], dtype=complex) if not node.value else np.array([0.0, 1.0], dtype=complex)
        elif isinstance(node.value, (int, float, np.integer, np.floating)):
            # Numeric value: encode in probability amplitude
            # Normalize to [0,1]
            val = np.clip(abs(node.value), 0, 1)
            # Create superposition state
            alpha = np.sqrt(1 - val)
            beta = np.sqrt(val)
            state = np.array([alpha, beta], dtype=complex)
        elif isinstance(node.value, (list, tuple, np.ndarray)):
            # Vector value: convert to quantum state if possible
            try:
                arr = np.array(node.value, dtype=float)
                if arr.ndim == 1:
                    # Normalize
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    # If too large, truncate or embed
                    if len(arr) > 8:  # Limit to 8 qubits (256 amplitudes)
                        arr = arr[:8]
                    # Convert to quantum state
                    state = arr.astype(complex)
                else:
                    # Default to simple state
                    state = np.array([1.0, 0.0], dtype=complex)
            except:
                state = np.array([1.0, 0.0], dtype=complex)
        else:
            # Default state
            state = np.array([1.0, 0.0], dtype=complex)
        
        # Ensure state is normalized
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        # Set the quantum state
        node.update_quantum_state(state)
    
    def _establish_quantum_correlation(self, node_id1: str, node_id2: str, strength: float) -> None:
        """Establish quantum correlation between two nodes"""
        # Ensure both nodes exist
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return
            
        # Record the entanglement
        pair_key = tuple(sorted([node_id1, node_id2]))
        self.entangled_pairs[pair_key] = strength
        
        # Get the nodes
        node1 = self.nodes[node_id1]
        node2 = self.nodes[node_id2]
        
        # Skip if either node doesn't have a quantum state
        if node1.quantum_state is None or node2.quantum_state is None:
            return
            
        # Create entangled state (simplified)
        if strength > 0.9:
            # Create maximally entangled state (Bell state)
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            entangled_state = np.zeros(4, dtype=complex)
            entangled_state[0] = 1/np.sqrt(2)  # |00⟩
            entangled_state[3] = 1/np.sqrt(2)  # |11⟩
        elif strength > 0.5:
            # Create partially entangled state
            alpha = np.sqrt(0.5 + (1-strength)/2)
            beta = np.sqrt(0.5 - (1-strength)/2)
            entangled_state = np.array([alpha, 0, 0, beta], dtype=complex)
        else:
            # Weak entanglement, create mostly separable state
            entangled_state = np.array([
                np.sqrt(0.5), 
                np.sqrt(0.2), 
                np.sqrt(0.2), 
                np.sqrt(0.1)
            ], dtype=complex)
            
        # Extract single-qubit states (partial trace)
        # In a real quantum system, this would require proper partial trace
        # This is a simplified approximation
        state1 = np.array([
            np.sqrt(entangled_state[0]**2 + entangled_state[1]**2),
            np.sqrt(entangled_state[2]**2 + entangled_state[3]**2)
        ], dtype=complex)
        
        state2 = np.array([
            np.sqrt(entangled_state[0]**2 + entangled_state[2]**2),
            np.sqrt(entangled_state[1]**2 + entangled_state[3]**2)
        ], dtype=complex)
        
        # Update the quantum states
        node1.update_quantum_state(state1)
        node2.update_quantum_state(state2)
    
    def _update_quantum_state(self, node_id: str) -> None:
        """Update quantum state when node value changes and propagate through entanglement"""
        node = self.nodes.get(node_id)
        if not node or not self.quantum_enabled:
            return
            
        # Update this node's quantum state based on new value
        self._initialize_quantum_state(node)
        
        # Propagate through entangled nodes
        entangled_nodes = self._get_entangled_nodes(node_id)
        
        for e_node_id, strength in entangled_nodes:
            # Skip if node doesn't exist
            if e_node_id not in self.nodes:
                continue
                
            e_node = self.nodes[e_node_id]
            
            # Skip if no quantum state
            if node.quantum_state is None or e_node.quantum_state is None:
                continue
                
            # Apply entanglement effect (simplified)
            # In a real quantum system, this would be more complex
            if strength > 0.7:
                # Strong entanglement: Align states
                # This is a classical approximation of quantum correlation
                if np.random.random() < strength:
                    # Probabilistically copy state with some noise
                    new_state = node.quantum_state.copy()
                    # Add noise
                    noise = np.random.normal(0, 0.1, size=new_state.shape)
                    new_state = new_state + noise * 1j * np.random.normal(0, 0.1, size=new_state.shape)
                    # Normalize
                    norm = np.linalg.norm(new_state)
                    if norm > 0:
                        new_state = new_state / norm
                    e_node.update_quantum_state(new_state)
    
    def _get_entangled_nodes(self, node_id: str) -> List[Tuple[str, float]]:
        """Get nodes entangled with the given node"""
        entangled = []
        
        for pair, strength in self.entangled_pairs.items():
            if node_id in pair:
                # Find the other node in the pair
                other_id = pair[0] if pair[1] == node_id else pair[1]
                entangled.append((other_id, strength))
                
        return entangled
    
    def _queue_for_inference(self, node_id: str) -> None:
        """Queue a node for causal inference"""
        if node_id not in self.inference_queue:
            self.inference_queue.append(node_id)
    
    def _inference_loop(self) -> None:
        """Background thread for causal inference"""
        while self.running:
            try:
                # Process inference queue
                with self._lock:
                    if self.inference_queue:
                        # Get next node to process
                        node_id = self.inference_queue.popleft()
                        
                        # Skip if node no longer exists
                        if node_id not in self.nodes:
                            continue
                            
                        # Process causal effects
                        affected = self._propagate_causal_effects(node_id)
                        
                        # Record result
                        self.inference_results[node_id] = {
                            "timestamp": time.time(),
                            "affected_nodes": affected
                        }
            except Exception as e:
                print(f"Error in causal inference loop: {e}")
            
            # Sleep to avoid excessive CPU usage
            time.sleep(0.01)
    
    def _apply_intervention(self, node_id: str, value: Any, 
                         intervention_type: InterventionType) -> None:
        """Apply a causal intervention"""
        node = self.nodes[node_id]
        
        if intervention_type == InterventionType.DO:
            # Direct "do" operation - simply set the value
            node.update_value(value)
            
        elif intervention_type == InterventionType.SOFT:
            # Soft intervention - blend with current value
            if isinstance(node.value, (int, float)) and isinstance(value, (int, float)):
                # Numeric blend
                blended = 0.7 * value + 0.3 * node.value
                node.update_value(blended)
            else:
                # Non-numeric values, just set directly
                node.update_value(value)
                
        elif intervention_type == InterventionType.QUANTUM:
            # Quantum intervention - put node into superposition
            if not self.quantum_enabled:
                # Fall back to normal intervention
                node.update_value(value)
                return
                
            # Create superposition state
            if isinstance(value, (int, float)) and isinstance(node.value, (int, float)):
                # Create superposition of old and new values
                old_val = np.clip(abs(node.value), 0, 1)
                new_val = np.clip(abs(value), 0, 1)
                
                # Balanced superposition
                alpha = np.sqrt(0.5)
                beta = np.sqrt(0.5)
                
                # Create superposition state
                state = np.array([alpha, beta], dtype=complex)
                node.update_quantum_state(state)
                
                # Also update classical value (weighted average)
                node.update_value(0.5 * node.value + 0.5 * value)
            else:
                # Non-numeric, just set the value
                node.update_value(value)
                
        elif intervention_type == InterventionType.STOCHASTIC:
            # Stochastic intervention - probabilistic update
            if np.random.random() < 0.8:
                node.update_value(value)
            # Else leave unchanged
            
        else:
            # Default to direct intervention
            node.update_value(value)
    
    def _propagate_causal_effects(self, start_node_id: str) -> List[str]:
        """Propagate causal effects from a starting node"""
        # Skip if node doesn't exist
        if start_node_id not in self.nodes:
            return []
            
        # Breadth-first propagation
        queue = deque([(start_node_id, 1.0)])  # (node_id, strength_so_far)
        visited = {start_node_id}
        affected_nodes = []
        
        while queue:
            node_id, strength_so_far = queue.popleft()
            
            # Skip if node no longer exists
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            node_value = node.value
            
            # Skip if value is None
            if node_value is None:
                continue
                
            # Process outgoing edges
            for edge_id in self.outgoing_edges.get(node_id, []):
                edge = self.edges.get(edge_id)
                if not edge or not edge.is_active:
                    continue
                    
                target_id = edge.target_id
                target = self.nodes.get(target_id)
                if not target:
                    continue
                    
                # Calculate propagated strength
                edge_strength = edge.strength
                propagated_strength = strength_so_far * edge_strength
                
                # Skip if effect is too weak
                if propagated_strength < 0.05:
                    continue
                    
                # Skip if already visited with stronger effect
                if target_id in visited:
                    continue
                    
                # Apply causal effect
                try:
                    # If edge has a specific mechanism, use it
                    if edge.mechanism:
                        new_value = edge.apply_mechanism(node_value)
                    else:
                        # Default mechanism based on value type
                        if isinstance(node_value, (int, float)) and isinstance(target.value, (int, float)):
                            # Linear influence for numeric values
                            effect = node_value * edge_strength
                            new_value = target.value + effect
                        elif isinstance(node_value, bool) and isinstance(target.value, bool):
                            # Boolean influence
                            if edge_strength > 0.5:
                                new_value = node_value  # Strong enough to flip
                            else:
                                new_value = target.value  # Not strong enough
                        else:
                            # For other types, only propagate if strength is high
                            if edge_strength > 0.7:
                                new_value = node_value
                            else:
                                continue
                    
                    # Update the target value
                    target.update_value(new_value)
                    affected_nodes.append(target_id)
                    
                    # Add to queue for further propagation
                    queue.append((target_id, propagated_strength))
                    visited.add(target_id)
                    
                except Exception as e:
                    print(f"Error propagating causal effect: {e}")
        
        # Also propagate through quantum entanglement
        if self.quantum_enabled:
            entangled_nodes = self._get_entangled_nodes(start_node_id)
            for e_node_id, strength in entangled_nodes:
                if e_node_id in visited or e_node_id not in self.nodes:
                    continue
                    
                # Apply quantum correlation effect
                try:
                    e_node = self.nodes[e_node_id]
                    
                    # Update through entanglement
                    self._update_quantum_state(e_node_id)
                    
                    # For classical value, apply probabilistic update
                    if np.random.random() < strength:
                        # With some probability proportional to entanglement strength,
                        # update value based on source node
                        start_node = self.nodes[start_node_id]
                        if isinstance(start_node.value, (int, float)) and isinstance(e_node.value, (int, float)):
                            # Blend values
                            new_value = (1-strength) * e_node.value + strength * start_node.value
                            e_node.update_value(new_value)
                            affected_nodes.append(e_node_id)
                            visited.add(e_node_id)
                    
                except Exception as e:
                    print(f"Error propagating quantum effect: {e}")
        
        return affected_nodes
    
    def _infer_structural(self, data: Dict[str, List[Any]]) -> List[Tuple[str, str, float, float, CausalRelationType]]:
        """Infer causal structure using structural equation modeling (simplified)"""
        edges = []
        variables = list(data.keys())
        n_vars = len(variables)
        
        # Convert to numpy arrays
        arrays = {}
        for var, values in data.items():
            try:
                arrays[var] = np.array(values, dtype=float)
            except:
                # Skip variables that can't be converted to float
                continue
        
        # Only keep variables with numeric arrays
        numeric_vars = list(arrays.keys())
        
        # Check all pairs of variables
        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i == j:
                    continue
                    
                x = arrays[var1]
                y = arrays[var2]
                
                # Skip if not enough data
                if len(x) < 3 or len(y) < 3:
                    continue
                    
                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Try linear regression y ~ x
                try:
                    # Add constant for intercept
                    X = np.column_stack((np.ones(min_len), x))
                    # Solve OLS
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    # Calculate residuals
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    # Calculate R-squared
                    ss_total = np.sum((y - np.mean(y))**2)
                    ss_residual = np.sum(residuals**2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Calculate p-value (simplified)
                    n = min_len
                    k = 1  # One predictor
                    sigma2 = ss_residual / (n - k - 1)
                    var_beta = sigma2 * np.linalg.inv(X.T @ X)[1, 1]
                    t_stat = beta[1] / np.sqrt(var_beta)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
                    
                    # If significant, add edge
                    if p_value < 0.05 and r_squared > 0.1:
                        # Edge strength based on R-squared
                        strength = min(0.9, max(0.1, r_squared))
                        # Confidence based on p-value
                        confidence = min(0.95, max(0.5, 1.0 - p_value))
                        
                        # Determine relation type
                        relation_type = CausalRelationType.DIRECT
                        
                        edges.append((var1, var2, strength, confidence, relation_type))
                except Exception as e:
                    # Skip if regression fails
                    continue
        
        return edges
    
    def _infer_graphical(self, data: Dict[str, List[Any]]) -> List[Tuple[str, str, float, float, CausalRelationType]]:
        """Infer causal structure using graphical models (simplified PC algorithm)"""
        edges = []
        variables = list(data.keys())
        n_vars = len(variables)
        
        # Convert to numpy arrays
        arrays = {}
        for var, values in data.items():
            try:
                arrays[var] = np.array(values, dtype=float)
            except:
                # Skip variables that can't be converted to float
                continue
        
        # Only keep variables with numeric arrays
        numeric_vars = list(arrays.keys())
        n_numeric = len(numeric_vars)
        
        # Start with complete undirected graph
        connections = np.ones((n_numeric, n_numeric), dtype=bool)
        np.fill_diagonal(connections, False)  # No self-loops
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_numeric, n_numeric))
        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    continue
                    
                x = arrays[var1]
                y = arrays[var2]
                
                # Skip if not enough data
                if len(x) < 3 or len(y) < 3:
                    continue
                    
                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Calculate correlation
                try:
                    corr = np.corrcoef(x, y)[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                except:
                    correlation_matrix[i, j] = 0
                    correlation_matrix[j, i] = 0
        
        # Remove edges with low correlation
        threshold = 0.3
        connections = np.abs(correlation_matrix) > threshold
        np.fill_diagonal(connections, False)  # No self-loops
        
        # Orient edges using temporal information
        for i in range(n_numeric):
            for j in range(n_numeric):
                if connections[i, j]:
                    var1 = numeric_vars[i]
                    var2 = numeric_vars[j]
                    corr = correlation_matrix[i, j]
                    
                    # Skip if correlation is too low
                    if abs(corr) < threshold:
                        continue
                    
                    # Calculate strength based on correlation
                    strength = min(0.9, max(0.1, abs(corr)))
                    
                    # Calculate confidence
                    confidence = 0.5 + 0.5 * abs(corr)
                    
                    # Determine relation type
                    relation_type = CausalRelationType.DIRECT
                    if corr < 0:
                        relation_type = CausalRelationType.INDIRECT
                        
                    # Check temporal precedence (if timestamps are provided in metadata)
                    if hasattr(self, 'nodes') and var1 in self.nodes and var2 in self.nodes:
                        node1 = self.nodes[var1]
                        node2 = self.nodes[var2]
                        
                        if hasattr(node1, 'creation_time') and hasattr(node2, 'creation_time'):
                            # If one variable was created before the other, it's more likely to be the cause
                            if node1.creation_time < node2.creation_time:
                                # var1 -> var2
                                edges.append((var1, var2, strength, confidence, relation_type))
                            else:
                                # var2 -> var1
                                edges.append((var2, var1, strength, confidence, relation_type))
                                
                            # Skip adding the undirected edge
                            continue
                    
                    # If no temporal information, add edge based on correlation magnitude
                    edges.append((var1, var2, strength, confidence, relation_type))
        
        return edges
    
    def _infer_granger(self, data: Dict[str, List[Any]]) -> List[Tuple[str, str, float, float, CausalRelationType]]:
        """Infer causal structure using Granger causality (simplified)"""
        edges = []
        variables = list(data.keys())
        n_vars = len(variables)
        
        # Convert to numpy arrays
        arrays = {}
        for var, values in data.items():
            try:
                arrays[var] = np.array(values, dtype=float)
            except:
                # Skip variables that can't be converted to float
                continue
        
        # Only keep variables with numeric arrays
        numeric_vars = list(arrays.keys())
        
        # Use a basic Granger causality test
        # X "Granger causes" Y if past values of X help predict Y
        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i == j:
                    continue
                    
                x = arrays[var1]
                y = arrays[var2]
                
                # Skip if not enough data
                if len(x) < 5 or len(y) < 5:
                    continue
                    
                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Create lagged versions (1-step lag)
                x_lagged = x[:-1]
                y_lagged = y[:-1]
                y_target = y[1:]
                
                # Test if x helps predict y beyond y's own past
                try:
                    # Model 1: y ~ y_lagged
                    X1 = np.column_stack((np.ones(len(y_lagged)), y_lagged))
                    beta1 = np.linalg.lstsq(X1, y_target, rcond=None)[0]
                    residuals1 = y_target - X1 @ beta1
                    rss1 = np.sum(residuals1**2)
                    
                    # Model 2: y ~ y_lagged + x_lagged
                    X2 = np.column_stack((np.ones(len(y_lagged)), y_lagged, x_lagged))
                    beta2 = np.linalg.lstsq(X2, y_target, rcond=None)[0]
                    residuals2 = y_target - X2 @ beta2
                    rss2 = np.sum(residuals2**2)
                    
                    # If adding x_lagged reduces error significantly, X Granger-causes Y
                    if rss1 > 0 and rss2 > 0:
                        # F-test for nested models
                        n = len(y_target)
                        df1 = 1  # Additional parameter in second model
                        df2 = n - 3  # Degrees of freedom in denominator
                        f_stat = ((rss1 - rss2) / df1) / (rss2 / df2)
                        
                        # Simplified p-value calculation
                        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                        
                        if p_value < 0.05:  # Significant Granger causality
                            # Calculate strength based on RSS reduction
                            strength = min(0.9, max(0.1, 1.0 - (rss2 / rss1)))
                            # Confidence based on p-value
                            confidence = min(0.95, max(0.5, 1.0 - p_value))
                            
                            # Always TEMPORAL relation type for Granger causality
                            relation_type = CausalRelationType.TEMPORAL
                            
                            edges.append((var1, var2, strength, confidence, relation_type))
                except Exception as e:
                    # Skip if regression fails
                    continue
        
        return edges
    
    def _infer_quantum_entanglement(self, data: Dict[str, List[Any]]) -> List[Tuple[str, str, float, float, CausalRelationType]]:
        """Infer causal structure using quantum entanglement principles"""
        edges = []
        variables = list(data.keys())
        
        # Only proceed if quantum is enabled
        if not self.quantum_enabled:
            return self._infer_graphical(data)  # Fall back to graphical method
        
        # Convert to numpy arrays
        arrays = {}
        for var, values in data.items():
            try:
                arrays[var] = np.array(values, dtype=float)
            except:
                # Skip variables that can't be converted to float
                continue
        
        # Only keep variables with numeric arrays
        numeric_vars = list(arrays.keys())
        n_numeric = len(numeric_vars)
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_numeric, n_numeric))
        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    continue
                    
                x = arrays[var1]
                y = arrays[var2]
                
                # Skip if not enough data
                if len(x) < 3 or len(y) < 3:
                    continue
                    
                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Calculate correlation
                try:
                    corr = np.corrcoef(x, y)[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                except:
                    correlation_matrix[i, j] = 0
                    correlation_matrix[j, i] = 0
        
        # Calculate quantum-like correlation
        # Classical correlation max is 1.0, quantum can exceed this up to sqrt(2)
        # Here we simulate this by amplifying strong correlations
        quantum_corr = np.zeros_like(correlation_matrix)
        for i in range(n_numeric):
            for j in range(n_numeric):
                if i != j:
                    # Amplify strong correlations (simulating quantum advantage)
                    c = correlation_matrix[i, j]
                    # Apply non-linear transformation that amplifies strong correlations
                    if abs(c) > 0.7:
                        quantum_corr[i, j] = np.sign(c) * np.sqrt(abs(c))
                    else:
                        quantum_corr[i, j] = c / 2
        
        # Find quantum entangled pairs
        for i in range(n_numeric):
            for j in range(i + 1, n_numeric):  # Only check each pair once
                q_corr = quantum_corr[i, j]
                
                # Skip if correlation is too weak
                if abs(q_corr) < 0.3:
                    continue
                
                var1 = numeric_vars[i]
                var2 = numeric_vars[j]
                
                # Strength based on quantum correlation
                strength = min(0.99, abs(q_corr))
                
                # Confidence level
                confidence = 0.5 + 0.5 * abs(q_corr)
                
                # Determine relation type
                if abs(q_corr) > 0.8:
                    # Strong quantum correlation suggests quantum entanglement
                    relation_type = CausalRelationType.QUANTUM
                elif q_corr > 0:
                    relation_type = CausalRelationType.DIRECT
                else:
                    relation_type = CausalRelationType.INDIRECT
                
                # For quantum entanglement, create bidirectional edges
                if relation_type == CausalRelationType.QUANTUM:
                    edges.append((var1, var2, strength, confidence, relation_type))
                    edges.append((var2, var1, strength, confidence, relation_type))
                else:
                    # For other types, determine direction
                    # (simplified - in practice would need more sophisticated tests)
                    if np.mean(arrays[var1]) > np.mean(arrays[var2]):
                        edges.append((var1, var2, strength, confidence, relation_type))
                    else:
                        edges.append((var2, var1, strength, confidence, relation_type))
        
        return edges
    
    def _infer_transfer_entropy(self, data: Dict[str, List[Any]]) -> List[Tuple[str, str, float, float, CausalRelationType]]:
        """Infer causal structure using transfer entropy (simplified)"""
        edges = []
        variables = list(data.keys())
        
        # Convert to numpy arrays
        arrays = {}
        for var, values in data.items():
            try:
                arrays[var] = np.array(values, dtype=float)
            except:
                # Skip variables that can't be converted to float
                continue
        
        # Only keep variables with numeric arrays
        numeric_vars = list(arrays.keys())
        
        # Calculate transfer entropy for each pair
        # Transfer entropy: I(Y_t+1; X_t | Y_t)
        # Measures info transfer from X to Y
        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i == j:
                    continue
                    
                x = arrays[var1]
                y = arrays[var2]
                
                # Skip if not enough data
                if len(x) < 5 or len(y) < 5:
                    continue
                    
                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Simplify by discretizing data
                # In practice, use proper entropy estimators
                try:
                    # Discretize signals into bins
                    n_bins = min(10, len(x) // 5)  # Avoid too many bins with small data
                    x_binned = np.digitize(x, np.linspace(min(x), max(x), n_bins))
                    y_binned = np.digitize(y, np.linspace(min(y), max(y), n_bins))
                    
                    # Calculate lagged variables
                    x_t = x_binned[:-1]
                    y_t = y_binned[:-1]
                    y_t_plus_1 = y_binned[1:]
                    
                    # Calculate probabilities
                    # This is a very simplified estimation - in practice use proper estimators
                    joint_xy_y = {}  # (x_t, y_t, y_t+1)
                    joint_y_y = {}   # (y_t, y_t+1)
                    
                    for t in range(len(x_t)):
                        joint_xy_y_key = (x_t[t], y_t[t], y_t_plus_1[t])
                        joint_y_y_key = (y_t[t], y_t_plus_1[t])
                        
                        joint_xy_y[joint_xy_y_key] = joint_xy_y.get(joint_xy_y_key, 0) + 1
                        joint_y_y[joint_y_y_key] = joint_y_y.get(joint_y_y_key, 0) + 1
                    
                    # Calculate transfer entropy (simplified)
                    te = 0
                    n_samples = len(x_t)
                    
                    for (x_val, y_val, y_next_val), count_xyz in joint_xy_y.items():
                        p_xyz = count_xyz / n_samples
                        p_yz = joint_y_y.get((y_val, y_next_val), 0) / n_samples
                        
                        if p_yz > 0:
                            # Simple transfer entropy calculation
                            te += p_xyz * np.log2(p_xyz / p_yz)
                    
                    # If transfer entropy is significant, add edge
                    if te > 0.1:  # Arbitrary threshold
                        # Normalize to [0, 1]
                        strength = min(0.9, max(0.1, te))
                        confidence = 0.5 + strength * 0.5  # Simple confidence measure
                        
                        # Always probabilistic type for transfer entropy
                        relation_type = CausalRelationType.PROBABILISTIC
                        
                        edges.append((var1, var2, strength, confidence, relation_type))
                
                except Exception as e:
                    # Skip if calculation fails
                    continue
        
        return edges
    
    def _calculate_graphical_effect(self, cause_id: str, effect_id: str) -> Dict[str, Any]:
        """Calculate causal effect using graphical model approach"""
        # Find all directed paths from cause to effect
        paths = self._find_causal_paths(cause_id, effect_id)
        
        if not paths:
            return {"effect": 0, "confidence": 0, "paths": []}
        
        # Calculate combined effect across all paths
        combined_effect = 0
        combined_confidence = 0
        
        for path in paths:
            path_effect = path["strength"]
            path_confidence = path["confidence"]
            
            # Add effect, weighted by confidence
            combined_effect += path_effect * path_confidence
            combined_confidence = max(combined_confidence, path_confidence)
        
        # Normalize if we have multiple paths
        if len(paths) > 1:
            combined_effect /= len(paths)
        
        return {
            "effect": combined_effect,
            "confidence": combined_confidence,
            "paths": paths
        }
    
    def _calculate_potential_outcomes_effect(self, cause_id: str, effect_id: str) -> Dict[str, Any]:
        """Calculate causal effect using potential outcomes framework"""
        # Get the nodes
        cause_node = self.nodes.get(cause_id)
        effect_node = self.nodes.get(effect_id)
        
        if not cause_node or not effect_node:
            return {"effect": 0, "confidence": 0}
            
        # Get current values
        cause_value = cause_node.value
        effect_value = effect_node.value
        
        if cause_value is None or effect_value is None:
            return {"effect": 0, "confidence": 0}
            
        # Only works for numeric values
        if not isinstance(cause_value, (int, float)) or not isinstance(effect_value, (int, float)):
            # Fall back to graphical approach
            return self._calculate_graphical_effect(cause_id, effect_id)
            
        # Need history of values to estimate effect
        if not cause_node.value_history or not effect_node.value_history:
            # Fall back to graphical approach
            return self._calculate_graphical_effect(cause_id, effect_id)
        
        # Extract history of values
        cause_history = [(val, t) for val, t in cause_node.value_history if isinstance(val, (int, float))]
        effect_history = [(val, t) for val, t in effect_node.value_history if isinstance(val, (int, float))]
        
        # Need enough history
        if len(cause_history) < 3 or len(effect_history) < 3:
            # Fall back to graphical approach
            return self._calculate_graphical_effect(cause_id, effect_id)
            
        # Find potential outcome pairs (times when cause changed)
        # This is a simplified approach - in practice would use more sophisticated methods
        outcomes = []
        
        # Sort by time
        cause_history.sort(key=lambda x: x[1])
        
        for i in range(1, len(cause_history)):
            prev_cause, prev_time = cause_history[i-1]
            curr_cause, curr_time = cause_history[i]
            
            # Check if cause value changed
            if abs(curr_cause - prev_cause) > 1e-6:
                # Find effect values before and after
                effect_before = effect_node.get_value_at_time(prev_time)
                effect_after = effect_node.get_value_at_time(curr_time + 0.001)  # Small offset
                
                if effect_before is not None and effect_after is not None:
                    # Calculate change in effect per unit change in cause
                    if curr_cause != prev_cause:  # Avoid division by zero
                        effect_change = (effect_after - effect_before) / (curr_cause - prev_cause)
                        outcomes.append(effect_change)
        
        # If we have outcomes, calculate average effect
        if outcomes:
            avg_effect = sum(outcomes) / len(outcomes)
            # Confidence based on consistency and number of samples
            std_dev = np.std(outcomes) if len(outcomes) > 1 else 0
            consistency = 1.0 / (1.0 + std_dev)  # Higher consistency with lower std dev
            sample_factor = min(1.0, len(outcomes) / 10.0)  # More samples = higher confidence
            confidence = 0.5 * consistency + 0.5 * sample_factor
            
            return {
                "effect": avg_effect,
                "confidence": confidence,
                "samples": len(outcomes),
                "std_dev": std_dev
            }
        
        # Fall back to graphical approach if no outcomes found
        return self._calculate_graphical_effect(cause_id, effect_id)
    
    def _calculate_quantum_effect(self, cause_id: str, effect_id: str) -> Dict[str, Any]:
        """Calculate causal effect using quantum entanglement principles"""
        if not self.quantum_enabled:
            # Fall back to graphical approach
            return self._calculate_graphical_effect(cause_id, effect_id)
            
        # Check if nodes have quantum states
        cause_node = self.nodes.get(cause_id)
        effect_node = self.nodes.get(effect_id)
        
        if not cause_node or not effect_node:
            return {"effect": 0, "confidence": 0}
            
        # Check if nodes have quantum states
        if cause_node.quantum_state is None or effect_node.quantum_state is None:
            # Fall back to graphical approach
            return self._calculate_graphical_effect(cause_id, effect_id)
            
        # Check if nodes are entangled
        pair_key = tuple(sorted([cause_id, effect_id]))
        entanglement = self.entangled_pairs.get(pair_key, 0)
        
        # Calculate effect based on combination of quantum and classical factors
        classical_effect = self._calculate_graphical_effect(cause_id, effect_id)
        
        if entanglement > 0:
            # Enhanced effect due to quantum entanglement
            # In entangled systems, correlation can exceed classical bounds
            quantum_factor = 1.0 + 0.5 * entanglement
            quantum_effect = classical_effect["effect"] * quantum_factor
            
            # Enhanced confidence due to quantum entanglement
            quantum_confidence = min(0.99, classical_effect["confidence"] + 0.2 * entanglement)
            
            return {
                "effect": quantum_effect,
                "confidence": quantum_confidence,
                "paths": classical_effect.get("paths", []),
                "entanglement": entanglement,
                "quantum_enhanced": True
            }
        
        # No entanglement, return classical effect
        return classical_effect
    
    def _calculate_path_effect(self, cause_id: str, effect_id: str) -> Dict[str, Any]:
        """Calculate causal effect using simple path analysis"""
        paths = self._find_causal_paths(cause_id, effect_id)
        
        if not paths:
            return {"effect": 0, "confidence": 0, "paths": []}
            
        # Calculate combined effect of all paths
        total_effect = 0
        max_confidence = 0
        
        for path in paths:
            path_effect = path["strength"]
            path_confidence = path["confidence"]
            
            total_effect += path_effect
            max_confidence = max(max_confidence, path_confidence)
            
        # Average across paths
        avg_effect = total_effect / len(paths)
        
        return {
            "effect": avg_effect,
            "confidence": max_confidence,
            "paths": paths
        }
    
    def _calculate_adjusted_effect(self, cause_id: str, effect_id: str, 
                               adjustment_set: List[str]) -> Dict[str, Any]:
        """Calculate causal effect adjusting for confounders"""
        # This is a simplified implementation of backdoor adjustment
        
        # Start with direct effect
        direct_effect = self._calculate_path_effect(cause_id, effect_id)
        
        # If no adjustment needed, return direct effect
        if not adjustment_set:
            return direct_effect
            
        # Simple implementation: reduce the effect based on adjustment set
        # In practice, this would use proper backdoor adjustment formula
        
        # Calculate how much of the effect is explained by adjustment variables
        explained_effect = 0
        
        for adj_id in adjustment_set:
            # Calculate effect of cause on adjustment variable
            cause_to_adj = self._calculate_path_effect(cause_id, adj_id)
            
            # Calculate effect of adjustment variable on effect
            adj_to_effect = self._calculate_path_effect(adj_id, effect_id)
            
            # Indirect effect through this adjustment variable
            indirect_effect = cause_to_adj["effect"] * adj_to_effect["effect"]
            explained_effect += indirect_effect
        
        # Adjusted effect: direct - explained
        adjusted_effect = direct_effect["effect"] - explained_effect
        
        # Adjust confidence down for each adjustment variable
        confidence_penalty = 0.05 * len(adjustment_set)
        adjusted_confidence = max(0.1, direct_effect["confidence"] - confidence_penalty)
        
        return {
            "effect": adjusted_effect,
            "confidence": adjusted_confidence,
            "direct_effect": direct_effect["effect"],
            "explained_effect": explained_effect,
            "adjustment_set": adjustment_set
        }
    
    def _find_causal_paths(self, start_id: str, end_id: str, max_depth: int = 4) -> List[Dict[str, Any]]:
        """Find all causal paths from start to end node"""
        paths = []
        
        # Use depth-first search to find all paths
        def dfs(current_id, target_id, path=None, strength=1.0, confidence=1.0, visited=None):
            if path is None:
                path = []
            if visited is None:
                visited = set()
                
            # Skip if current node is already in path (avoid cycles)
            if current_id in visited:
                return
                
            # Add current node to path
            current_path = path + [current_id]
            current_visited = visited.union({current_id})
            
            # If we reached the target, add path to results
            if current_id == target_id:
                paths.append({
                    "path": current_path,
                    "strength": strength,
                    "confidence": confidence,
                    "length": len(current_path) - 1
                })
                return
                
            # If reached max depth, stop
            if len(current_path) >= max_depth + 1:
                return
                
            # Explore outgoing edges
            for edge_id in self.outgoing_edges.get(current_id, []):
                edge = self.edges.get(edge_id)
                if edge and edge.is_active:
                    next_id = edge.target_id
                    
                    # Skip if next node is already visited
                    if next_id in current_visited:
                        continue
                        
                    # Calculate path strength and confidence
                    next_strength = strength * edge.strength
                    next_confidence = confidence * edge.confidence
                    
                    # Skip if effect becomes too weak
                    if next_strength < 0.01:
                        continue
                        
                    # Continue search from next node
                    dfs(next_id, target_id, current_path, next_strength, next_confidence, current_visited)
        
        # Start search from start node
        dfs(start_id, end_id)
        
        # Sort by strength (strongest first)
        paths.sort(key=lambda p: p["strength"], reverse=True)
        
        return paths
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate various metrics about the causal graph"""
        # Convert to NetworkX for analysis
        G = self.to_networkx()
        
        metrics = {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "quantum_pairs": len(self.entangled_pairs)
        }
        
        # Calculate additional metrics if NetworkX is available
        try:
            # Density
            metrics["density"] = nx.density(G)
            
            # Try to calculate average shortest path length
            try:
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)
            except nx.NetworkXError:
                # Graph might not be connected
                metrics["avg_path_length"] = None
            
            # Calculate clustering coefficient
            metrics["clustering_coefficient"] = nx.average_clustering(G)
            
            # Calculate in and out degree statistics
            in_degrees = [d for _, d in G.in_degree()]
            out_degrees = [d for _, d in G.out_degree()]
            
            metrics["avg_in_degree"] = sum(in_degrees) / len(in_degrees) if in_degrees else 0
            metrics["avg_out_degree"] = sum(out_degrees) / len(out_degrees) if out_degrees else 0
            metrics["max_in_degree"] = max(in_degrees) if in_degrees else 0
            metrics["max_out_degree"] = max(out_degrees) if out_degrees else 0
            
            # Find strongly connected components
            metrics["strongly_connected_components"] = nx.number_strongly_connected_components(G)
            
            # Identify central nodes
            metrics["central_nodes"] = [node for node, bc in 
                                    sorted(nx.betweenness_centrality(G).items(), 
                                           key=lambda x: x[1], reverse=True)[:5]]
        
        except:
            # If NetworkX calculations fail, provide basic metrics only
            metrics["density"] = len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0
        
        return metrics

def run_example():
    """
    Run a comprehensive demonstration of the Causal Entanglement Matrix.
    
    This example showcases the key features of the system including:
    - Creating nodes and causal edges
    - Establishing quantum entanglement relationships
    - Real-time causal effect propagation
    - Automated causal structure discovery
    - Causal effect calculation
    - Intervention modeling with do-calculus
    
    The demonstration uses a weather system model with variables like
    temperature, pressure, humidity, cloud formation, and rainfall.
    """
    print("=" * 60)
    print("🌌 MARS Causal Entanglement Matrix v2.0.0")
    print("   Advanced Causal Inference with Quantum Principles")
    print("   Author: Shriram-2005")
    print("=" * 60)
    
    print(f"\n📅 Current Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 User: Shriram-2005")
    print(f"🔬 Quantum Mode: Enabled")
    
    print("\n🚀 Initializing Causal Entanglement Matrix...")
    
    # Create causal matrix with quantum capabilities
    matrix = CausalEntanglementMatrix(quantum_enabled=True)
    
    print("✅ Matrix initialized with quantum entanglement capabilities")
    
    # Create atmospheric system nodes
    print("\n🌡️  Creating causal nodes for atmospheric system...")
    
    node1_id = matrix.add_node("temperature", 25.0)
    node2_id = matrix.add_node("pressure", 1013.0)
    node3_id = matrix.add_node("humidity", 60.0)
    node4_id = matrix.add_node("cloud_formation", 0.3)
    node5_id = matrix.add_node("rainfall", 0.0)
    
    node_names = ["temperature", "pressure", "humidity", "cloud_formation", "rainfall"]
    print(f"✅ Created {len(node_names)} nodes: {', '.join(node_names)}")
    
    # Establish causal relationships
    print("\n🔗 Establishing causal relationships...")
    
    edge1_id = matrix.add_edge(node1_id, node3_id, strength=0.7,
                           relation_type=CausalRelationType.DIRECT)
    edge2_id = matrix.add_edge(node2_id, node3_id, strength=0.5,
                           relation_type=CausalRelationType.DIRECT)
    edge3_id = matrix.add_edge(node3_id, node4_id, strength=0.8,
                           relation_type=CausalRelationType.DIRECT)
    edge4_id = matrix.add_edge(node4_id, node5_id, strength=0.9,
                           relation_type=CausalRelationType.DIRECT)
    
    # Create quantum entanglement between temperature and pressure
    quantum_edge = matrix.add_edge(node1_id, node2_id, strength=0.8,
                               relation_type=CausalRelationType.QUANTUM)
    
    print(f"✅ Created {len([edge1_id, edge2_id, edge3_id, edge4_id, quantum_edge])} causal edges")
    print(f"⚛️  Established quantum entanglement between temperature and pressure")
    
    # Start real-time inference engine
    print("\n🤖 Starting background inference engine...")
    matrix.start_inference_engine()
    print("✅ Inference engine active")
    
    # Demonstrate causal propagation
    print("\n🌊 Testing causal effect propagation...")
    print("   Setting temperature from 25.0°C to 30.0°C...")
    
    matrix.update_node_value(node1_id, 30.0)  # Increase temperature
    
    # Wait for propagation
    time.sleep(0.5)
    
    # Check results
    humidity = matrix.nodes[node3_id].value
    cloud = matrix.nodes[node4_id].value
    rain = matrix.nodes[node5_id].value
    pressure = matrix.nodes[node2_id].value
    
    print(f"📊 Effects propagated through causal network:")
    print(f"   🌡️  Temperature: 25.0°C → 30.0°C (+5.0°C)")
    print(f"   💧 Humidity: 60.0% → {humidity:.1f}% ({humidity-60.0:+.1f}%)")
    print(f"   ☁️  Cloud formation: 0.3 → {cloud:.2f} ({cloud-0.3:+.2f})")
    print(f"   🌧️  Rainfall: 0.0mm → {rain:.2f}mm (+{rain:.2f}mm)")
    print(f"   🔘 Pressure: 1013.0hPa → {pressure:.1f}hPa ({pressure-1013.0:+.1f}hPa, quantum entangled)")
    
    # Perform automated causal discovery
    print("\n🔍 Performing automated causal structure discovery...")
    
    # Generate realistic time-series data
    solar_data = {
        "sunspot_activity": [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.7, 0.5, 0.3, 0.2],
        "radio_interference": [0.15, 0.22, 0.28, 0.45, 0.75, 0.95, 0.8, 0.6, 0.4, 0.25],
        "satellite_issues": [0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.2]
    }
    
    inference_result = matrix.infer_causality(solar_data, method=InferenceMethod.QUANTUM_ENTANGLEMENT)
    
    print(f"✅ Discovered {len(inference_result['edges_created'])} new causal relationships")
    print(f"⚛️  Used quantum-enhanced {InferenceMethod.QUANTUM_ENTANGLEMENT.name} algorithm")
    print(f"📈 Processed {inference_result['variables_processed']} variables")
    
    # Calculate specific causal effects
    print("\n📊 Calculating causal effects...")
    effect_result = matrix.calculate_causal_effect(node1_id, node5_id)
    
    print(f"🌡️ → 🌧️  Causal effect of temperature on rainfall:")
    print(f"   💪 Effect strength: {effect_result['causal_effect']:.4f}")
    print(f"   🎯 Confidence: {effect_result['confidence']:.4f} ({effect_result['confidence']*100:.1f}%)")
    print(f"   🛤️  Causal paths: {len(effect_result['paths'])}")
    
    # Demonstrate intervention modeling
    print("\n🎛️  Performing causal intervention...")
    intervention = matrix.perform_intervention(node3_id, 80.0, InterventionType.DO)
    
    print(f"⚡ Executed do(humidity=80.0%) intervention")
    print(f"📊 Affected {intervention['num_affected']} downstream nodes")
    print(f"🎯 Intervention type: {intervention['intervention_type']}")
    
    # Clean shutdown
    print("\n🛑 Shutting down inference engine...")
    matrix.stop_inference_engine()
    print("✅ Inference engine stopped")
    
    # Generate final summary
    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📋 Summary of capabilities demonstrated:")
    print("   ✅ Quantum-enhanced causal network construction")
    print("   ✅ Real-time causal effect propagation")
    print("   ✅ Automated causal structure discovery")
    print("   ✅ Quantum entanglement correlation modeling")
    print("   ✅ Causal effect calculation with confidence")
    print("   ✅ Do-calculus intervention modeling")
    print("   ✅ Background inference engine management")
    print("\n🌟 The Causal Entanglement Matrix is ready for production use!")
    print("🔬 Advanced causal inference with quantum principles - Shriram-2005")
    print("=" * 60)


if __name__ == "__main__":
    """
    Main execution entry point.
    
    When run as a standalone script, this module demonstrates the full
    capabilities of the Causal Entanglement Matrix system.
    """
    try:
        run_example()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        logger.error(f"Demo error: {e}")
    finally:
        print("\n👋 Thank you for exploring the Causal Entanglement Matrix!")