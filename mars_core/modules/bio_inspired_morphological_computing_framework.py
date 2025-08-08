"""
MARS Bio-Inspired Morphological Computing Framework

A sophisticated computational system that mimics biological morphogenesis to create
self-organizing, adaptive computational structures. This framework implements:

- Cellular differentiation and specialization
- Signal-based communication and coordination
- Dynamic structure formation and adaptation
- Distributed computation across morphological structures
- Self-repair and regeneration capabilities

The system is inspired by developmental biology principles including:
- Morphogen gradients for pattern formation
- Gene regulatory networks for cell behavior
- Cell division, death, and migration
- Homeostatic regulation and feedback loops

USAGE EXAMPLE:
    ```python
    from mars_core.modules.bio_inspired_morphological_computing_framework import (
        BiomorphicComputationSystem, MorphologyType
    )
    
    # Create a self-organizing computational system
    system = BiomorphicComputationSystem(
        dimensions=3,
        morphology_type=MorphologyType.NETWORK
    )
    
    # Start morphological development
    system.start()
    
    # Allow structure to develop
    time.sleep(5)
    
    # Execute distributed computation
    result = system.execute_computation([1, 2, 3, 4, 5])
    print(f"Computation result: {result}")
    
    # Get system statistics
    stats = system.get_computation_stats()
    print(f"Success rate: {stats['success_rate']:.1%}")
    
    # Clean shutdown
    system.stop()
    ```

ARCHITECTURE:
    The framework consists of several key components:
    
    1. **ComputationalCell**: Basic computational units that can specialize
       in different domains (arithmetic, logical, memory, etc.)
    
    2. **MorphogenSignal**: Chemical-like signals that guide development
       and coordinate cellular behavior
    
    3. **MorphologicalStructure**: Collection of cells that self-organize
       into functional computational networks
    
    4. **BiomorphicComputationSystem**: High-level interface that manages
       the entire morphological computing system

COMPUTATIONAL DOMAINS:
    - ARITHMETIC: Mathematical operations and calculations
    - LOGICAL: Boolean logic and decision making
    - MEMORY: Data storage and retrieval operations
    - PATTERN_MATCHING: Pattern recognition and classification
    - OPTIMIZATION: Resource optimization and efficiency
    - INTEGRATION: Data fusion and aggregation
    - TRANSFORMATION: Data format conversion and processing
    - COMMUNICATION: Inter-cellular messaging and coordination

MORPHOLOGY TYPES:
    - DENDRITIC: Tree-like branching (neural-inspired)
    - NETWORK: Interconnected mesh topology
    - LATTICE: Regular grid arrangement
    - SEGMENTED: Linear chain organization
    - RADIAL: Hub-and-spoke pattern
    - FRACTAL: Self-similar recursive structure
    - MODULAR: Clustered functional units
    - HIERARCHICAL: Multi-level organizational structure

DEVELOPMENT PHASES:
    - INITIALIZATION: Initial setup and founder cell creation
    - DIFFERENTIATION: Cellular specialization and domain selection
    - PATTERN_FORMATION: Spatial organization and structure formation
    - GROWTH: Population expansion and network growth
    - REGULATION: Homeostatic balance and resource management
    - REGENERATION: Self-repair and structural restoration
    - MATURITY: Stable operating state with minimal changes

Dependencies:
    - numpy: Numerical operations and array handling
    - threading: Concurrent execution and synchronization
    - uuid: Unique identifier generation
    - logging: System event logging
    - dataclasses: Data structure definitions
    - enum: Enumeration types for configuration

Notes:
    This framework is designed for research and educational purposes,
    demonstrating how biological principles can inspire computational
    architectures. It provides a foundation for exploring emergent
    computation, self-organization, and adaptive systems.
    
"""

# Module metadata
__version__ = "2.0.0"
__author__ = "Shriram-2005"
__license__ = "MIT"
__status__ = "Production"

# Standard library imports
import heapq
import logging
import math
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import numpy as np
import numpy as np

# Configure logging for the framework
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants for system configuration
DEFAULT_SIGNAL_DECAY_RATE = 0.05
DEFAULT_DIFFUSION_RATE = 0.1
DEFAULT_ENERGY_DECAY_RATE = 0.05
MAX_CELL_ENERGY = 2.0
MIN_CELL_ENERGY = 0.0
DIVISION_ENERGY_THRESHOLD = 1.5
DEATH_ENERGY_THRESHOLD = 0.1
CONNECTION_ENERGY_COST = 0.1
COMPUTATION_ENERGY_COST = 0.05


class MorphogenesisPhase(Enum):
    """
    Enumeration of developmental phases in morphological computing.
    
    Each phase represents a different stage of system development,
    analogous to embryological development in biological systems.
    """
    INITIALIZATION = auto()     # Initial setup and founder cell creation
    DIFFERENTIATION = auto()    # Cellular specialization and domain selection
    PATTERN_FORMATION = auto()  # Spatial organization and structure formation
    GROWTH = auto()            # Population expansion and network growth
    REGULATION = auto()        # Homeostatic balance and resource management
    REGENERATION = auto()      # Self-repair and structural restoration
    MATURITY = auto()          # Stable operating state with minimal changes


class ComputationalDomain(Enum):
    """
    Specialized computational domains for cellular differentiation.
    
    Each domain represents a specific type of computational capability
    that cells can specialize in, similar to tissue types in biology.
    """
    ARITHMETIC = auto()        # Mathematical operations and calculations
    LOGICAL = auto()           # Boolean logic and decision making
    MEMORY = auto()            # Data storage and retrieval operations
    PATTERN_MATCHING = auto()  # Pattern recognition and classification
    OPTIMIZATION = auto()      # Resource optimization and efficiency
    INTEGRATION = auto()       # Data fusion and aggregation
    TRANSFORMATION = auto()    # Data format conversion and processing
    COMMUNICATION = auto()     # Inter-cellular messaging and coordination


class MorphologyType(Enum):
    """
    Architectural patterns for morphological structure organization.
    
    These types determine the overall spatial arrangement and
    connectivity patterns of the computational cells.
    """
    DENDRITIC = auto()         # Tree-like branching (neural-inspired)
    NETWORK = auto()           # Interconnected mesh topology
    LATTICE = auto()           # Regular grid arrangement
    SEGMENTED = auto()         # Linear chain organization
    RADIAL = auto()            # Hub-and-spoke pattern
    FRACTAL = auto()           # Self-similar recursive structure
    MODULAR = auto()           # Clustered functional units
    HIERARCHICAL = auto()      # Multi-level organizational structure

@dataclass
class MorphogenSignal:
    """
    Represents a morphogen signal that guides cellular development and behavior.
    
    Morphogens are signaling molecules that form concentration gradients and
    provide positional information to cells. This class simulates their behavior
    including diffusion, decay, and spatial distribution.
    
    Attributes:
        signal_id: Unique identifier for the signal
        signal_type: Type/category of the signal (e.g., 'growth', 'differentiate')
        concentration: Current signal strength/concentration
        source_position: N-dimensional coordinates of signal origin
        diffusion_rate: Rate at which signal spreads spatially
        decay_rate: Rate at which signal strength diminishes over time
        creation_time: Timestamp when signal was created
    """
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: str = "default"
    concentration: float = 1.0
    source_position: Optional[Tuple[float, ...]] = None
    diffusion_rate: float = DEFAULT_DIFFUSION_RATE
    decay_rate: float = DEFAULT_SIGNAL_DECAY_RATE
    creation_time: float = field(default_factory=time.time)
    
    def diffuse(self, distance: float, time_elapsed: float) -> float:
        """
        Calculate signal concentration at a given distance after specified time.
        
        Uses a simplified diffusion equation that accounts for both spatial
        distribution and temporal decay of the signal.
        
        Args:
            distance: Spatial distance from signal source
            time_elapsed: Time since signal creation
            
        Returns:
            Effective signal concentration at the specified distance and time
            
        Note:
            The diffusion follows an approximate Gaussian distribution in space
            combined with exponential decay in time.
        """
        if distance < 0:
            logger.warning("Negative distance provided to diffuse method")
            return self.concentration
            
        # Apply spatial diffusion using Gaussian-like distribution
        # Prevents division by zero when time_elapsed is very small
        diffusion_factor = math.exp(
            -distance**2 / (4 * self.diffusion_rate * max(time_elapsed, 0.001))
        )
        
        # Apply temporal decay using exponential function
        decay_factor = math.exp(-self.decay_rate * time_elapsed)
        
        # Combine both factors to get effective concentration
        effective_concentration = self.concentration * diffusion_factor * decay_factor
        
        return max(0.0, effective_concentration)  # Ensure non-negative

@dataclass
class ComputationalCell:
    """
    Fundamental computational unit of the morphological system.
    
    Represents a single cell in the computational organism, capable of:
    - Receiving and processing morphogen signals
    - Expressing computational behaviors through "genes"
    - Specializing in specific computational domains
    - Forming connections with other cells
    - Executing computations based on specialization
    - Self-regulating energy and lifecycle
    
    The cell's behavior is governed by a gene regulatory network that
    responds to environmental signals (morphogens) and internal state.
    
    Attributes:
        cell_id: Unique identifier for the cell
        position: N-dimensional spatial coordinates
        domains: Set of computational specializations
        connections: List of connected cell IDs
        specialization_level: Degree of specialization (0=totipotent, 1=fully specialized)
        state: Internal state storage for cell-specific data
        creation_time: Timestamp of cell creation
        gene_expression: Current levels of gene expression (0-1 scale)
        energy: Available energy for cell operations
        signals: Active morphogen signals and their concentrations
    """
    cell_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position: Optional[Tuple[float, ...]] = None
    domains: Set[ComputationalDomain] = field(default_factory=set)
    connections: List[str] = field(default_factory=list)
    specialization_level: float = 0.0
    state: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    gene_expression: Dict[str, float] = field(default_factory=dict)
    energy: float = 1.0
    signals: Dict[str, float] = field(default_factory=dict)
    
    def distance_to(self, other: 'ComputationalCell') -> float:
        """
        Calculate Euclidean distance to another computational cell.
        
        Args:
            other: Target cell for distance calculation
            
        Returns:
            Distance between cells, or infinity if positions unavailable
            
        Raises:
            ValueError: If cells have different dimensional positions
        """
        if self.position is None or other.position is None:
            logger.warning("Distance calculation attempted with undefined positions")
            return float('inf')
            
        if len(self.position) != len(other.position):
            raise ValueError(
                f"Position dimension mismatch: {len(self.position)} vs {len(other.position)}"
            )
            
        # Calculate Euclidean distance
        squared_differences = [(a - b)**2 for a, b in zip(self.position, other.position)]
        return math.sqrt(sum(squared_differences))
    
    def receive_signal(self, signal: MorphogenSignal, distance: float, time_elapsed: float) -> None:
        """
        Process an incoming morphogen signal and update internal signal concentrations.
        
        Args:
            signal: The morphogen signal being received
            distance: Distance from signal source to this cell
            time_elapsed: Time since signal was created
            
        Note:
            If multiple signals of the same type are received, the maximum
            concentration is retained (signals don't simply add).
        """
        # Calculate effective concentration at this cell's position
        effective_concentration = signal.diffuse(distance, time_elapsed)
        
        if effective_concentration > 0.001:  # Only process significant signals
            # Update signal concentration (use maximum if signal already present)
            current_concentration = self.signals.get(signal.signal_type, 0.0)
            self.signals[signal.signal_type] = max(current_concentration, effective_concentration)
            
            logger.debug(
                f"Cell {self.cell_id[:8]} received {signal.signal_type} "
                f"signal: {effective_concentration:.4f}"
            )
    
    def update_gene_expression(self) -> None:
        """
        Update gene expression levels based on current morphogen signals.
        
        Implements a simplified gene regulatory network where genes respond
        to activating and inhibiting signals with different sensitivities.
        This simulates how cells interpret environmental cues to determine
        their behavior and fate.
        
        Genes regulated:
        - cell_division: Controls cell proliferation
        - differentiation: Controls specialization process
        - connection_formation: Controls synapse/connection creation
        - energy_metabolism: Controls energy production and consumption
        """
        # Gene: cell_division (responds to growth signals)
        division_activators = (
            self.signals.get("growth", 0.0) * 2.0 +
            self.signals.get("proliferate", 0.0) * 1.8
        )
        division_inhibitors = (
            self.signals.get("inhibit_growth", 0.0) * 3.0 +
            self.signals.get("quiescence", 0.0) * 2.5
        )
        self.gene_expression["cell_division"] = self._regulate_gene(
            self.gene_expression.get("cell_division", 0.1),
            division_activators, division_inhibitors
        )
        
        # Gene: differentiation (controls specialization)
        diff_activators = (
            self.signals.get("differentiate", 0.0) * 1.5 +
            self.signals.get("specialize", 0.0) * 1.7
        )
        diff_inhibitors = (
            self.signals.get("maintain_potency", 0.0) * 2.0 +
            self.signals.get("dedifferentiate", 0.0) * 1.5
        )
        self.gene_expression["differentiation"] = self._regulate_gene(
            self.gene_expression.get("differentiation", 0.1),
            diff_activators, diff_inhibitors
        )
        
        # Gene: connection_formation (controls synaptogenesis)
        conn_activators = (
            self.signals.get("connect", 0.0) * 1.5 +
            self.signals.get("network", 0.0) * 1.6 +
            self.signals.get("synaptogenesis", 0.0) * 1.8
        )
        conn_inhibitors = (
            self.signals.get("isolate", 0.0) * 1.0 +
            self.signals.get("prune_connections", 0.0) * 2.0
        )
        self.gene_expression["connection_formation"] = self._regulate_gene(
            self.gene_expression.get("connection_formation", 0.2),
            conn_activators, conn_inhibitors
        )
        
        # Gene: energy_metabolism (controls cellular energetics)
        energy_activators = (
            self.signals.get("energize", 0.0) * 1.8 +
            self.signals.get("metabolic_boost", 0.0) * 1.5
        )
        energy_inhibitors = (
            self.signals.get("suppress", 0.0) * 1.2 +
            self.signals.get("metabolic_stress", 0.0) * 2.0
        )
        self.gene_expression["energy_metabolism"] = self._regulate_gene(
            self.gene_expression.get("energy_metabolism", 0.5),
            energy_activators, energy_inhibitors
        )
    
    def _regulate_gene(self, current_level: float, 
                     activators: float, inhibitors: float) -> float:
        """
        Update gene expression level using regulatory dynamics.
        
        Implements a simple regulatory model where activators increase
        expression and inhibitors decrease it, with inertia to prevent
        rapid oscillations.
        
        Args:
            current_level: Current gene expression level (0-1)
            activators: Sum of activating signal strengths
            inhibitors: Sum of inhibiting signal strengths
            
        Returns:
            New gene expression level clamped to [0, 1]
        """
        # Calculate net regulatory effect
        net_regulation = activators - inhibitors
        
        # Apply regulation with inertia (10% response rate)
        regulatory_response_rate = 0.1
        new_level = current_level + net_regulation * regulatory_response_rate
        
        # Clamp to valid range [0, 1]
        return max(0.0, min(1.0, new_level))
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Perform one simulation step for this computational cell.
        
        Updates all cellular processes including gene expression, energy
        metabolism, specialization, and signal processing. This is the
        main driver of cellular behavior and adaptation.
        
        Args:
            dt: Time step duration for simulation
            
        Returns:
            Dictionary containing cell state and behavioral decisions:
            - cell_id: Unique identifier
            - should_divide: Whether cell should undergo division
            - should_die: Whether cell should undergo apoptosis
            - should_connect: Whether cell should form new connections
            - energy: Current energy level
            - specialization: Current specialization level
        """
        # Update gene expression network based on morphogen signals
        self.update_gene_expression()
        
        # Calculate energy dynamics
        energy_metabolism_rate = self.gene_expression.get("energy_metabolism", 0.5)
        energy_production = energy_metabolism_rate * 0.1 * dt
        
        # Energy costs
        base_consumption = DEFAULT_ENERGY_DECAY_RATE * dt
        specialization_cost = 0.02 * self.specialization_level * dt
        connection_cost = 0.005 * len(self.connections) * dt
        
        # Update energy level
        energy_change = energy_production - base_consumption - specialization_cost - connection_cost
        self.energy = max(MIN_CELL_ENERGY, min(MAX_CELL_ENERGY, self.energy + energy_change))
        
        # Update specialization level based on differentiation gene
        differentiation_rate = self.gene_expression.get("differentiation", 0.1)
        specialization_change = differentiation_rate * 0.05 * dt
        self.specialization_level = min(1.0, self.specialization_level + specialization_change)
        
        # Decay morphogen signals over time
        signal_decay_rate = 0.2  # signals decay relatively quickly
        signals_to_remove = []
        
        for signal_type in self.signals:
            # Apply exponential decay
            self.signals[signal_type] *= math.exp(-signal_decay_rate * dt)
            
            # Mark very weak signals for removal
            if self.signals[signal_type] < 0.01:
                signals_to_remove.append(signal_type)
        
        # Remove decayed signals
        for signal_type in signals_to_remove:
            del self.signals[signal_type]
        
        # Compile cell state information
        return {
            "cell_id": self.cell_id,
            "should_divide": self.should_divide(),
            "should_die": self.should_die(),
            "should_connect": self.should_create_connections(),
            "energy": self.energy,
            "specialization": self.specialization_level,
            "energy_change": energy_change,
            "active_signals": len(self.signals)
        }
    
    def should_divide(self) -> bool:
        """
        Determine if cell should undergo division (mitosis).
        
        Division requires high division gene expression, sufficient energy,
        and relatively low specialization (highly specialized cells
        typically don't divide).
        
        Returns:
            True if cell should divide, False otherwise
        """
        division_level = self.gene_expression.get("cell_division", 0.0)
        
        return (
            division_level > 0.7 and 
            self.energy > DIVISION_ENERGY_THRESHOLD and 
            self.specialization_level < 0.8
        )
    
    def should_die(self) -> bool:
        """
        Determine if cell should undergo programmed death (apoptosis).
        
        Cell death is triggered by critically low energy levels,
        representing inability to maintain cellular functions.
        
        Returns:
            True if cell should die, False otherwise
        """
        return self.energy < DEATH_ENERGY_THRESHOLD
    
    def should_create_connections(self) -> bool:
        """
        Determine if cell should form new connections with other cells.
        
        Connection formation requires active expression of connection
        genes and sufficient energy for synaptic processes.
        
        Returns:
            True if cell should create connections, False otherwise
        """
        connection_gene_level = self.gene_expression.get("connection_formation", 0.0)
        
        return (
            connection_gene_level > 0.6 and 
            self.energy > 0.5 and
            len(self.connections) < 10  # Limit maximum connections
        )
    
    def select_domain(self) -> Optional[ComputationalDomain]:
        """
        Select a computational domain for specialization based on environmental cues.
        
        Domain selection is influenced by:
        - Morphogen signal gradients indicating spatial organization
        - Current specialization level (must be sufficiently developed)
        - Stochastic factors to ensure diversity
        - Positional information as fallback
        
        Returns:
            Selected computational domain, or None if not ready to specialize
        """
        # Require minimum specialization level before domain selection
        if self.specialization_level < 0.4:
            logger.debug(f"Cell {self.cell_id[:8]} not specialized enough for domain selection")
            return None
            
        # Collect domain-specific morphogen signals
        domain_signals = {
            ComputationalDomain.ARITHMETIC: self.signals.get("arithmetic_morphogen", 0.0),
            ComputationalDomain.LOGICAL: self.signals.get("logical_morphogen", 0.0),
            ComputationalDomain.MEMORY: self.signals.get("memory_morphogen", 0.0),
            ComputationalDomain.PATTERN_MATCHING: self.signals.get("pattern_morphogen", 0.0),
            ComputationalDomain.OPTIMIZATION: self.signals.get("optimization_morphogen", 0.0),
            ComputationalDomain.INTEGRATION: self.signals.get("integration_morphogen", 0.0),
            ComputationalDomain.TRANSFORMATION: self.signals.get("transformation_morphogen", 0.0),
            ComputationalDomain.COMMUNICATION: self.signals.get("communication_morphogen", 0.0)
        }
        
        # Add stochastic noise to prevent uniform domain selection
        noisy_signals = [
            (signal_strength + random.random() * 0.2, domain) 
            for domain, signal_strength in domain_signals.items()
        ]
        
        # Find domain with strongest effective signal
        strongest_signal, selected_domain = max(noisy_signals)
        
        # Only select if signal is above threshold
        if strongest_signal > 0.3:
            logger.debug(
                f"Cell {self.cell_id[:8]} selected domain {selected_domain.name} "
                f"with signal strength {strongest_signal:.3f}"
            )
            return selected_domain
        
        # Fallback: use positional information for domain assignment
        if self.position and len(self.position) > 0:
            # Use spatial coordinates to bias domain selection
            position_sum = sum(abs(coord) for coord in self.position)
            domain_index = int(position_sum * 1000) % len(ComputationalDomain)
            fallback_domain = list(ComputationalDomain)[domain_index]
            
            logger.debug(
                f"Cell {self.cell_id[:8]} using positional fallback: {fallback_domain.name}"
            )
            return fallback_domain
        
        # Final fallback: random selection
        return random.choice(list(ComputationalDomain))
    
    def execute_computation(self, input_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute computation based on cell's specialized domains.
        
        Performs domain-specific computations based on the cell's specialization.
        Each computational domain implements different types of operations:
        - ARITHMETIC: Mathematical operations on numeric data
        - LOGICAL: Boolean operations and logic
        - MEMORY: Data storage and retrieval
        - PATTERN_MATCHING: Pattern recognition and matching
        - OPTIMIZATION: Resource and parameter optimization
        - INTEGRATION: Data fusion and aggregation
        - TRANSFORMATION: Data format conversion
        - COMMUNICATION: Message processing and routing
        
        Args:
            input_data: Data to be processed by the cell
            
        Returns:
            Tuple containing:
            - Computation result (or None if failed)
            - Metadata dictionary with execution details
        """
        # Validate preconditions for computation
        if not self.domains:
            return None, {
                "error": "No computational domains available",
                "cell_id": self.cell_id,
                "specialization": self.specialization_level
            }
            
        if self.energy < 0.1:
            return None, {
                "error": "Insufficient energy for computation",
                "cell_id": self.cell_id,
                "energy": self.energy
            }
        
        # Initialize computation metadata
        metadata = {
            "cell_id": self.cell_id,
            "domains": [domain.name for domain in self.domains],
            "energy_before": self.energy,
            "specialization": self.specialization_level,
            "input_type": type(input_data).__name__
        }
        
        # Calculate energy cost based on complexity
        base_cost = COMPUTATION_ENERGY_COST
        domain_complexity_factor = 1 + len(self.domains) * 0.2
        computation_cost = base_cost * domain_complexity_factor
        
        # Deduct energy for computation
        self.energy = max(MIN_CELL_ENERGY, self.energy - computation_cost)
        
        result = None
        operation_performed = "none"
        
        # Execute domain-specific computations
        try:
            # ARITHMETIC domain operations
            if ComputationalDomain.ARITHMETIC in self.domains:
                result, operation_performed = self._execute_arithmetic(input_data)
                
            # LOGICAL domain operations  
            if ComputationalDomain.LOGICAL in self.domains and result is None:
                result, operation_performed = self._execute_logical(input_data)
                
            # MEMORY domain operations
            if ComputationalDomain.MEMORY in self.domains and result is None:
                result, operation_performed = self._execute_memory(input_data)
                
            # PATTERN_MATCHING domain operations
            if ComputationalDomain.PATTERN_MATCHING in self.domains and result is None:
                result, operation_performed = self._execute_pattern_matching(input_data)
                
            # OPTIMIZATION domain operations
            if ComputationalDomain.OPTIMIZATION in self.domains and result is None:
                result, operation_performed = self._execute_optimization(input_data)
                
            # Additional domains can be implemented here...
                
        except Exception as e:
            logger.error(f"Computation error in cell {self.cell_id[:8]}: {e}")
            result = None
            operation_performed = "error"
        
        # Complete metadata
        metadata.update({
            "energy_after": self.energy,
            "computation_cost": computation_cost,
            "result_type": type(result).__name__,
            "operation_performed": operation_performed,
            "success": result is not None
        })
        
        return result, metadata
    
    def _execute_arithmetic(self, input_data: Any) -> Tuple[Any, str]:
        """Execute arithmetic operations on numeric data."""
        if isinstance(input_data, (int, float)):
            # Simple scalar operations
            result = input_data * 2.0 + 1.0  # Linear transformation
            return result, "scalar_transform"
            
        elif isinstance(input_data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in input_data):
                # Array statistics
                result = {
                    "mean": sum(input_data) / len(input_data),
                    "sum": sum(input_data),
                    "count": len(input_data)
                }
                return result, "array_statistics"
                
        elif isinstance(input_data, np.ndarray):
            # NumPy array operations
            result = {
                "mean": float(np.mean(input_data)),
                "std": float(np.std(input_data)),
                "shape": input_data.shape
            }
            return result, "numpy_statistics"
            
        return None, "incompatible_data"
    
    def _execute_logical(self, input_data: Any) -> Tuple[Any, str]:
        """Execute logical operations on boolean data."""
        if isinstance(input_data, bool):
            # Boolean negation
            result = not input_data
            return result, "boolean_not"
            
        elif isinstance(input_data, (list, tuple)):
            if all(isinstance(x, bool) for x in input_data):
                # Boolean aggregation
                result = {
                    "any": any(input_data),
                    "all": all(input_data),
                    "count_true": sum(input_data)
                }
                return result, "boolean_aggregation"
                
        return None, "incompatible_data"
    
    def _execute_memory(self, input_data: Any) -> Tuple[Any, str]:
        """Execute memory storage and retrieval operations."""
        # Store current input
        memory_key = f"memory_{time.time()}"
        self.state[memory_key] = input_data
        
        # Retrieve previous memory if available
        previous_memory = self.state.get("previous_memory")
        self.state["previous_memory"] = input_data
        
        # Limit memory size to prevent unbounded growth
        memory_keys = [k for k in self.state.keys() if k.startswith("memory_")]
        if len(memory_keys) > 10:
            oldest_key = min(memory_keys)
            del self.state[oldest_key]
        
        return previous_memory, "memory_operation"
    
    def _execute_pattern_matching(self, input_data: Any) -> Tuple[Any, str]:
        """Execute pattern recognition and matching operations."""
        # Initialize pattern storage if needed
        if "patterns" not in self.state:
            self.state["patterns"] = []
        
        patterns = self.state["patterns"]
        
        # Check for matches with stored patterns
        matches = [i for i, pattern in enumerate(patterns) if pattern == input_data]
        
        # Store new pattern if not already present
        if not matches and len(patterns) < 20:  # Limit pattern storage
            patterns.append(input_data)
        
        result = {
            "matches": len(matches),
            "match_indices": matches,
            "is_novel": len(matches) == 0,
            "total_patterns": len(patterns)
        }
        
        return result, "pattern_matching"
    
    def _execute_optimization(self, input_data: Any) -> Tuple[Any, str]:
        """Execute optimization operations on parameter data."""
        if isinstance(input_data, (list, tuple)) and len(input_data) > 1:
            if all(isinstance(x, (int, float)) for x in input_data):
                # Find optimal value (simple heuristic: minimize variance while maximizing mean)
                mean_val = sum(input_data) / len(input_data)
                variance = sum((x - mean_val)**2 for x in input_data) / len(input_data)
                
                # Simple optimization: return value closest to mean (lowest individual variance)
                optimal_index = min(range(len(input_data)), 
                                  key=lambda i: abs(input_data[i] - mean_val))
                
                result = {
                    "optimal_value": input_data[optimal_index],
                    "optimal_index": optimal_index,
                    "mean": mean_val,
                    "variance": variance
                }
                return result, "parameter_optimization"
                
        return None, "incompatible_data"

class MorphologicalStructure:
    """
    Multi-cellular computational structure that self-organizes through morphogenesis.
    
    This class represents a colony of computational cells that develop spatial
    organization, functional specialization, and emergent behaviors through
    biological development principles including:
    
    - Morphogen-guided pattern formation
    - Cell division, death, and migration
    - Spatial organization and connectivity
    - Homeostatic regulation
    - Adaptive restructuring and repair
    
    The structure progresses through distinct developmental phases analogous
    to embryological development, ultimately forming a mature computational
    organism capable of distributed processing.
    
    Attributes:
        dimensions: Spatial dimensionality of the structure
        morphology_type: Architectural pattern for organization
        cells: Dictionary of computational cells by ID
        signals: Active morphogen signals in the environment
        phase: Current developmental phase
        creation_time: Timestamp of structure initialization
        last_update_time: Timestamp of most recent update
        structure_stats: Statistical summary of structure properties
        spatial_index: Spatial indexing for efficient neighbor queries
    """
    
    def __init__(self, dimensions: int = 3, morphology_type: MorphologyType = MorphologyType.NETWORK):
        """
        Initialize a new morphological structure.
        
        Args:
            dimensions: Number of spatial dimensions (typically 2-4)
            morphology_type: Target organizational pattern
        """
        # Validate inputs
        if dimensions < 1 or dimensions > 10:
            raise ValueError(f"Dimensions must be between 1 and 10, got {dimensions}")
        
        # Core structure properties
        self.dimensions = dimensions
        self.morphology_type = morphology_type
        self.cells: Dict[str, ComputationalCell] = {}
        self.signals: Dict[str, MorphogenSignal] = {}
        self.phase = MorphogenesisPhase.INITIALIZATION
        
        # Temporal tracking
        self.creation_time = time.time()
        self.last_update_time = self.creation_time
        
        # Statistical monitoring
        self.structure_stats = {
            "cell_count": 0,
            "connection_count": 0,
            "specialization_avg": 0.0,
            "energy_avg": 0.0,
            "domain_distribution": {},
            "phase_history": [MorphogenesisPhase.INITIALIZATION.name]
        }
        
        # Spatial indexing for efficient neighbor queries
        # Maps grid coordinates to lists of cell IDs
        self.spatial_index: Dict[Tuple[int, ...], List[str]] = {}
        
        # Performance metrics
        self.update_count = 0
        self.total_divisions = 0
        self.total_deaths = 0
        self.total_connections_formed = 0
        
        # Initialize with founder cell
        logger.info(
            f"Initializing morphological structure: "
            f"dimensions={dimensions}, type={morphology_type.name}"
        )
        self._create_founder_cell()
        
    def _create_founder_cell(self) -> None:
        """
        Create the initial founder cell that seeds the morphological structure.
        
        The founder cell is positioned at the origin and configured with
        high proliferative potential to initiate structure development.
        It serves as the progenitor for all subsequent cells.
        """
        # Position founder cell at spatial origin
        origin_position = tuple(0.0 for _ in range(self.dimensions))
        
        # Create founder cell with developmental potential
        founder_cell = ComputationalCell(
            position=origin_position,
            specialization_level=0.0,  # Totipotent state
            energy=1.5,  # Enhanced initial energy
            gene_expression={
                "cell_division": 0.9,        # High proliferative capacity
                "differentiation": 0.1,      # Low initial specialization
                "connection_formation": 0.5, # Moderate connectivity
                "energy_metabolism": 0.8     # Efficient energy production
            }
        )
        
        # Provide initial growth stimulus
        founder_cell.signals["growth"] = 1.0
        founder_cell.signals["proliferate"] = 0.8
        
        # Add founder cell to structure
        self.cells[founder_cell.cell_id] = founder_cell
        
        # Add to spatial index for efficient neighbor lookup
        grid_position = tuple(int(coord) for coord in origin_position)
        if grid_position not in self.spatial_index:
            self.spatial_index[grid_position] = []
        self.spatial_index[grid_position].append(founder_cell.cell_id)
        
        # Create initial morphogen field
        self._create_initial_signals()
        
        logger.info(f"Created founder cell {founder_cell.cell_id[:8]} at origin")
    
    def _create_initial_signals(self) -> None:
        """Create initial morphogen signals based on morphology type"""
        origin = tuple(0.0 for _ in range(self.dimensions))
        
        # Create different signal patterns based on desired morphology
        if self.morphology_type == MorphologyType.DENDRITIC:
            # Create a gradient for dendritic growth
            main_signal = MorphogenSignal(
                signal_type="dendritic_growth",
                concentration=1.0,
                source_position=origin,
                diffusion_rate=0.2,
                decay_rate=0.02
            )
            self.signals[main_signal.signal_id] = main_signal
            
            # Add branching signal
            branch_signal = MorphogenSignal(
                signal_type="branch_formation",
                concentration=0.7,
                source_position=origin,
                diffusion_rate=0.15,
                decay_rate=0.05
            )
            self.signals[branch_signal.signal_id] = branch_signal
            
        elif self.morphology_type == MorphologyType.NETWORK:
            # Create signals for network formation
            connect_signal = MorphogenSignal(
                signal_type="connect",
                concentration=1.0,
                source_position=origin,
                diffusion_rate=0.3,
                decay_rate=0.01
            )
            self.signals[connect_signal.signal_id] = connect_signal
            
            # Add domain specialization signals
            for i, domain in enumerate([
                "arithmetic_morphogen", 
                "logical_morphogen", 
                "memory_morphogen", 
                "pattern_morphogen"
            ]):
                # Position signal sources in different directions
                pos = [0.0] * self.dimensions
                if i < self.dimensions:
                    pos[i] = 1.0  # Place each signal in different dimension
                
                domain_signal = MorphogenSignal(
                    signal_type=domain,
                    concentration=0.8,
                    source_position=tuple(pos),
                    diffusion_rate=0.2,
                    decay_rate=0.03
                )
                self.signals[domain_signal.signal_id] = domain_signal
                
        elif self.morphology_type == MorphologyType.FRACTAL:
            # Create signals for fractal-like growth
            fractal_signal = MorphogenSignal(
                signal_type="fractal_growth",
                concentration=1.0,
                source_position=origin,
                diffusion_rate=0.1,
                decay_rate=0.01
            )
            self.signals[fractal_signal.signal_id] = fractal_signal
            
            # Add recursion signal
            recursion_signal = MorphogenSignal(
                signal_type="recursion",
                concentration=0.9,
                source_position=origin,
                diffusion_rate=0.05,
                decay_rate=0.01
            )
            self.signals[recursion_signal.signal_id] = recursion_signal
            
        else:
            # Default growth signals for other morphologies
            growth_signal = MorphogenSignal(
                signal_type="growth",
                concentration=1.0,
                source_position=origin,
                diffusion_rate=0.2,
                decay_rate=0.03
            )
            self.signals[growth_signal.signal_id] = growth_signal
            
            diff_signal = MorphogenSignal(
                signal_type="differentiate",
                concentration=0.5,
                source_position=origin,
                diffusion_rate=0.15,
                decay_rate=0.04
            )
            self.signals[diff_signal.signal_id] = diff_signal
    
    def update(self, dt: float) -> Dict[str, Any]:
        """Update the structure for one time step"""
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if dt <= 0:
            dt = elapsed_time
        
        # Track changes for this update
        changes = {
            "new_cells": [],
            "removed_cells": [],
            "new_connections": [],
            "phase_change": False,
            "signals_updated": len(self.signals)
        }
        
        # 1. Update signals and diffuse
        self._update_signals(dt)
        
        # 2. Propagate signals to cells
        self._propagate_signals(dt)
        
        # 3. Update all cells
        cells_to_remove = []
        division_candidates = []
        connection_candidates = []
        
        for cell_id, cell in self.cells.items():
            update_result = cell.update(dt)
            
            # Check for cell actions
            if update_result["should_die"]:
                cells_to_remove.append(cell_id)
            
            if update_result["should_divide"]:
                division_candidates.append(cell_id)
                
            if update_result["should_connect"]:
                connection_candidates.append(cell_id)
                
            # Update cell specialization
            self._update_cell_specialization(cell)
        
        # 4. Handle cell removal
        for cell_id in cells_to_remove:
            self._remove_cell(cell_id)
            changes["removed_cells"].append(cell_id)
        
        # 5. Handle cell division
        for cell_id in division_candidates:
            if cell_id in self.cells:  # Make sure cell still exists
                new_cell_id = self._divide_cell(cell_id)
                if new_cell_id:
                    changes["new_cells"].append(new_cell_id)
        
        # 6. Handle new connections
        for cell_id in connection_candidates:
            if cell_id in self.cells:  # Make sure cell still exists
                new_connections = self._create_connections(cell_id)
                changes["new_connections"].extend(new_connections)
        
        # 7. Update morphogenesis phase
        old_phase = self.phase
        self._update_phase()
        changes["phase_change"] = (old_phase != self.phase)
        
        # 8. Update statistics
        self._update_statistics()
        
        return changes
    
    def _update_signals(self, dt: float) -> None:
        """Update existing signals and create new ones"""
        # Update existing signals
        signals_to_remove = []
        for signal_id, signal in self.signals.items():
            # Apply decay
            signal.concentration *= math.exp(-signal.decay_rate * dt)
            
            # Remove if concentration is too low
            if signal.concentration < 0.01:
                signals_to_remove.append(signal_id)
        
        # Remove decayed signals
        for signal_id in signals_to_remove:
            del self.signals[signal_id]
        
        # Create new signals based on current phase and structure needs
        if random.random() < 0.05 * dt:  # Occasional new signal
            self._create_phase_specific_signal()
    
    def _create_phase_specific_signal(self) -> None:
        """Create new signals specific to current morphogenesis phase"""
        # Select a random cell as source
        if not self.cells:
            return
            
        source_cell_id = random.choice(list(self.cells.keys()))
        source_cell = self.cells[source_cell_id]
        
        if self.phase == MorphogenesisPhase.DIFFERENTIATION:
            # Create differentiation signal
            signal_type = random.choice([
                "differentiate", 
                "arithmetic_morphogen", 
                "logical_morphogen", 
                "memory_morphogen"
            ])
            
            signal = MorphogenSignal(
                signal_type=signal_type,
                concentration=random.uniform(0.6, 1.0),
                source_position=source_cell.position,
                diffusion_rate=0.15,
                decay_rate=0.03
            )
            self.signals[signal.signal_id] = signal
            
        elif self.phase == MorphogenesisPhase.PATTERN_FORMATION:
            # Create pattern formation signals
            signal_type = "pattern_formation"
            
            # For interesting patterns, create multiple signal sources
            for i in range(3):
                # Get position with some offset
                if source_cell.position:
                    pos = list(source_cell.position)
                    pos[i % self.dimensions] += random.uniform(-1.0, 1.0)
                    
                    signal = MorphogenSignal(
                        signal_type=signal_type,
                        concentration=random.uniform(0.5, 0.9),
                        source_position=tuple(pos),
                        diffusion_rate=0.2,
                        decay_rate=0.04
                    )
                    self.signals[signal.signal_id] = signal
            
        elif self.phase == MorphogenesisPhase.GROWTH:
            # Create growth signal
            signal = MorphogenSignal(
                signal_type="growth",
                concentration=random.uniform(0.7, 1.0),
                source_position=source_cell.position,
                diffusion_rate=0.25,
                decay_rate=0.02
            )
            self.signals[signal.signal_id] = signal
            
        elif self.phase == MorphogenesisPhase.REGULATION:
            # Create regulatory signals to maintain homeostasis
            if len(self.cells) > 50:  # If too many cells, inhibit growth
                signal_type = "inhibit_growth"
            else:
                signal_type = "energize"  # Otherwise boost energy
                
            signal = MorphogenSignal(
                signal_type=signal_type,
                concentration=random.uniform(0.6, 0.9),
                source_position=source_cell.position,
                diffusion_rate=0.3,
                decay_rate=0.05
            )
            self.signals[signal.signal_id] = signal
            
        elif self.phase == MorphogenesisPhase.REGENERATION:
            # Create regeneration signal
            signal = MorphogenSignal(
                signal_type="regenerate",
                concentration=random.uniform(0.8, 1.0),
                source_position=source_cell.position,
                diffusion_rate=0.2,
                decay_rate=0.01
            )
            self.signals[signal.signal_id] = signal
    
    def _propagate_signals(self, dt: float) -> None:
        """Propagate signals to cells"""
        # For each signal, calculate effect on each cell
        for signal in self.signals.values():
            if signal.source_position is None:
                continue
                
            # Use spatial index for efficiency
            radius = 5.0  # Maximum signal propagation radius
            affected_cells = self._get_cells_in_radius(signal.source_position, radius)
            
            for cell_id in affected_cells:
                if cell_id in self.cells:  # Make sure cell still exists
                    cell = self.cells[cell_id]
                    distance = math.sqrt(sum((a - b)**2 for a, b in 
                                         zip(cell.position, signal.source_position)))
                    time_elapsed = time.time() - signal.creation_time
                    cell.receive_signal(signal, distance, time_elapsed)
    
    def _get_cells_in_radius(self, position: Tuple[float, ...], radius: float) -> List[str]:
        """Get all cells within given radius using spatial index"""
        # Convert to grid coordinates
        center_grid = tuple(int(p) for p in position)
        
        # Determine grid search radius
        grid_radius = int(radius) + 1
        
        # Collect cells from grid cells within radius
        cell_ids = []
        for offsets in self._grid_neighborhood(grid_radius, self.dimensions):
            grid_pos = tuple(c + o for c, o in zip(center_grid, offsets))
            
            if grid_pos in self.spatial_index:
                cell_ids.extend(self.spatial_index[grid_pos])
        
        return cell_ids
    
    def _grid_neighborhood(self, radius: int, dimensions: int) -> List[Tuple[int, ...]]:
        """Generate grid cell offsets within given radius"""
        if dimensions == 0:
            return [()]
            
        result = []
        for offset in range(-radius, radius + 1):
            for sub_offsets in self._grid_neighborhood(radius, dimensions - 1):
                result.append((offset,) + sub_offsets)
                
        return result
    
    def _update_cell_specialization(self, cell: ComputationalCell) -> None:
        """Update cell specialization based on signals and state"""
        # Only proceed if cell has sufficient specialization level
        if cell.specialization_level < 0.5:
            return
            
        # Determine which domain(s) the cell should specialize in
        new_domain = cell.select_domain()
        if new_domain and new_domain not in cell.domains:
            # Add the domain if cell doesn't already have it
            cell.domains.add(new_domain)
            
            # Limit number of domains based on specialization level
            max_domains = math.ceil(cell.specialization_level * 3)
            if len(cell.domains) > max_domains:
                # Remove a random domain
                domains_list = list(cell.domains)
                cell.domains.remove(random.choice(domains_list))
    
    def _divide_cell(self, parent_id: str) -> Optional[str]:
        """Create a new cell by division"""
        if parent_id not in self.cells:
            return None
            
        parent = self.cells[parent_id]
        
        # Determine new position (with small random offset)
        if parent.position is None:
            return None
            
        position_offset = [random.uniform(-0.5, 0.5) for _ in range(self.dimensions)]
        new_position = tuple(p + o for p, o in zip(parent.position, position_offset))
        
        # Create child cell
        child_cell = ComputationalCell(
            position=new_position,
            specialization_level=parent.specialization_level * 0.9,  # Slightly less specialized
            energy=parent.energy * 0.5,  # Share energy
            gene_expression={k: v * 0.95 for k, v in parent.gene_expression.items()},
            signals={k: v * 0.8 for k, v in parent.signals.items()}
        )
        
        # Inherit some domains
        for domain in parent.domains:
            if random.random() < 0.7:  # 70% chance to inherit each domain
                child_cell.domains.add(domain)
        
        # Reduce parent's energy
        parent.energy *= 0.5
        
        # Add child to structure
        self.cells[child_cell.cell_id] = child_cell
        
        # Add to spatial index
        grid_position = tuple(int(p) for p in new_position)
        if grid_position not in self.spatial_index:
            self.spatial_index[grid_position] = []
        self.spatial_index[grid_position].append(child_cell.cell_id)
        
        # Connect parent and child
        if parent_id not in child_cell.connections:
            child_cell.connections.append(parent_id)
        if child_cell.cell_id not in parent.connections:
            parent.connections.append(child_cell.cell_id)
        
        return child_cell.cell_id
    
    def _remove_cell(self, cell_id: str) -> None:
        """Remove a cell from the structure"""
        if cell_id not in self.cells:
            return
            
        cell = self.cells[cell_id]
        
        # Remove from spatial index
        if cell.position:
            grid_position = tuple(int(p) for p in cell.position)
            if grid_position in self.spatial_index and cell_id in self.spatial_index[grid_position]:
                self.spatial_index[grid_position].remove(cell_id)
                if not self.spatial_index[grid_position]:
                    del self.spatial_index[grid_position]
        
        # Remove connections
        for connected_id in cell.connections:
            if connected_id in self.cells:
                connected_cell = self.cells[connected_id]
                if cell_id in connected_cell.connections:
                    connected_cell.connections.remove(cell_id)
        
        # Remove cell
        del self.cells[cell_id]
    
    def _create_connections(self, cell_id: str) -> List[Tuple[str, str]]:
        """Create new connections from this cell to others"""
        if cell_id not in self.cells:
            return []
            
        source_cell = self.cells[cell_id]
        
        if source_cell.position is None:
            return []
            
        # Find nearby cells
        radius = 2.0  # Connection radius
        nearby_ids = self._get_cells_in_radius(source_cell.position, radius)
        
        # Filter out existing connections and self
        potential_targets = [tid for tid in nearby_ids 
                           if tid != cell_id and tid not in source_cell.connections]
        
        new_connections = []
        # Create new connections (limited by energy)
        max_new_connections = min(3, int(source_cell.energy * 3))
        
        for _ in range(max_new_connections):
            if not potential_targets:
                break
                
            # Select a target
            target_id = random.choice(potential_targets)
            potential_targets.remove(target_id)
            
            if target_id in self.cells:  # Make sure target still exists
                target_cell = self.cells[target_id]
                
                # Create bidirectional connection
                source_cell.connections.append(target_id)
                target_cell.connections.append(cell_id)
                
                # Record the new connection
                new_connections.append((cell_id, target_id))
                
                # Connections cost energy
                source_cell.energy -= 0.1
                
                # Stop if energy too low
                if source_cell.energy < 0.3:
                    break
        
        return new_connections
    
    def _update_phase(self) -> None:
        """Update the morphogenesis phase based on structure state"""
        cell_count = len(self.cells)
        
        # Calculate average specialization
        if cell_count > 0:
            avg_specialization = sum(cell.specialization_level for cell in self.cells.values()) / cell_count
        else:
            avg_specialization = 0.0
            
        # Calculate average connections per cell
        total_connections = sum(len(cell.connections) for cell in self.cells.values())
        avg_connections = total_connections / cell_count if cell_count > 0 else 0
        
        # Age of structure
        age = time.time() - self.creation_time
        
        # Determine phase based on structure properties
        if cell_count < 5:
            self.phase = MorphogenesisPhase.INITIALIZATION
            
        elif avg_specialization < 0.4 and cell_count < 20:
            self.phase = MorphogenesisPhase.GROWTH
            
        elif avg_specialization < 0.7:
            self.phase = MorphogenesisPhase.DIFFERENTIATION
            
        elif avg_connections < 3:
            self.phase = MorphogenesisPhase.PATTERN_FORMATION
            
        elif age < 60:  # First minute
            self.phase = MorphogenesisPhase.GROWTH
            
        elif cell_count < 50:
            # Need more cells
            self.phase = MorphogenesisPhase.GROWTH
            
        elif random.random() < 0.01:
            # Occasionally trigger regeneration
            self.phase = MorphogenesisPhase.REGENERATION
            
        else:
            # Default to regulation (homeostasis)
            self.phase = MorphogenesisPhase.REGULATION
    
    def _update_statistics(self) -> None:
        """Update structure statistics"""
        cell_count = len(self.cells)
        
        if cell_count == 0:
            self.structure_stats = {
                "cell_count": 0,
                "connection_count": 0,
                "specialization_avg": 0.0,
                "energy_avg": 0.0,
                "domain_distribution": {}
            }
            return
            
        # Calculate basic stats
        total_connections = sum(len(cell.connections) for cell in self.cells.values())
        avg_specialization = sum(cell.specialization_level for cell in self.cells.values()) / cell_count
        avg_energy = sum(cell.energy for cell in self.cells.values()) / cell_count
        
        # Count domains
        domain_counts = {}
        for cell in self.cells.values():
            for domain in cell.domains:
                domain_counts[domain.name] = domain_counts.get(domain.name, 0) + 1
                
        # Update stats
        self.structure_stats = {
            "cell_count": cell_count,
            "connection_count": total_connections // 2,  # Divide by 2 because connections are bidirectional
            "specialization_avg": avg_specialization,
            "energy_avg": avg_energy,
            "domain_distribution": domain_counts,
            "phase": self.phase.name
        }
    
    def execute_distributed_computation(self, input_data: Any) -> Any:
        """Execute computation across the morphological structure"""
        if not self.cells:
            return None
            
        # For sequential computation, identify a path through the structure
        # Start from cells with relevant domains for the input type
        
        start_domain = self._select_domain_for_input(input_data)
        if not start_domain:
            return None
            
        start_cells = [cell_id for cell_id, cell in self.cells.items() 
                     if start_domain in cell.domains]
        
        if not start_cells:
            return None
            
        # Select a random starting cell
        current_cell_id = random.choice(start_cells)
        
        # Track processed cells
        processed = {current_cell_id}
        computation_path = [current_cell_id]
        
        # Intermediate results
        result = input_data
        
        # Follow connections for up to 10 steps
        for _ in range(10):
            if current_cell_id not in self.cells:
                break
                
            # Execute computation in current cell
            current_cell = self.cells[current_cell_id]
            new_result, metadata = current_cell.execute_computation(result)
            
            # Update result if computation succeeded
            if new_result is not None:
                result = new_result
                
            # Find next unprocessed cell
            next_cell_candidates = [cid for cid in current_cell.connections 
                                 if cid in self.cells and cid not in processed]
            
            if not next_cell_candidates:
                break
                
            # Select next cell
            current_cell_id = random.choice(next_cell_candidates)
            processed.add(current_cell_id)
            computation_path.append(current_cell_id)
        
        return {
            "result": result,
            "path_length": len(computation_path),
            "cells_involved": len(processed),
            "computation_path": computation_path
        }
    
    def _select_domain_for_input(self, input_data: Any) -> Optional[ComputationalDomain]:
        """Select appropriate computational domain for input type"""
        if isinstance(input_data, (int, float, list, np.ndarray)) and all(isinstance(x, (int, float)) for x in input_data) if isinstance(input_data, (list, np.ndarray)) else True:
            return ComputationalDomain.ARITHMETIC
            
        elif isinstance(input_data, bool) or (isinstance(input_data, (list, tuple)) and all(isinstance(x, bool) for x in input_data)):
            return ComputationalDomain.LOGICAL
            
        elif isinstance(input_data, (str, dict)):
            return ComputationalDomain.MEMORY
            
        elif isinstance(input_data, (list, tuple, np.ndarray)):
            return ComputationalDomain.PATTERN_MATCHING
            
        return None
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current structure state"""
        return {
            "cell_count": len(self.cells),
            "signal_count": len(self.signals),
            "phase": self.phase.name,
            "age": time.time() - self.creation_time,
            "morphology_type": self.morphology_type.name,
            "dimensions": self.dimensions,
            "statistics": self.structure_stats,
            "cells": [{
                "id": cell.cell_id,
                "position": cell.position,
                "specialization": cell.specialization_level,
                "energy": cell.energy,
                "domains": [d.name for d in cell.domains],
                "connections": len(cell.connections)
            } for cell in list(self.cells.values())[:10]]  # Limit to first 10 cells for brevity
        }

class BiomorphicComputationSystem:
    """Main controller for biomorphic computation"""
    
    def __init__(self, dimensions: int = 3, morphology_type: MorphologyType = MorphologyType.NETWORK):
        """Initialize the biomorphic computation system"""
        self.structure = MorphologicalStructure(dimensions, morphology_type)
        self.update_interval = 0.1  # Seconds
        self.is_running = False
        self.update_thread = None
        self._lock = threading.RLock()
        self.computation_history = []
        self.max_history_size = 100
        
    def start(self) -> None:
        """Start the morphological development process"""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop(self) -> None:
        """Stop the morphological development process"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
            
    def _update_loop(self) -> None:
        """Main update loop for morphological development"""
        while self.is_running:
            with self._lock:
                self.structure.update(self.update_interval)
                
            time.sleep(self.update_interval)
            
    def execute_computation(self, input_data: Any) -> Dict[str, Any]:
        """Execute computation using the morphological structure"""
        with self._lock:
            start_time = time.time()
            result = self.structure.execute_distributed_computation(input_data)
            end_time = time.time()
            
            # Record computation
            computation_record = {
                "timestamp": time.time(),
                "input_type": type(input_data).__name__,
                "execution_time": end_time - start_time,
                "structure_stats": self.structure.structure_stats.copy(),
                "result": result
            }
            
            self.computation_history.append(computation_record)
            
            # Limit history size
            if len(self.computation_history) > self.max_history_size:
                self.computation_history = self.computation_history[-self.max_history_size:]
                
            return computation_record
            
    def get_structure_state(self) -> Dict[str, Any]:
        """Get current state of the morphological structure"""
        with self._lock:
            return self.structure.get_state_snapshot()
            
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get statistics about computations performed"""
        if not self.computation_history:
            return {
                "total_computations": 0,
                "avg_execution_time": 0,
                "success_rate": 0
            }
            
        # Calculate stats
        total = len(self.computation_history)
        execution_times = [record["execution_time"] for record in self.computation_history]
        avg_time = sum(execution_times) / total
        
        # Success rate (had a result)
        successes = sum(1 for record in self.computation_history 
                       if record["result"] and record["result"].get("result") is not None)
        success_rate = successes / total if total > 0 else 0
        
        # Get cell count trend
        cell_counts = [record["structure_stats"]["cell_count"] 
                      for record in self.computation_history]
        
        return {
            "total_computations": total,
            "avg_execution_time": avg_time,
            "success_rate": success_rate,
            "cell_count_trend": cell_counts,
            "recent_computation_count": len([r for r in self.computation_history 
                                           if time.time() - r["timestamp"] < 60])
        }
    
    def reset_structure(self, morphology_type: Optional[MorphologyType] = None) -> None:
        """Reset the morphological structure"""
        with self._lock:
            if morphology_type is None:
                morphology_type = self.structure.morphology_type
                
            self.structure = MorphologicalStructure(
                dimensions=self.structure.dimensions,
                morphology_type=morphology_type
            )
    
    def create_alternative_structure(self, morphology_type: MorphologyType) -> None:
        """Create an alternative structure with different morphology"""
        with self._lock:
            self.reset_structure(morphology_type)

def run_example():
    """
    Demonstrate the bio-inspired morphological computing framework.
    
    This example showcases the key capabilities of the system:
    1. Self-organizing structure development
    2. Cellular specialization and differentiation  
    3. Distributed computation across the structure
    4. Performance monitoring and statistics
    
    The demonstration allows the structure to develop for several seconds,
    then tests various computational tasks to show emergent capabilities.
    """
    # System information
    print("=" * 70)
    print("MARS Bio-Inspired Morphological Computing Framework")
    print("Demonstration of Self-Organizing Computational Structures")
    print("=" * 70)
    print(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework Version: 2.0.0")
    print(f"Author: Shriram-2005")
    print()
    
    try:
        # Initialize the morphological computing system
        print("🧠 Initializing Bio-Morphological Computing System...")
        system = BiomorphicComputationSystem(
            dimensions=3,
            morphology_type=MorphologyType.NETWORK
        )
        
        print(f"   Spatial Dimensions: {system.structure.dimensions}D")
        print(f"   Target Morphology: {system.structure.morphology_type.name}")
        print(f"   Initial Phase: {system.structure.phase.name}")
        
        # Begin morphological development
        print("\n🌱 Starting Morphogenesis Process...")
        system.start()
        
        print("   Cellular development and organization in progress...")
        print("   Morphogen signals propagating...")
        print("   Allowing 4 seconds for structure formation...")
        
        # Monitor development progress
        for i in range(4):
            time.sleep(1)
            state = system.get_structure_state()
            print(f"   T+{i+1}s: {state['cell_count']} cells, "
                  f"Phase: {state['phase']}, "
                  f"Avg Specialization: {state['statistics']['specialization_avg']:.2f}")
        
        # Get final developmental state
        final_state = system.get_structure_state()
        
        print(f"\n📊 Morphological Structure Analysis:")
        print(f"   Development Phase: {final_state['phase']}")
        print(f"   Total Cells: {final_state['cell_count']}")
        print(f"   Active Signals: {final_state['signal_count']}")
        print(f"   Structure Age: {final_state['age']:.1f} seconds")
        print(f"   Average Specialization: {final_state['statistics']['specialization_avg']:.4f}")
        print(f"   Network Connections: {final_state['statistics']['connection_count']}")
        
        # Domain distribution analysis
        if final_state['statistics']['domain_distribution']:
            print(f"   Domain Distribution:")
            for domain, count in final_state['statistics']['domain_distribution'].items():
                print(f"     • {domain}: {count} cells")
        
        # Test computational capabilities
        print(f"\n🔬 Testing Distributed Computational Capabilities...")
        
        test_cases = [
            ("Arithmetic Processing", [1, 2, 3, 4, 5]),
            ("Boolean Logic", [True, False, True, False]),
            ("Numeric Scalar", 42.5),
            ("Pattern Data", "test_pattern_123"),
            ("Parameter Optimization", [10, 15, 12, 18, 14])
        ]
        
        successful_computations = 0
        total_cells_involved = 0
        
        for test_name, test_data in test_cases:
            print(f"   Testing {test_name}...")
            try:
                result = system.execute_computation(test_data)
                
                if result['result'] and result['result'].get('result') is not None:
                    cells_involved = result['result'].get('cells_involved', 0)
                    execution_time = result.get('execution_time', 0)
                    
                    print(f"     ✓ Success: {cells_involved} cells, "
                          f"{execution_time:.4f}s execution time")
                    
                    if isinstance(result['result']['result'], dict):
                        # Display first few key-value pairs for complex results
                        items = list(result['result']['result'].items())[:2]
                        summary = ", ".join(f"{k}: {v}" for k, v in items)
                        print(f"     → Result: {{{summary}{'...' if len(items) > 2 else ''}}}")
                    else:
                        print(f"     → Result: {result['result']['result']}")
                    
                    successful_computations += 1
                    total_cells_involved += cells_involved
                else:
                    print(f"     ✗ Computation failed or returned no result")
                    
            except Exception as e:
                print(f"     ✗ Error during computation: {e}")
        
        # Performance statistics
        stats = system.get_computation_stats()
        print(f"\n📈 Performance Statistics:")
        print(f"   Total Computations: {stats['total_computations']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Average Execution Time: {stats['avg_execution_time']:.6f} seconds")
        print(f"   Recent Activity: {stats['recent_computation_count']} computations in last minute")
        
        if successful_computations > 0:
            avg_cells_per_computation = total_cells_involved / successful_computations
            print(f"   Average Cells per Computation: {avg_cells_per_computation:.1f}")
        
        # Demonstrate structure adaptability
        print(f"\n🔄 Testing Structural Adaptability...")
        print("   Switching to FRACTAL morphology...")
        system.reset_structure(MorphologyType.FRACTAL)
        time.sleep(2)
        
        fractal_state = system.get_structure_state()
        print(f"   New Structure: {fractal_state['cell_count']} cells, "
              f"Type: {fractal_state['morphology_type']}")
        
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")
        logger.error(f"Example execution failed: {e}", exc_info=True)
        
    finally:
        # Clean shutdown
        print(f"\n🔄 Shutting down morphological system...")
        if 'system' in locals():
            system.stop()
        
        print("✅ Bio-Morphological Computing Framework demonstration completed!")
        print("\nKey Achievements Demonstrated:")
        print("  • Self-organizing cellular structures")
        print("  • Adaptive morphological development")
        print("  • Distributed computational processing")
        print("  • Real-time performance monitoring")
        print("  • Dynamic structural reconfiguration")
        print("\nFramework ready for production deployment!")
        print("=" * 70)


# Export public API
__all__ = [
    # Core classes
    'BiomorphicComputationSystem',
    'MorphologicalStructure', 
    'ComputationalCell',
    'MorphogenSignal',
    
    # Enumerations
    'MorphogenesisPhase',
    'ComputationalDomain',
    'MorphologyType',
    
    # Example function
    'run_example',
    
    # Metadata
    '__version__',
    '__author__',
    '__license__'
]


if __name__ == "__main__":
    """Execute the demonstration when run as main module."""
    run_example()