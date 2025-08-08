"""
MARS Cognitive Manifold Evolution System

A sophisticated implementation of cognitive evolution using multidimensional manifold theory.
This module provides a framework for representing and evolving cognitive concepts within
a geometric manifold space, enabling advanced reasoning patterns through spatial relationships.

The system models cognitive elements as points in a high-dimensional manifold, where:
- Spatial proximity represents conceptual similarity
- Element movement represents learning and adaptation
- Manifold geometry encodes different cognitive structures
- Evolution strategies enable dynamic knowledge refinement

Key Features:
- Multiple manifold geometries (Euclidean, Hyperbolic, Spherical, etc.)
- Dynamic spatial indexing for efficient neighbor queries
- Adaptive evolution strategies for knowledge optimization
- Cognitive process simulation (abstraction, synthesis, analogy, etc.)
- Thread-safe operations for concurrent access

Example Usage:
    # Create a cognitive manifold evolution system
    system = CognitiveManifoldEvolution(
        dimensions=8,
        manifold_type=ManifoldType.MIXED_CURVATURE
    )
    
    # Add cognitive concepts
    concept_id = system.add_concept("Neural networks", type_tags=["ai", "ml"])
    
    # Apply cognitive processes
    result = system.apply_process(CognitiveProcess.ASSOCIATION, [concept_id])
    
    # Evolve the system
    evolution_stats = system.evolve_once(EvolutionStrategy.GRADIENT_ASCENT)

Dependencies:
    - numpy: Numerical computing for manifold operations
    - scipy: Scientific computing for sparse matrices and optimization
    - threading: Concurrent access and background processing
    - pickle: Serialization for persistence
    
"""

# Standard library imports
import heapq
import math
import os
import pickle
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance_matrix

# Module constants
__version__ = "2.0.0"
__author__ = "Shriram-2005"

# Default configuration constants
DEFAULT_DIMENSIONS = 8
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MUTATION_RATE = 0.05
DEFAULT_SELECTION_PRESSURE = 0.7
DEFAULT_EXPLORATION_RATE = 0.3

# Spatial indexing constants
DEFAULT_GRID_RESOLUTION = 0.2
FINE_GRID_RESOLUTION = 0.1
NEIGHBOR_SEARCH_RADIUS = 1

# Evolution constants
MAX_EVOLUTION_ITERATIONS = 100
MIN_FITNESS_THRESHOLD = 0.1
COMPLEXITY_SCALE_FACTOR = 1000
ACTIVATION_DECAY_RATE = 0.1

class ManifoldType(Enum):
    """
    Enumeration of supported cognitive manifold geometries.
    
    Each manifold type represents a different geometric structure that can encode
    cognitive relationships in unique ways:
    
    - EUCLIDEAN: Flat space with standard distance metrics
    - HYPERBOLIC: Negative curvature, good for hierarchical structures  
    - SPHERICAL: Positive curvature, finite space with wrapping
    - TOROIDAL: Donut-shaped with periodic boundaries
    - MIXED_CURVATURE: Variable curvature across dimensions
    - RIEMANN: General Riemannian manifold structure
    - SYMPLECTIC: Preserves symplectic structure (Hamiltonian systems)
    - PROJECTIVE: Projective geometry with infinity points
    - CALABI_YAU: Complex manifold structure
    - EMERGENT: Self-organizing based on element distribution
    """
    EUCLIDEAN = auto()           # Flat manifold with standard metrics
    HYPERBOLIC = auto()          # Negative curvature manifold
    SPHERICAL = auto()           # Positive curvature, spherical geometry
    TOROIDAL = auto()            # Donut-shaped with periodic boundaries
    MIXED_CURVATURE = auto()     # Variable curvature across dimensions
    RIEMANN = auto()             # General Riemannian manifold
    SYMPLECTIC = auto()          # Symplectic manifold structure
    PROJECTIVE = auto()          # Projective geometry
    CALABI_YAU = auto()          # Complex manifold structure
    EMERGENT = auto()            # Self-organizing manifold


class CognitiveProcess(Enum):
    """
    Enumeration of cognitive processes that can be applied to manifold elements.
    
    These processes simulate different types of human-like reasoning and
    knowledge manipulation operations:
    
    - ABSTRACTION: Extract general patterns from specific instances
    - ASSOCIATION: Find related concepts based on similarity
    - INFERENCE: Draw logical conclusions from premises
    - PREDICTION: Forecast future states or outcomes
    - ANALOGY: Find structural similarities between different domains
    - GENERALIZATION: Extend specific knowledge to broader contexts
    - CATEGORIZATION: Group concepts into meaningful categories
    - SYNTHESIS: Combine multiple concepts into new composite ideas
    - DECOMPOSITION: Break complex concepts into simpler components
    - TRANSFORMATION: Modify concepts through systematic changes
    - REFLECTION: Meta-cognitive analysis of reasoning processes
    - OPTIMIZATION: Improve representation efficiency and accuracy
    """
    ABSTRACTION = auto()         # Form abstract concepts from instances
    ASSOCIATION = auto()         # Create conceptual associations
    INFERENCE = auto()           # Draw logical inferences
    PREDICTION = auto()          # Make predictions about future states
    ANALOGY = auto()             # Find analogical relationships
    GENERALIZATION = auto()      # Generalize from specific cases
    CATEGORIZATION = auto()      # Categorize concepts into groups
    SYNTHESIS = auto()           # Combine concepts into new ideas
    DECOMPOSITION = auto()       # Break concepts into components
    TRANSFORMATION = auto()      # Transform concept representations
    REFLECTION = auto()          # Meta-cognitive reflection
    OPTIMIZATION = auto()        # Optimize cognitive representations


class EvolutionStrategy(Enum):
    """
    Enumeration of strategies for evolving the cognitive manifold.
    
    Each strategy represents a different approach to improving the manifold's
    organization and knowledge representation:
    
    - GRADIENT_ASCENT: Gradual improvement following fitness gradients
    - EXPLORATION: Random exploration of new conceptual territories
    - EXPLOITATION: Refine and optimize existing knowledge areas
    - BIFURCATION: Create conceptual branches from successful ideas
    - PRUNING: Remove low-performing or outdated concepts
    - CONSOLIDATION: Merge similar concepts to reduce redundancy
    - RESTRUCTURING: Reorganize manifold topology and connections
    - DIMENSIONALITY: Adapt the manifold's dimensional structure
    - CURVATURE: Modify manifold curvature properties
    - PUNCTUATED: Implement rapid evolutionary bursts
    - SYMBIOTIC: Enable co-evolution between related concepts
    - QUANTUM_LEAP: Execute discontinuous improvements
    """
    GRADIENT_ASCENT = auto()     # Gradual fitness-based improvement
    EXPLORATION = auto()         # Random exploration of new regions
    EXPLOITATION = auto()        # Refine existing knowledge areas
    BIFURCATION = auto()         # Create conceptual branches
    PRUNING = auto()             # Remove underperforming elements
    CONSOLIDATION = auto()       # Merge similar concepts
    RESTRUCTURING = auto()       # Reorganize manifold structure
    DIMENSIONALITY = auto()      # Adapt dimensional structure
    CURVATURE = auto()           # Modify manifold curvature
    PUNCTUATED = auto()          # Implement evolutionary bursts
    SYMBIOTIC = auto()           # Enable concept co-evolution
    QUANTUM_LEAP = auto()        # Execute discontinuous improvements

@dataclass
class CognitiveElement:
    """
    Represents a single cognitive element within the manifold space.
    
    A CognitiveElement encapsulates both the geometric representation (coordinates, velocity)
    and semantic content of a cognitive concept. Elements can move through the manifold,
    form connections with other elements, and evolve over time.
    
    Attributes:
        element_id (str): Unique identifier for this element
        coordinates (np.ndarray): Position in the manifold space
        velocity (np.ndarray): Movement vector for dynamic positioning
        content (Any): The actual cognitive content (concept, data, etc.)
        creation_time (float): Timestamp when element was created
        last_updated (float): Timestamp of most recent update
        fitness (float): Quality/usefulness score [0.0, 1.0]
        stability (float): Resistance to change [0.0, 1.0]
        complexity (float): Measure of concept complexity [0.0, 1.0]
        activation (float): Current activation level [0.0, 1.0]
        type_tags (Set[str]): Categorical tags for indexing
        connections (Dict[str, float]): Connections to other elements
        metadata (Dict[str, Any]): Additional element metadata
    """
    
    # Core identification and spatial properties
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coordinates: Optional[np.ndarray] = None   # Position in manifold space
    velocity: Optional[np.ndarray] = None      # Movement vector for dynamics
    
    # Content and semantic properties
    content: Any = None                        # Actual cognitive content
    
    # Temporal properties
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Quality and fitness metrics
    fitness: float = 0.5                       # Overall quality score [0.0, 1.0]
    stability: float = 0.5                     # Resistance to change [0.0, 1.0]
    complexity: float = 0.0                    # Content complexity [0.0, 1.0]
    activation: float = 0.0                    # Current activation [0.0, 1.0]
    
    # Organizational and relational properties
    type_tags: Set[str] = field(default_factory=set)              # Category tags
    connections: Dict[str, float] = field(default_factory=dict)   # element_id -> strength
    metadata: Dict[str, Any] = field(default_factory=dict)        # Additional data
    
    def update_coordinates(self, new_coords: np.ndarray, 
                         manifold_type: ManifoldType = ManifoldType.EUCLIDEAN) -> None:
        """
        Update element coordinates with manifold-appropriate projection.
        
        Args:
            new_coords: New coordinate values to set
            manifold_type: Type of manifold for proper coordinate projection
            
        Note:
            Coordinates are automatically projected to respect manifold constraints.
            For example, spherical manifolds normalize to unit sphere.
        """
        if new_coords is None:
            return
            
        self.coordinates = self._project_to_manifold(new_coords, manifold_type)
        self.last_updated = time.time()
    
    def update_velocity(self, new_velocity: np.ndarray, damping: float = 0.9) -> None:
        """
        Update element velocity with optional damping.
        
        Args:
            new_velocity: New velocity vector to apply
            damping: Damping factor for existing velocity [0.0, 1.0]
            
        Note:
            Velocity is blended with existing velocity using the damping factor.
            Higher damping preserves more of the existing velocity.
        """
        if new_velocity is None:
            return
            
        if self.velocity is None:
            self.velocity = new_velocity.copy()
        self.last_updated = time.time()
    
    def update_fitness(self, fitness_update: float) -> None:
        """
        Update element fitness score using exponential smoothing.
        
        Args:
            fitness_update: New fitness value to incorporate [0.0, 1.0]
            
        Note:
            Uses exponential smoothing (90% old, 10% new) to prevent
            sudden fitness changes while allowing gradual adaptation.
        """
        # Blend with existing fitness using exponential smoothing
        self.fitness = 0.9 * self.fitness + 0.1 * fitness_update
        self.fitness = max(0.0, min(1.0, self.fitness))  # Clamp to valid range
        self.last_updated = time.time()
    
    def update_activation(self, activation: float, decay: bool = True) -> None:
        """
        Update activation level with optional temporal decay.
        
        Args:
            activation: Activation value to add [0.0, 1.0]
            decay: Whether to apply temporal decay to existing activation
            
        Note:
            Activation naturally decays over time with exponential decay rate.
            New activation is added to the decayed existing activation.
        """
        if decay:
            # Apply exponential decay based on time since last update
            time_delta = time.time() - self.last_updated
            decay_factor = math.exp(-0.1 * time_delta)  # 10% decay rate
            self.activation *= decay_factor
            
        # Add new activation
        self.activation += activation
        self.activation = max(0.0, min(1.0, self.activation))  # Clamp to valid range
        self.last_updated = time.time()
    
    def add_connection(self, target_id: str, strength: float) -> None:
        """
        Add or update a connection to another element.
        
        Args:
            target_id: ID of the target element to connect to
            strength: Connection strength [0.0, 1.0]
            
        Note:
            Connection strengths represent the degree of conceptual
            relationship between elements.
        """
        self.connections[target_id] = max(0.0, min(1.0, strength))
        self.last_updated = time.time()
    
    def remove_connection(self, target_id: str) -> None:
        """
        Remove a connection to another element.
        
        Args:
            target_id: ID of the target element to disconnect from
        """
        if target_id in self.connections:
            del self.connections[target_id]
            self.last_updated = time.time()
    
    def get_age(self) -> float:
        """
        Get the age of this element in seconds.
        
        Returns:
            Age in seconds since element creation
        """
        return time.time() - self.creation_time
    
    
    def _project_to_manifold(self, coords: np.ndarray, 
                          manifold_type: ManifoldType) -> np.ndarray:
        """
        Project coordinates onto the specified manifold geometry.
        
        Args:
            coords: Raw coordinates to project
            manifold_type: Type of manifold geometry to project onto
            
        Returns:
            Projected coordinates that respect manifold constraints
            
        Note:
            Each manifold type has specific geometric constraints:
            - Euclidean: No constraints (identity projection)
            - Hyperbolic: Projects to Poincaré disk (unit disk)
            - Spherical: Projects to unit sphere surface
            - Toroidal: Wraps coordinates within unit cube
        """
        if coords is None:
            return None
            
        if manifold_type == ManifoldType.EUCLIDEAN:
            # Euclidean space requires no projection
            return coords.copy()
            
        elif manifold_type == ManifoldType.HYPERBOLIC:
            # Project to Poincaré disk model of hyperbolic space
            norm = np.linalg.norm(coords)
            if norm >= 1.0:
                # Ensure coordinates stay within unit disk
                return coords / (norm + 1e-10) * 0.99
            return coords.copy()
            
        elif manifold_type == ManifoldType.SPHERICAL:
            # Project to unit sphere surface
            norm = np.linalg.norm(coords)
            if norm > 1e-10:  # Avoid division by zero
                return coords / norm
            # If zero vector, return arbitrary point on sphere
            result = np.zeros_like(coords)
            result[0] = 1.0
            return result
            
        elif manifold_type == ManifoldType.TOROIDAL:
            # Project to torus by wrapping coordinates within unit cube
            return coords % 1.0
            
        else:
            # For other manifold types, default to Euclidean projection
            return coords.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert element to dictionary representation for serialization.
        
        Returns:
            Dictionary containing all element properties in JSON-serializable format
            
        Note:
            NumPy arrays are converted to lists for JSON compatibility.
            This is useful for saving/loading and debugging purposes.
        """
        return {
            "element_id": self.element_id,
            "coordinates": self.coordinates.tolist() if self.coordinates is not None else None,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "fitness": self.fitness,
            "stability": self.stability,
            "complexity": self.complexity,
            "activation": self.activation,
            "type_tags": list(self.type_tags),
            "connection_count": len(self.connections),
            "metadata": self.metadata
        }

class CognitiveManifold:
    """
    Multi-dimensional manifold representing cognitive space.
    
    The CognitiveManifold is the core container that manages a collection of
    CognitiveElements within a geometric space. It provides:
    
    - Spatial organization and indexing of cognitive elements
    - Manifold-aware distance calculations and projections
    - Dynamic evolution strategies for knowledge optimization
    - Cognitive process simulation (association, synthesis, etc.)
    - Thread-safe operations for concurrent access
    - Persistence capabilities for saving/loading states
    
    The manifold uses spatial indexing for efficient neighbor queries and
    maintains various indices for fast lookups by type, fitness, etc.
    
    Attributes:
        dimensions (int): Dimensionality of the manifold space
        manifold_type (ManifoldType): Geometric structure type
        elements (Dict[str, CognitiveElement]): All elements in the manifold
        spatial_index (Dict[str, List[str]]): Grid-based spatial index
        type_index (Dict[str, Set[str]]): Index by element type tags
        stats (Dict[str, Any]): Global manifold statistics
        manifold_properties (Dict[str, Any]): Geometric properties
        evolution_params (Dict[str, Any]): Evolution algorithm parameters
    """
    
    def __init__(self, dimensions: int = 8, manifold_type: ManifoldType = ManifoldType.EUCLIDEAN):
        """
        Initialize a new cognitive manifold.
        
        Args:
            dimensions: Number of dimensions in the manifold space (default: 8)
            manifold_type: Geometric structure of the manifold (default: Euclidean)
            
        Note:
            Higher dimensions allow more complex relationships but increase
            computational cost. 8 dimensions provide a good balance for most
            cognitive modeling tasks.
        """
        # Core manifold properties
        self.dimensions = dimensions
        self.manifold_type = manifold_type
        
        # Storage for cognitive elements
        self.elements: Dict[str, CognitiveElement] = {}
        
        # Spatial indexing for efficient neighbor queries
        # Grid-based spatial index: grid_cell_id -> [element_ids]
        self.spatial_index: Dict[str, List[str]] = {}
        
        # Type-based indexing for fast categorization queries
        # type_tag -> {element_ids}
        self.type_index: Dict[str, Set[str]] = {}
        
        # Global manifold statistics
        self.stats = {
            "creation_time": time.time(),      # When manifold was created
            "last_evolved": time.time(),       # Last evolution timestamp
            "evolution_count": 0,              # Number of evolution cycles
            "total_fitness": 0.0,              # Sum of all element fitness
            "avg_complexity": 0.0,             # Average element complexity
            "avg_stability": 0.5,              # Average element stability
            "curvature": 0.0                   # Estimated manifold curvature
        }
        
        # Geometric properties of the manifold
        self.manifold_properties = {
            "curvature": 0.0,                  # Scalar curvature measure
            "diameter": 1.0,                   # Maximum distance between points
            "volume": 1.0,                     # Manifold volume measure
            "dimensionality": dimensions       # Effective dimensionality (may adapt)
        }
        
        # Evolution algorithm parameters
        self.evolution_params = {
            "learning_rate": 0.01,             # Rate of adaptation/change
            "temperature": 1.0,                # Exploration vs exploitation balance
            "mutation_rate": 0.05,             # Probability of random changes
            "selection_pressure": 0.7,         # Strength of fitness-based selection
            "exploration_rate": 0.3            # Rate of exploring new regions
        }
        
        # Thread synchronization for concurrent access
        self._lock = threading.RLock()
    
    def add_element(self, content: Any, coordinates: np.ndarray = None, 
                  type_tags: List[str] = None, fitness: float = 0.5) -> str:
        """
        Add a new cognitive element to the manifold.
        
        Args:
            content: The cognitive content to store (any serializable object)
            coordinates: Initial position in manifold (random if None)
            type_tags: Categorical tags for organization (optional)
            fitness: Initial fitness score [0.0, 1.0] (default: 0.5)
            
        Returns:
            Unique identifier for the created element
            
        Note:
            If coordinates are not provided, they are randomly generated
            according to the manifold type. The element is automatically
            indexed for efficient spatial and type-based queries.
        """
        with self._lock:
            # Generate random coordinates if not provided
            if coordinates is None:
                coordinates = self._generate_random_coordinates()
                
            # Project coordinates to manifold
            coordinates = self._project_to_manifold(coordinates)
            
            # Initialize velocity
            velocity = np.zeros_like(coordinates)
            
            # Create the element
            element = CognitiveElement(
                coordinates=coordinates,
                velocity=velocity,
                content=content,
                fitness=fitness,
                type_tags=set(type_tags) if type_tags else set()
            )
            
            # Calculate complexity
            element.complexity = self._calculate_complexity(content)
            
            # Add to manifold
            element_id = element.element_id
            self.elements[element_id] = element
            
            # Add to spatial index
            grid_cell = self._get_grid_cell(coordinates)
            if grid_cell not in self.spatial_index:
                self.spatial_index[grid_cell] = []
            self.spatial_index[grid_cell].append(element_id)
            
            # Add to type indices
            for tag in element.type_tags:
                if tag not in self.type_index:
                    self.type_index[tag] = set()
                self.type_index[tag].add(element_id)
            
            # Update statistics
            self._update_stats()
            
            return element_id
    
    def remove_element(self, element_id: str) -> bool:
        """Remove an element from the manifold"""
        with self._lock:
            if element_id not in self.elements:
                return False
                
            element = self.elements[element_id]
            
            # Remove from spatial index
            if element.coordinates is not None:
                grid_cell = self._get_grid_cell(element.coordinates)
                if grid_cell in self.spatial_index and element_id in self.spatial_index[grid_cell]:
                    self.spatial_index[grid_cell].remove(element_id)
                    if not self.spatial_index[grid_cell]:
                        del self.spatial_index[grid_cell]
            
            # Remove from type indices
            for tag in element.type_tags:
                if tag in self.type_index and element_id in self.type_index[tag]:
                    self.type_index[tag].remove(element_id)
                    if not self.type_index[tag]:
                        del self.type_index[tag]
            
            # Remove connections to this element
            for other_id, other_element in self.elements.items():
                if element_id in other_element.connections:
                    other_element.remove_connection(element_id)
            
            # Remove the element
            del self.elements[element_id]
            
            # Update statistics
            self._update_stats()
            
            return True
    
    def get_element(self, element_id: str) -> Optional[CognitiveElement]:
        """Get an element by ID"""
        return self.elements.get(element_id)
    
    def update_element(self, element_id: str, content: Any = None, 
                     coordinates: np.ndarray = None, 
                     velocity: np.ndarray = None,
                     fitness: Optional[float] = None,
                     activation: Optional[float] = None,
                     type_tags: Optional[List[str]] = None) -> bool:
        """Update an element's properties"""
        with self._lock:
            if element_id not in self.elements:
                return False
                
            element = self.elements[element_id]
            
            # Update content if provided
            if content is not None:
                element.content = content
                element.complexity = self._calculate_complexity(content)
            
            # Update coordinates if provided
            if coordinates is not None:
                # Remove from old spatial index
                if element.coordinates is not None:
                    old_grid_cell = self._get_grid_cell(element.coordinates)
                    if old_grid_cell in self.spatial_index and element_id in self.spatial_index[old_grid_cell]:
                        self.spatial_index[old_grid_cell].remove(element_id)
                
                # Update coordinates
                element.update_coordinates(coordinates, self.manifold_type)
                
                # Add to new spatial index
                new_grid_cell = self._get_grid_cell(element.coordinates)
                if new_grid_cell not in self.spatial_index:
                    self.spatial_index[new_grid_cell] = []
                self.spatial_index[new_grid_cell].append(element_id)
            
            # Update velocity if provided
            if velocity is not None:
                element.update_velocity(velocity)
            
            # Update fitness if provided
            if fitness is not None:
                element.update_fitness(fitness)
            
            # Update activation if provided
            if activation is not None:
                element.update_activation(activation)
            
            # Update type tags if provided
            if type_tags is not None:
                # Remove from old type indices
                for tag in element.type_tags:
                    if tag in self.type_index and element_id in self.type_index[tag]:
                        self.type_index[tag].remove(element_id)
                
                # Update type tags
                element.type_tags = set(type_tags)
                
                # Add to new type indices
                for tag in element.type_tags:
                    if tag not in self.type_index:
                        self.type_index[tag] = set()
                    self.type_index[tag].add(element_id)
            
            # Update statistics
            self._update_stats()
            
            return True
    
    def connect_elements(self, source_id: str, target_id: str, strength: float) -> bool:
        """Create or update a connection between elements"""
        with self._lock:
            if source_id not in self.elements or target_id not in self.elements:
                return False
                
            # Add or update connection
            self.elements[source_id].add_connection(target_id, strength)
            
            return True
    
    def disconnect_elements(self, source_id: str, target_id: str) -> bool:
        """Remove a connection between elements"""
        with self._lock:
            if source_id not in self.elements:
                return False
                
            # Remove connection
            self.elements[source_id].remove_connection(target_id)
            
            return True
    
    def get_nearest_elements(self, coordinates: np.ndarray, 
                          k: int = 5, max_distance: float = None) -> List[Tuple[str, float]]:
        """Find k nearest elements to the given coordinates"""
        with self._lock:
            # Project query coordinates to manifold
            coordinates = self._project_to_manifold(coordinates)
            
            # Get nearby grid cells
            nearby_cells = self._get_nearby_grid_cells(coordinates)
            
            # Collect elements from nearby cells
            candidates = []
            for cell in nearby_cells:
                candidates.extend(self.spatial_index.get(cell, []))
            
            # Calculate distances
            distances = []
            for element_id in candidates:
                if element_id in self.elements:
                    element = self.elements[element_id]
                    if element.coordinates is not None:
                        dist = self._calculate_distance(coordinates, element.coordinates)
                        if max_distance is None or dist <= max_distance:
                            distances.append((element_id, dist))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Return k nearest
            return distances[:k]
    
    def get_elements_by_type(self, type_tag: str) -> List[str]:
        """Get elements with the specified type tag"""
        return list(self.type_index.get(type_tag, set()))
    
    def activate_element(self, element_id: str, activation: float = 0.5, 
                       spread: bool = False, spread_factor: float = 0.5) -> List[str]:
        """Activate an element and optionally spread to connected elements"""
        with self._lock:
            activated_elements = []
            
            if element_id not in self.elements:
                return activated_elements
                
            # Activate the target element
            element = self.elements[element_id]
            element.update_activation(activation)
            activated_elements.append(element_id)
            
            # Spread activation to connected elements
            if spread:
                for connected_id, connection_strength in element.connections.items():
                    if connected_id in self.elements:
                        # Calculate spread activation
                        spread_activation = activation * connection_strength * spread_factor
                        if spread_activation > 0.01:  # Only spread significant activation
                            self.elements[connected_id].update_activation(spread_activation)
                            activated_elements.append(connected_id)
            
            return activated_elements
    
    def decay_activations(self, decay_rate: float = 0.1) -> None:
        """Decay all element activations"""
        with self._lock:
            current_time = time.time()
            for element in self.elements.values():
                # Calculate time since last update
                dt = current_time - element.last_updated
                
                # Apply exponential decay
                decay_factor = math.exp(-decay_rate * dt)
                element.activation *= decay_factor
                element.last_updated = current_time
    
    def evolve(self, strategy: EvolutionStrategy, 
             iterations: int = 1, learning_rate: float = None) -> Dict[str, Any]:
        """
        Evolve the manifold using the specified evolutionary strategy.
        
        Args:
            strategy: Evolution strategy to apply (see EvolutionStrategy enum)
            iterations: Number of evolution cycles to perform (default: 1)
            learning_rate: Learning rate override (uses default if None)
            
        Returns:
            Dictionary containing evolution statistics and results:
            - strategy: Name of strategy used
            - iterations: Number of iterations performed
            - execution_time: Time taken in seconds
            - elements_affected: Number of elements modified
            - fitness_improvement: Change in average fitness
            - Additional strategy-specific metrics
            
        Note:
            Evolution is thread-safe and will not proceed if the manifold
            is empty. Different strategies have different effects:
            - GRADIENT_ASCENT: Gradual fitness-based improvements
            - EXPLORATION: Random search for new regions
            - CONSOLIDATION: Merge similar elements
            - PRUNING: Remove low-fitness elements
            - BIFURCATION: Create variations of successful elements
        """
        with self._lock:
            if not self.elements:
                return {"evolved": False, "reason": "No elements in manifold"}
                
            # Use provided learning rate or default
            lr = learning_rate if learning_rate is not None else self.evolution_params["learning_rate"]
            
            # Statistics
            stats = {
                "strategy": strategy.name,
                "iterations": iterations,
                "learning_rate": lr,
                "start_time": time.time(),
                "elements_affected": 0,
                "fitness_improvement": 0.0,
                "structural_changes": 0
            }
            
            # Track initial average fitness
            initial_fitness = self._calculate_average_fitness()
            
            # Apply the evolution strategy
            if strategy == EvolutionStrategy.GRADIENT_ASCENT:
                result = self._evolve_gradient_ascent(iterations, lr)
            elif strategy == EvolutionStrategy.EXPLORATION:
                result = self._evolve_exploration(iterations, lr)
            elif strategy == EvolutionStrategy.CONSOLIDATION:
                result = self._evolve_consolidation(iterations, lr)
            elif strategy == EvolutionStrategy.PRUNING:
                result = self._evolve_pruning(iterations, lr)
            elif strategy == EvolutionStrategy.BIFURCATION:
                result = self._evolve_bifurcation(iterations, lr)
            elif strategy == EvolutionStrategy.DIMENSIONALITY:
                result = self._evolve_dimensionality(iterations, lr)
            elif strategy == EvolutionStrategy.QUANTUM_LEAP:
                result = self._evolve_quantum_leap(iterations, lr)
            else:
                # Default to gradient ascent
                result = self._evolve_gradient_ascent(iterations, lr)
            
            # Update manifold properties
            self._update_manifold_properties()
            
            # Update statistics
            self.stats["evolution_count"] += 1
            self.stats["last_evolved"] = time.time()
            self._update_stats()
            
            # Calculate fitness improvement
            final_fitness = self._calculate_average_fitness()
            stats["fitness_improvement"] = final_fitness - initial_fitness
            
            # Add results from specific evolution method
            stats.update(result)
            
            # Calculate execution time
            stats["execution_time"] = time.time() - stats["start_time"]
            
            return stats
    
    def process(self, process_type: CognitiveProcess, 
              inputs: List[str], 
              params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply a cognitive process to manifold elements.
        
        Args:
            process_type: Type of cognitive process to apply
            inputs: List of element IDs to process
            params: Optional parameters for the process
            
        Returns:
            Dictionary containing process results:
            - success: Whether the process completed successfully
            - process_type: Name of the applied process
            - input_elements: List of input element IDs
            - result: Process-specific output (varies by process type)
            - timestamp: When the process was executed
            
        Supported processes:
            - ABSTRACTION: Extract general patterns from specific instances
            - ASSOCIATION: Find conceptually related elements
            - INFERENCE: Draw logical conclusions from premises
            - PREDICTION: Forecast future states or outcomes
            - ANALOGY: Find structural similarities between domains
            - SYNTHESIS: Combine multiple concepts into new ideas
            - CATEGORIZATION: Group concepts into meaningful categories
            
        Note:
            Process parameters can customize behavior:
            - activation_threshold: Minimum activation for consideration
            - output_count: Maximum number of results to return
            - creativity: Balance between literal and creative processing
            - depth: How many relationship hops to consider
        """
        with self._lock:
            if not inputs:
                return {"success": False, "reason": "No input elements provided"}
                
            # Validate inputs
            valid_inputs = [input_id for input_id in inputs if input_id in self.elements]
            if not valid_inputs:
                return {"success": False, "reason": "No valid input elements"}
                
            # Default parameters
            process_params = {
                "activation_threshold": 0.3,
                "output_count": 3,
                "creativity": 0.5,
                "depth": 2
            }
            
            # Update with provided parameters
            if params:
                process_params.update(params)
            
            # Apply the cognitive process
            if process_type == CognitiveProcess.ABSTRACTION:
                result = self._process_abstraction(valid_inputs, process_params)
            elif process_type == CognitiveProcess.ASSOCIATION:
                result = self._process_association(valid_inputs, process_params)
            elif process_type == CognitiveProcess.INFERENCE:
                result = self._process_inference(valid_inputs, process_params)
            elif process_type == CognitiveProcess.ANALOGY:
                result = self._process_analogy(valid_inputs, process_params)
            elif process_type == CognitiveProcess.SYNTHESIS:
                result = self._process_synthesis(valid_inputs, process_params)
            else:
                # Default to association
                result = self._process_association(valid_inputs, process_params)
            
            # Add basic process info to result
            result.update({
                "process_type": process_type.name,
                "input_elements": valid_inputs,
                "timestamp": time.time()
            })
            
            return result
    
    def get_manifold_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the manifold"""
        with self._lock:
            # Basic statistics
            stats = {
                "element_count": len(self.elements),
                "creation_time": self.stats["creation_time"],
                "age_seconds": time.time() - self.stats["creation_time"],
                "evolution_count": self.stats["evolution_count"],
                "last_evolved": self.stats["last_evolved"],
                "dimensions": self.dimensions,
                "manifold_type": self.manifold_type.name,
            }
            
            # Calculate activation statistics
            activations = [e.activation for e in self.elements.values()]
            if activations:
                stats["activation"] = {
                    "mean": np.mean(activations),
                    "max": max(activations),
                    "active_elements": sum(1 for a in activations if a > 0.1)
                }
            
            # Calculate fitness statistics
            fitness_values = [e.fitness for e in self.elements.values()]
            if fitness_values:
                stats["fitness"] = {
                    "mean": np.mean(fitness_values),
                    "min": min(fitness_values),
                    "max": max(fitness_values)
                }
            
            # Calculate complexity statistics
            complexity_values = [e.complexity for e in self.elements.values()]
            if complexity_values:
                stats["complexity"] = {
                    "mean": np.mean(complexity_values),
                    "min": min(complexity_values),
                    "max": max(complexity_values)
                }
            
            # Count connections
            connection_count = sum(len(e.connections) for e in self.elements.values())
            stats["connection_count"] = connection_count
            
            if self.elements:
                stats["avg_connections_per_element"] = connection_count / len(self.elements)
            
            # Type tag distribution
            stats["type_tag_counts"] = {tag: len(elements) for tag, elements in self.type_index.items()}
            
            # Manifold properties
            stats["manifold_properties"] = self.manifold_properties.copy()
            
            # Evolution parameters
            stats["evolution_params"] = self.evolution_params.copy()
            
            return stats
    
    def save_to_file(self, filepath: str) -> bool:
        """Save the manifold to a file"""
        try:
            # Create a serializable representation
            data = {
                "dimensions": self.dimensions,
                "manifold_type": self.manifold_type.name,
                "elements": {},
                "stats": self.stats,
                "manifold_properties": self.manifold_properties,
                "evolution_params": self.evolution_params
            }
            
            # Add elements
            for element_id, element in self.elements.items():
                # Convert numpy arrays to lists
                element_data = {
                    "element_id": element.element_id,
                    "coordinates": element.coordinates.tolist() if element.coordinates is not None else None,
                    "velocity": element.velocity.tolist() if element.velocity is not None else None,
                    "content": element.content,
                    "creation_time": element.creation_time,
                    "last_updated": element.last_updated,
                    "fitness": element.fitness,
                    "stability": element.stability,
                    "complexity": element.complexity,
                    "activation": element.activation,
                    "type_tags": list(element.type_tags),
                    "connections": element.connections,
                    "metadata": element.metadata
                }
                data["elements"][element_id] = element_data
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving manifold: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CognitiveManifold':
        """Load a manifold from a file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Create manifold
            manifold = cls(
                dimensions=data["dimensions"],
                manifold_type=ManifoldType[data["manifold_type"]]
            )
            
            # Restore stats and properties
            manifold.stats = data["stats"]
            manifold.manifold_properties = data["manifold_properties"]
            manifold.evolution_params = data["evolution_params"]
            
            # Load elements
            for element_id, element_data in data["elements"].items():
                # Convert lists back to numpy arrays
                coordinates = None
                if element_data["coordinates"] is not None:
                    coordinates = np.array(element_data["coordinates"])
                    
                velocity = None
                if element_data["velocity"] is not None:
                    velocity = np.array(element_data["velocity"])
                    
                # Create element
                element = CognitiveElement(
                    element_id=element_data["element_id"],
                    coordinates=coordinates,
                    velocity=velocity,
                    content=element_data["content"],
                    creation_time=element_data["creation_time"],
                    last_updated=element_data["last_updated"],
                    fitness=element_data["fitness"],
                    stability=element_data["stability"],
                    complexity=element_data["complexity"],
                    activation=element_data["activation"],
                    type_tags=set(element_data["type_tags"]),
                    connections=element_data["connections"],
                    metadata=element_data["metadata"]
                )
                
                # Add to manifold
                manifold.elements[element_id] = element
                
                # Add to indices
                if coordinates is not None:
                    grid_cell = manifold._get_grid_cell(coordinates)
                    if grid_cell not in manifold.spatial_index:
                        manifold.spatial_index[grid_cell] = []
                    manifold.spatial_index[grid_cell].append(element_id)
                
                for tag in element.type_tags:
                    if tag not in manifold.type_index:
                        manifold.type_index[tag] = set()
                    manifold.type_index[tag].add(element_id)
            
            return manifold
            
        except Exception as e:
            print(f"Error loading manifold: {e}")
            return cls()  # Return empty manifold
    
    def _generate_random_coordinates(self) -> np.ndarray:
        """Generate random coordinates appropriate for the manifold type"""
        if self.manifold_type == ManifoldType.EUCLIDEAN:
            # Uniform random in [-1, 1]
            return np.random.uniform(-1, 1, self.dimensions)
            
        elif self.manifold_type == ManifoldType.HYPERBOLIC:
            # Random point in Poincaré disk model
            # Generate random direction
            direction = np.random.normal(0, 1, self.dimensions)
            direction = direction / np.linalg.norm(direction)
            
            # Generate random radius with appropriate distribution for hyperbolic space
            # (more points near the boundary)
            r = random.random() ** 0.5  # Square root to get appropriate distribution
            
            return direction * r * 0.99  # Stay within unit disk
            
        elif self.manifold_type == ManifoldType.SPHERICAL:
            # Random point on unit sphere
            coords = np.random.normal(0, 1, self.dimensions)
            return coords / np.linalg.norm(coords)
            
        elif self.manifold_type == ManifoldType.TOROIDAL:
            # Random point on torus (simplified as point in unit square)
            return np.random.random(self.dimensions)
            
        else:
            # Default to Euclidean
            return np.random.uniform(-1, 1, self.dimensions)
    
    def _project_to_manifold(self, coordinates: np.ndarray) -> np.ndarray:
        """Project coordinates to the manifold"""
        if coordinates is None:
            return None
            
        if self.manifold_type == ManifoldType.EUCLIDEAN:
            # No projection needed
            return coordinates
            
        elif self.manifold_type == ManifoldType.HYPERBOLIC:
            # Project to Poincaré disk model of hyperbolic space
            norm = np.linalg.norm(coordinates)
            if norm >= 1.0:
                # Project to unit disk
                return coordinates / norm * 0.99
            return coordinates
            
        elif self.manifold_type == ManifoldType.SPHERICAL:
            # Project to unit sphere
            norm = np.linalg.norm(coordinates)
            if norm > 0:
                return coordinates / norm
            return coordinates
            
        elif self.manifold_type == ManifoldType.TOROIDAL:
            # Project to torus (simplified as wrapping around unit square)
            return coordinates % 1.0
            
        elif self.manifold_type == ManifoldType.MIXED_CURVATURE:
            # Mixed curvature manifold - different projection for different dimensions
            # Project first half of dimensions to hyperbolic space
            # and second half to spherical space
            mid = self.dimensions // 2
            
            result = np.copy(coordinates)
            
            # Project first half to hyperbolic space
            first_half = result[:mid]
            norm1 = np.linalg.norm(first_half)
            if norm1 >= 1.0:
                result[:mid] = first_half / norm1 * 0.99
                
            # Project second half to spherical space
            second_half = result[mid:]
            norm2 = np.linalg.norm(second_half)
            if norm2 > 0:
                result[mid:] = second_half / norm2
                
            return result
            
        elif self.manifold_type == ManifoldType.EMERGENT:
            # For emergent manifolds, adapt projection based on density
            # First, check if we have enough elements to define the manifold
            if len(self.elements) < 10:
                # Not enough elements, use Euclidean
                return coordinates
                
            # Otherwise, find the nearest neighbors and adapt to their local structure
            nearest = self.get_nearest_elements(coordinates, k=5, max_distance=None)
            if not nearest:
                return coordinates
                
            # Calculate local curvature based on neighbor configuration
            # (this is a simplified approximation)
            neighbor_coords = [self.elements[nid].coordinates for nid, _ in nearest 
                            if self.elements[nid].coordinates is not None]
            
            if len(neighbor_coords) < 3:
                return coordinates
            
            # Calculate the centroid
            centroid = np.mean(neighbor_coords, axis=0)
            
            # Calculate average distance to centroid
            avg_dist = np.mean([np.linalg.norm(nc - centroid) for nc in neighbor_coords])
            
            # Project based on local curvature estimate
            dist_to_centroid = np.linalg.norm(coordinates - centroid)
            
            if dist_to_centroid > 2 * avg_dist:
                # Far from cluster, pull closer
                direction = (centroid - coordinates) / dist_to_centroid
                coordinates = coordinates + direction * (dist_to_centroid - 2 * avg_dist) * 0.5
            
            return coordinates
            
        else:
            # Default to Euclidean for other types
            return coordinates
    
    def _calculate_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate distance between two points according to manifold geometry"""
        if coords1 is None or coords2 is None:
            return float('inf')
            
        if self.manifold_type == ManifoldType.EUCLIDEAN:
            # Euclidean distance
            return np.linalg.norm(coords1 - coords2)
            
        elif self.manifold_type == ManifoldType.HYPERBOLIC:
            # Distance in Poincaré disk model
            # This is a simplified approximation of hyperbolic distance
            # For true hyperbolic distance, we'd need more complex formula
            
            # Calculate Euclidean distance
            euclidean_dist = np.linalg.norm(coords1 - coords2)
            
            # Adjust based on position in the disk
            norm1 = np.linalg.norm(coords1)
            norm2 = np.linalg.norm(coords2)
            
            # Points closer to boundary are actually farther apart
            boundary_factor = 1.0 / ((1.0 - norm1**2) * (1.0 - norm2**2) + 1e-8)
            
            return euclidean_dist * boundary_factor
            
        elif self.manifold_type == ManifoldType.SPHERICAL:
            # Distance on unit sphere (arc length)
            # Cosine of angle between vectors
            cos_angle = np.dot(coords1, coords2) / (np.linalg.norm(coords1) * np.linalg.norm(coords2))
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
            
            # Arc length = angle * radius (radius = 1)
            return np.arccos(cos_angle)
            
        elif self.manifold_type == ManifoldType.TOROIDAL:
            # Toroidal distance (wrap around)
            # Calculate distance along each dimension with wrapping
            delta = np.abs(coords1 - coords2)
            wrapped_delta = np.minimum(delta, 1.0 - delta)  # Wrap around if shorter
            
            # Euclidean distance with wrapped coordinates
            return np.linalg.norm(wrapped_delta)
            
        elif self.manifold_type == ManifoldType.MIXED_CURVATURE:
            # Mixed distance - different metrics for different dimensions
            mid = self.dimensions // 2
            
            # Hyperbolic distance for first half
            d1 = self._calculate_distance(coords1[:mid], coords2[:mid])
            
            # Spherical distance for second half
            d2 = self._calculate_distance(coords1[mid:], coords2[mid:])
            
            # Combined distance
            return d1 + d2
            
        else:
            # Default to Euclidean distance
            return np.linalg.norm(coords1 - coords2)
    
    def _get_grid_cell(self, coordinates: np.ndarray) -> str:
        """Convert coordinates to a grid cell identifier for spatial indexing"""
        if coordinates is None:
            return "null"
            
        # For manifolds with bounded coordinates, use different grid resolution
        if self.manifold_type in [ManifoldType.HYPERBOLIC, ManifoldType.SPHERICAL]:
            # Finer resolution for bounded manifolds
            grid_resolution = 0.1
        else:
            # Coarser resolution for unbounded manifolds
            grid_resolution = 0.2
            
        # Discretize coordinates
        grid_coords = tuple(int(c / grid_resolution) for c in coordinates)
        
        # Convert to string
        return ";".join(map(str, grid_coords))
    
    def _get_nearby_grid_cells(self, coordinates: np.ndarray, radius: int = 1) -> List[str]:
        """Get grid cells near the given coordinates"""
        if coordinates is None:
            return []
            
        # Get base grid cell
        base_cell = self._get_grid_cell(coordinates)
        
        # Parse base cell coordinates
        try:
            base_coords = tuple(map(int, base_cell.split(";")))
        except (ValueError, AttributeError):
            return [base_cell]  # Return just the base cell if parsing fails
            
        # Generate nearby cells
        nearby_cells = []
        
        # Generate all combinations of offsets within radius
        for offsets in self._generate_offsets(radius, len(base_coords)):
            # Apply offsets to base coordinates
            cell_coords = tuple(b + o for b, o in zip(base_coords, offsets))
            cell = ";".join(map(str, cell_coords))
            nearby_cells.append(cell)
            
        return nearby_cells
    
    def _generate_offsets(self, radius: int, dimensions: int) -> List[Tuple[int, ...]]:
        """Generate all offset combinations within radius"""
        if dimensions == 0:
            return [()]
            
        result = []
        for offset in range(-radius, radius + 1):
            for sub_offsets in self._generate_offsets(radius, dimensions - 1):
                result.append((offset,) + sub_offsets)
                
        return result
    
    def _calculate_complexity(self, content: Any) -> float:
        """
        Calculate the complexity measure of cognitive content.
        
        Args:
            content: The content to analyze (any type)
            
        Returns:
            Complexity score in range [0.0, 1.0]
            
        Note:
            Complexity is estimated based on content type and structure:
            - Numbers: Low complexity (0.1)
            - Strings: Based on length and character diversity
            - Lists/Tuples: Based on size and element diversity
            - Dictionaries: Based on size and nested structure
            - Objects: Based on attribute count
            
            Higher complexity indicates more sophisticated or detailed content.
        """
        if content is None:
            return 0.0
            
        # Different complexity calculations based on content type
        if isinstance(content, (int, float)):
            # Simple numeric value
            return 0.1
            
        elif isinstance(content, str):
            # Text complexity based on length and character diversity
            if not content:
                return 0.0
                
            length_factor = min(1.0, len(content) / 1000)
            unique_chars = len(set(content)) / len(content)
            
            return length_factor * (0.3 + 0.7 * unique_chars)
            
        elif isinstance(content, (list, tuple)):
            # List complexity based on length and element diversity
            if not content:
                return 0.0
                
            length_factor = min(1.0, len(content) / 100)
            
            # Try to estimate element diversity
            try:
                # Use string representation for comparison
                unique_elements = len(set(str(x) for x in content)) / len(content)
                return length_factor * (0.3 + 0.7 * unique_elements)
            except:
                return length_factor * 0.5
                
        elif isinstance(content, dict):
            # Dictionary complexity based on size and structure
            if not content:
                return 0.0
                
            size_factor = min(1.0, len(content) / 50)
            
            # Calculate average value complexity
            try:
                value_complexity = 0.0
                for value in content.values():
                    if isinstance(value, (list, dict, tuple)):
                        value_complexity += 0.2  # Nested structures add complexity
                    elif isinstance(value, str) and len(value) > 20:
                        value_complexity += 0.1  # Longer strings add complexity
                
                value_complexity = min(1.0, value_complexity)
                
                return size_factor * (0.4 + 0.6 * value_complexity)
            except:
                return size_factor * 0.5
                
        elif hasattr(content, "__dict__"):
            # Object with attributes
            try:
                # Use dictionary size as proxy for complexity
                attrs = vars(content)
                return min(1.0, len(attrs) / 20)
            except:
                return 0.3
                
        else:
            # Default complexity
            return 0.2
    
    def _calculate_average_fitness(self) -> float:
        """Calculate average fitness of all elements"""
        if not self.elements:
            return 0.0
            
        total_fitness = sum(element.fitness for element in self.elements.values())
        return total_fitness / len(self.elements)
    
    def _update_stats(self) -> None:
        """Update manifold statistics"""
        if not self.elements:
            self.stats["total_fitness"] = 0.0
            self.stats["avg_complexity"] = 0.0
            self.stats["avg_stability"] = 0.0
            return
            
        # Update total fitness
        self.stats["total_fitness"] = sum(element.fitness for element in self.elements.values())
        
        # Update average complexity
        self.stats["avg_complexity"] = sum(element.complexity for element in self.elements.values()) / len(self.elements)
        
        # Update average stability
        self.stats["avg_stability"] = sum(element.stability for element in self.elements.values()) / len(self.elements)
    
    def _update_manifold_properties(self) -> None:
        """Update properties of the manifold"""
        if len(self.elements) < 5:
            # Not enough elements to calculate meaningful properties
            return
            
        # Sample elements for analysis (up to 100 elements)
        sample_size = min(100, len(self.elements))
        sampled_elements = random.sample(list(self.elements.values()), sample_size)
        
        # Extract coordinates
        coords = [e.coordinates for e in sampled_elements if e.coordinates is not None]
        if len(coords) < 5:
            return
            
        coords_array = np.vstack(coords)
        
        # Estimate effective dimensionality using PCA
        try:
            # Center the data
            centered = coords_array - np.mean(coords_array, axis=0)
            
            # Calculate covariance matrix
            cov = centered.T @ centered
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            
            # Sort in descending order
            eigenvalues = np.flip(np.sort(eigenvalues))
            
            # Calculate explained variance ratio
            total_variance = np.sum(eigenvalues)
            if total_variance > 0:
                explained_variance_ratio = eigenvalues / total_variance
                
                # Effective dimensionality: number of dimensions needed to explain 90% of variance
                cumulative_variance = np.cumsum(explained_variance_ratio)
                effective_dim = np.searchsorted(cumulative_variance, 0.9) + 1
                
                # Update manifold properties
                self.manifold_properties["dimensionality"] = effective_dim
        except:
            pass
        
        # Estimate manifold curvature
        try:
            # Calculate pairwise distances
            dists = distance_matrix(coords_array, coords_array)
            np.fill_diagonal(dists, np.inf)  # Ignore self-distances
            
            # For each point, find the nearest neighbors
            k = min(10, len(coords_array) - 1)
            nearest_indices = np.argsort(dists, axis=1)[:, :k]
            
            # Estimate local curvature for each point
            curvatures = []
            for i, idx in enumerate(nearest_indices):
                # Get neighbors
                neighbors = coords_array[idx]
                
                # Calculate centroid
                centroid = np.mean(neighbors, axis=0)
                
                # Calculate average distance to centroid
                avg_dist = np.mean(np.linalg.norm(neighbors - centroid, axis=1))
                
                # Calculate average pairwise distance
                pair_dists = []
                for j in range(len(neighbors)):
                    for l in range(j+1, len(neighbors)):
                        pair_dists.append(np.linalg.norm(neighbors[j] - neighbors[l]))
                avg_pair_dist = np.mean(pair_dists) if pair_dists else 0
                
                # Calculate local curvature estimate
                # Positive curvature: points are closer to each other than to centroid
                # Negative curvature: points are farther from each other than from centroid
                if avg_dist > 0:
                    local_curvature = (avg_pair_dist / (2 * avg_dist)) - 1
                    curvatures.append(local_curvature)
            
            # Overall curvature is average of local curvatures
            if curvatures:
                avg_curvature = np.mean(curvatures)
                self.manifold_properties["curvature"] = avg_curvature
                self.stats["curvature"] = avg_curvature
        except:
            pass
        
        # Estimate manifold diameter
        try:
            # Use max of pairwise distances
            diameter = np.max(dists[dists != np.inf])
            self.manifold_properties["diameter"] = diameter
        except:
            pass
        
        # Estimate manifold volume (proportional to spread of points)
        try:
            # Calculate volume of bounding box as proxy
            # Calculate coordinate ranges
            ranges = np.ptp(coords_array, axis=0)
            volume = np.prod(ranges)
            self.manifold_properties["volume"] = volume
        except:
            pass
    
    def _evolve_gradient_ascent(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold using gradient ascent"""
        elements_affected = 0
        total_movement = 0.0
        
        for _ in range(iterations):
            for element_id, element in self.elements.items():
                # Skip if no coordinates
                if element.coordinates is None:
                    continue
                    
                # Find neighbors
                neighbors = self.get_nearest_elements(element.coordinates, k=5)
                if not neighbors:
                    continue
                    
                # Calculate gradient based on neighbor fitness
                gradient = np.zeros_like(element.coordinates)
                
                for neighbor_id, distance in neighbors:
                    if neighbor_id == element_id:
                        continue
                        
                    neighbor = self.elements.get(neighbor_id)
                    if neighbor is None or neighbor.coordinates is None:
                        continue
                    
                    # Calculate fitness gradient
                    fitness_diff = neighbor.fitness - element.fitness
                    
                    # Direction from element to neighbor
                    if distance > 0:
                        direction = (neighbor.coordinates - element.coordinates) / distance
                    else:
                        continue
                    
                    # Add to gradient (move toward higher fitness)
                    gradient += direction * fitness_diff
                
                # Apply gradient
                if np.any(gradient):
                    # Normalize gradient
                    gradient_norm = np.linalg.norm(gradient)
                    if gradient_norm > 0:
                        gradient = gradient / gradient_norm
                    
                    # Apply learning rate
                    update = gradient * learning_rate
                    
                    # Calculate movement magnitude
                    movement = np.linalg.norm(update)
                    total_movement += movement
                    
                    # Update coordinates
                    new_coords = element.coordinates + update
                    self.update_element(element_id, coordinates=new_coords)
                    
                    elements_affected += 1
        
        return {
            "elements_affected": elements_affected,
            "total_movement": total_movement,
            "avg_movement": total_movement / max(1, elements_affected)
        }
    
    def _evolve_exploration(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold by exploring new regions"""
        elements_affected = 0
        new_elements = []
        
        # Increase temperature for exploration
        temperature = self.evolution_params["temperature"] * 2.0
        
        for _ in range(iterations):
            # Add random elements
            for _ in range(max(1, len(self.elements) // 10)):
                # Generate random coordinates
                coords = self._generate_random_coordinates()
                
                # Generate random content
                content = self._generate_random_content()
                
                # Add element to manifold
                element_id = self.add_element(content, coords)
                new_elements.append(element_id)
            
            # Move existing elements randomly
            for element_id, element in list(self.elements.items()):
                # Skip with some probability
                if random.random() > 0.3:
                    continue
                    
                # Skip if no coordinates
                if element.coordinates is None:
                    continue
                    
                # Random movement vector
                random_vector = np.random.normal(0, 1, self.dimensions)
                
                # Scale by temperature and learning rate
                update = random_vector * temperature * learning_rate
                
                # Update coordinates
                new_coords = element.coordinates + update
                self.update_element(element_id, coordinates=new_coords)
                
                elements_affected += 1
        
        return {
            "elements_affected": elements_affected,
            "new_elements": new_elements,
            "total_new_elements": len(new_elements)
        }
    
    def _evolve_consolidation(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold by consolidating similar elements"""
        merged_elements = 0
        strengthened_connections = 0
        
        for _ in range(iterations):
            # Find similar elements
            similar_pairs = []
            
            # Check a sample of elements
            sample_size = min(50, len(self.elements))
            sampled_ids = random.sample(list(self.elements.keys()), sample_size)
            
            for i, element_id1 in enumerate(sampled_ids):
                element1 = self.elements[element_id1]
                if element1.coordinates is None:
                    continue
                    
                for element_id2 in sampled_ids[i+1:]:
                    element2 = self.elements[element_id2]
                    if element2.coordinates is None:
                        continue
                    
                    # Calculate similarity
                    distance = self._calculate_distance(element1.coordinates, element2.coordinates)
                    content_similarity = self._calculate_content_similarity(element1.content, element2.content)
                    
                    # Combined similarity
                    similarity = content_similarity * (1.0 / (1.0 + distance))
                    
                    if similarity > 0.7:
                        similar_pairs.append((element_id1, element_id2, similarity))
            
            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Process similar pairs
            for element_id1, element_id2, similarity in similar_pairs:
                # Skip if either element has been removed
                if element_id1 not in self.elements or element_id2 not in self.elements:
                    continue
                    
                element1 = self.elements[element_id1]
                element2 = self.elements[element_id2]
                
                if similarity > 0.9:
                    # Very similar - merge elements
                    # Keep the one with higher fitness
                    if element1.fitness >= element2.fitness:
                        keep_id, remove_id = element_id1, element_id2
                    else:
                        keep_id, remove_id = element_id2, element_id1
                        
                    keep_element = self.elements[keep_id]
                    remove_element = self.elements[remove_id]
                    
                    # Merge connections
                    for conn_id, strength in remove_element.connections.items():
                        if conn_id != keep_id and conn_id in self.elements:
                            # Add or update connection
                            current = keep_element.connections.get(conn_id, 0)
                            keep_element.add_connection(conn_id, max(current, strength))
                            
                            # Add reverse connection
                            self.connect_elements(conn_id, keep_id, strength)
                    
                    # Update fitness to average
                    combined_fitness = (keep_element.fitness + remove_element.fitness) / 2
                    keep_element.update_fitness(combined_fitness)
                    
                    # Remove the redundant element
                    self.remove_element(remove_id)
                    merged_elements += 1
                    
                else:
                    # Somewhat similar - create or strengthen connection
                    current_strength = element1.connections.get(element_id2, 0)
                    new_strength = max(current_strength, similarity)
                    
                    # Create bidirectional connection
                    self.connect_elements(element_id1, element_id2, new_strength)
                    self.connect_elements(element_id2, element_id1, new_strength)
                    
                    strengthened_connections += 1
        
        return {
            "merged_elements": merged_elements,
            "strengthened_connections": strengthened_connections
        }
    
    def _evolve_pruning(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold by pruning low-fitness elements"""
        removed_elements = 0
        elements_affected = 0
        
        for _ in range(iterations):
            # Identify low-fitness elements
            candidates = []
            
            for element_id, element in self.elements.items():
                # Check if element is low fitness and old enough
                if element.fitness < 0.3 and element.get_age() > 60:
                    candidates.append((element_id, element.fitness))
            
            # Sort by fitness (lowest first)
            candidates.sort(key=lambda x: x[1])
            
            # Remove a fraction of the lowest fitness elements
            target_remove = max(1, int(len(candidates) * 0.2))
            
            for i, (element_id, _) in enumerate(candidates):
                if i >= target_remove:
                    break
                    
                if element_id in self.elements:  # Double-check element still exists
                    self.remove_element(element_id)
                    removed_elements += 1
            
            # Strengthen remaining elements
            for element_id, element in self.elements.items():
                # Increase fitness slightly
                element.update_fitness(element.fitness + 0.05)
                elements_affected += 1
        
        return {
            "removed_elements": removed_elements,
            "elements_affected": elements_affected
        }
    
    def _evolve_bifurcation(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold by creating bifurcations from high-fitness elements"""
        new_elements = []
        split_events = 0
        
        for _ in range(iterations):
            # Find high-fitness elements
            candidates = []
            
            for element_id, element in self.elements.items():
                if element.fitness > 0.7 and element.coordinates is not None:
                    candidates.append(element_id)
            
            # Bifurcate a random subset
            sample_size = min(len(candidates), max(1, int(len(candidates) * 0.2)))
            if sample_size > 0:
                selected = random.sample(candidates, sample_size)
                
                for element_id in selected:
                    # Element may have been removed during the loop
                    if element_id not in self.elements:
                        continue
                        
                    element = self.elements[element_id]
                    
                    # Create variations
                    num_variations = random.randint(1, 3)
                    
                    for _ in range(num_variations):
                        # Create slight coordinate variation
                        variation_scale = learning_rate * 0.5
                        variation = np.random.normal(0, variation_scale, self.dimensions)
                        new_coords = element.coordinates + variation
                        
                        # Create content variation
                        new_content = self._mutate_content(element.content)
                        
                        # Add new element
                        new_id = self.add_element(
                            content=new_content,
                            coordinates=new_coords,
                            type_tags=list(element.type_tags),
                            fitness=element.fitness * 0.9  # Slightly lower initial fitness
                        )
                        
                        # Connect to parent
                        self.connect_elements(element_id, new_id, 0.8)
                        self.connect_elements(new_id, element_id, 0.8)
                        
                        new_elements.append(new_id)
                    
                    split_events += 1
        
        return {
            "new_elements": new_elements,
            "split_events": split_events,
            "total_new_elements": len(new_elements)
        }
    
    def _evolve_dimensionality(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold by changing its dimensionality"""
        elements_affected = 0
        old_dimensions = self.dimensions
        
        # Determine whether to increase or decrease dimensionality
        if random.random() < 0.5:
            # Increase dimensions
            new_dimensions = self.dimensions + 1
        else:
            # Decrease dimensions (minimum 2)
            new_dimensions = max(2, self.dimensions - 1)
        
        # Update manifold dimensions
        self.dimensions = new_dimensions
        
        # Update element coordinates
        for element_id, element in self.elements.items():
            if element.coordinates is None:
                continue
                
            old_coords = element.coordinates
            
            if new_dimensions > old_dimensions:
                # Add a new dimension with random values
                new_coord = np.random.normal(0, 0.1)
                new_coords = np.append(old_coords, new_coord)
                
            else:
                # Remove last dimension
                new_coords = old_coords[:-1]
                
            # Update element
            self.update_element(element_id, coordinates=new_coords)
            elements_affected += 1
        
        # Update manifold properties
        self.manifold_properties["dimensionality"] = new_dimensions
        self._update_manifold_properties()
        
        return {
            "old_dimensions": old_dimensions,
            "new_dimensions": new_dimensions,
            "elements_affected": elements_affected
        }
    
    def _evolve_quantum_leap(self, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """Evolve the manifold with discontinuous quantum leaps"""
        elements_affected = 0
        major_changes = 0
        
        # Higher learning rate for quantum leaps
        quantum_lr = learning_rate * 10.0
        
        for _ in range(iterations):
            # Select random elements for quantum leaps
            sample_size = max(1, int(len(self.elements) * 0.1))
            sampled_ids = random.sample(list(self.elements.keys()), min(sample_size, len(self.elements)))
            
            for element_id in sampled_ids:
                element = self.elements.get(element_id)
                if element is None or element.coordinates is None:
                    continue
                
                # Generate quantum leap
                # 1. Random direction
                direction = np.random.normal(0, 1, self.dimensions)
                direction = direction / np.linalg.norm(direction)
                
                # 2. Random leap magnitude
                magnitude = np.random.exponential(quantum_lr)
                
                # Apply leap
                new_coords = element.coordinates + direction * magnitude
                
                # Update element
                self.update_element(element_id, coordinates=new_coords)
                elements_affected += 1
                
                # Major content change with some probability
                if random.random() < 0.3:
                    # Quantum mutation of content
                    new_content = self._quantum_mutate_content(element.content)
                    self.update_element(element_id, content=new_content)
                    major_changes += 1
        
        return {
            "elements_affected": elements_affected,
            "major_changes": major_changes
        }
    
    def _generate_random_content(self) -> Any:
        """Generate random content for exploration"""
        content_type = random.choice(["number", "string", "list", "dict"])
        
        if content_type == "number":
            return random.uniform(-10, 10)
            
        elif content_type == "string":
            length = random.randint(3, 15)
            chars = "abcdefghijklmnopqrstuvwxyz "
            return ''.join(random.choice(chars) for _ in range(length))
            
        elif content_type == "list":
            length = random.randint(2, 5)
            return [random.uniform(-1, 1) for _ in range(length)]
            
        else:  # dict
            keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
            size = random.randint(1, 3)
            selected_keys = random.sample(keys, size)
            return {key: random.uniform(-1, 1) for key in selected_keys}
    
    def _calculate_content_similarity(self, content1: Any, content2: Any) -> float:
        """Calculate similarity between two content items"""
        # Handle None values
        if content1 is None or content2 is None:
            return 0.0
            
        # Check if same type
        if type(content1) != type(content2):
            return 0.1  # Different types have low similarity
            
        # Calculate based on content type
        if isinstance(content1, (int, float)) and isinstance(content2, (int, float)):
            # Numeric similarity
            max_val = max(abs(content1), abs(content2), 1.0)
            diff = abs(content1 - content2) / max_val
            return max(0.0, 1.0 - diff)
            
        elif isinstance(content1, str) and isinstance(content2, str):
            # String similarity (simplified)
            if not content1 or not content2:
                return 0.0 if (content1 or content2) else 1.0
                
            # Length ratio
            len_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
            
            # Character overlap
            chars1 = set(content1.lower())
            chars2 = set(content2.lower())
            char_overlap = len(chars1.intersection(chars2)) / len(chars1.union(chars2))
            
            return (len_ratio * 0.3 + char_overlap * 0.7)
            
        elif isinstance(content1, (list, tuple)) and isinstance(content2, (list, tuple)):
            # List similarity
            if not content1 or not content2:
                return 0.0 if (content1 or content2) else 1.0
                
            # Length ratio
            len_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
            
            # Try to calculate average element similarity
            try:
                # Compare elements at same positions
                min_len = min(len(content1), len(content2))
                element_sims = []
                
                for i in range(min_len):
                    sim = self._calculate_content_similarity(content1[i], content2[i])
                    element_sims.append(sim)
                
                if element_sims:
                    avg_element_sim = sum(element_sims) / len(element_sims)
                    return (len_ratio * 0.3 + avg_element_sim * 0.7)
                else:
                    return len_ratio
            except:
                return len_ratio
                
        elif isinstance(content1, dict) and isinstance(content2, dict):
            # Dictionary similarity
            if not content1 or not content2:
                return 0.0 if (content1 or content2) else 1.0
                
            # Key overlap
            keys1 = set(content1.keys())
            keys2 = set(content2.keys())
            
            key_overlap = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            
            # Try to calculate value similarity for common keys
            common_keys = keys1.intersection(keys2)
            if common_keys:
                value_sims = []
                for key in common_keys:
                    sim = self._calculate_content_similarity(content1[key], content2[key])
                    value_sims.append(sim)
                
                avg_value_sim = sum(value_sims) / len(value_sims)
                return (key_overlap * 0.5 + avg_value_sim * 0.5)
            else:
                return key_overlap
                
        else:
            # Default similarity for other types
            # Compare string representations as fallback
            str1 = str(content1)
            str2 = str(content2)
            
            if str1 == str2:
                return 1.0
                
            # Length ratio
            len_ratio = min(len(str1), len(str2)) / max(len(str1), len(str2))
            
            # Character overlap
            chars1 = set(str1)
            chars2 = set(str2)
            char_overlap = len(chars1.intersection(chars2)) / len(chars1.union(chars2))
            
            return (len_ratio * 0.3 + char_overlap * 0.7) * 0.5  # Reduced confidence for fallback
    
    def _mutate_content(self, content: Any) -> Any:
        """Create a mutated version of content"""
        if content is None:
            return None
            
        # Different mutation strategies based on content type
        if isinstance(content, (int, float)):
            # Mutate by adding random value
            mutation_scale = abs(content) * 0.2 + 0.1
            return content + random.uniform(-mutation_scale, mutation_scale)
            
        elif isinstance(content, str):
            if not content:
                return ""
                
            # String mutations
            mutation_type = random.choice(["substitute", "insert", "delete", "swap"])
            
            chars = list(content)
            
            if mutation_type == "substitute" and chars:
                # Substitute random character
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice("abcdefghijklmnopqrstuvwxyz ")
                
            elif mutation_type == "insert" and len(content) < 100:
                # Insert random character
                pos = random.randint(0, len(chars))
                chars.insert(pos, random.choice("abcdefghijklmnopqrstuvwxyz "))
                
            elif mutation_type == "delete" and len(chars) > 1:
                # Delete random character
                pos = random.randint(0, len(chars) - 1)
                del chars[pos]
                
            elif mutation_type == "swap" and len(chars) > 1:
                # Swap two adjacent characters
                pos = random.randint(0, len(chars) - 2)
                chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            
            return ''.join(chars)
            
        elif isinstance(content, list):
            if not content:
                return []
                
            # List mutations
            result = content.copy()
            
            mutation_type = random.choice(["modify", "add", "remove", "reorder"])
            
            if mutation_type == "modify" and result:
                # Modify a random element
                pos = random.randint(0, len(result) - 1)
                result[pos] = self._mutate_content(result[pos])
                
            elif mutation_type == "add" and len(result) < 10:
                # Add a new element
                if result:
                    # Similar to existing elements
                    template = random.choice(result)
                    new_element = self._mutate_content(template)
                else:
                    # Random element
                    new_element = random.uniform(-1, 1)
                    
                result.append(new_element)
                
            elif mutation_type == "remove" and len(result) > 1:
                # Remove a random element
                pos = random.randint(0, len(result) - 1)
                del result[pos]
                
            elif mutation_type == "reorder" and len(result) > 1:
                # Swap two elements
                i, j = random.sample(range(len(result)), 2)
                result[i], result[j] = result[j], result[i]
            
            return result
            
        elif isinstance(content, dict):
            if not content:
                return {}
                
            # Dictionary mutations
            result = content.copy()
            
            mutation_type = random.choice(["modify", "add", "remove"])
            
            if mutation_type == "modify" and result:
                # Modify a random value
                key = random.choice(list(result.keys()))
                result[key] = self._mutate_content(result[key])
                
            elif mutation_type == "add" and len(result) < 10:
                # Add a new key-value pair
                keys = ["alpha", "beta", "gamma", "delta", "epsilon", 
                       "zeta", "eta", "theta", "iota", "kappa"]
                
                # Find an unused key
                unused_keys = [k for k in keys if k not in result]
                if unused_keys:
                    new_key = random.choice(unused_keys)
                else:
                    # Generate a random key
                    new_key = f"key_{random.randint(1, 100)}"
                
                # Generate value similar to existing values
                if result:
                    template = random.choice(list(result.values()))
                    new_value = self._mutate_content(template)
                else:
                    new_value = random.uniform(-1, 1)
                    
                result[new_key] = new_value
                
            elif mutation_type == "remove" and len(result) > 1:
                # Remove a random key
                key = random.choice(list(result.keys()))
                del result[key]
            
            return result
            
        else:
            # Default: return unchanged for unsupported types
            return content
    
    def _quantum_mutate_content(self, content: Any) -> Any:
        """Create a significantly mutated version of content (quantum leap)"""
        if content is None:
            return self._generate_random_content()
            
        # For quantum mutation, we make more dramatic changes
        
        if isinstance(content, (int, float)):
            # Major value shift
            if random.random() < 0.3:
                # Sign flip with scale change
                return -content * random.uniform(0.5, 2.0)
            else:
                # Large random shift
                return content + random.uniform(-abs(content) - 5, abs(content) + 5)
                
        elif isinstance(content, str):
            if not content:
                return self._generate_random_content()
                
            mutation_level = random.choice(["moderate", "major", "complete"])
            
            if mutation_level == "complete":
                # Complete replacement
                return self._generate_random_content()
                
            elif mutation_level == "major":
                # Major changes
                words = content.split()
                if len(words) > 2:
                    # Shuffle and replace some words
                    random.shuffle(words)
                    
                    # Replace some words
                    replacement_count = max(1, len(words) // 3)
                    for i in range(replacement_count):
                        if i < len(words):
                            words[i] = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") 
                                            for _ in range(random.randint(2, 7)))
                    
                    return ' '.join(words)
                else:
                    # Not enough words, replace completely
                    return self._generate_random_content()
                    
            else:  # moderate
                # Apply multiple minor mutations
                result = content
                mutations = random.randint(2, 5)
                
                for _ in range(mutations):
                    result = self._mutate_content(result)
                    
                return result
                
        elif isinstance(content, list):
            if not content:
                return self._generate_random_content()
                
            mutation_level = random.choice(["moderate", "major", "complete"])
            
            if mutation_level == "complete":
                # Complete replacement
                return self._generate_random_content()
                
            elif mutation_level == "major" and len(content) > 0:
                # Major restructuring
                result = []
                
                # Keep some elements, replace others
                for item in content:
                    if random.random() < 0.3:
                        # Keep but possibly mutate
                        if random.random() < 0.5:
                            result.append(self._mutate_content(item))
                        else:
                            result.append(item)
                    else:
                        # Replace with new content
                        if isinstance(item, (int, float)):
                            result.append(random.uniform(-10, 10))
                        else:
                            result.append(self._generate_random_content())
                
                # Possibly add new elements
                if random.random() < 0.5:
                    for _ in range(random.randint(1, 3)):
                        result.append(self._generate_random_content())
                
                # Shuffle
                random.shuffle(result)
                
                return result
                
            else:  # moderate
                # Apply multiple minor mutations
                result = content.copy()
                mutations = random.randint(2, 5)
                
                for _ in range(mutations):
                    result = self._mutate_content(result)
                    
                return result
                
        elif isinstance(content, dict):
            if not content:
                return self._generate_random_content()
                
            mutation_level = random.choice(["moderate", "major", "complete"])
            
            if mutation_level == "complete":
                # Complete replacement
                return self._generate_random_content()
                
            elif mutation_level == "major":
                # Major restructuring
                result = {}
                
                # Keep some key-value pairs, replace others
                for key, value in content.items():
                    if random.random() < 0.3:
                        # Keep key but possibly mutate value
                        if random.random() < 0.5:
                            result[key] = self._mutate_content(value)
                        else:
                            result[key] = value
                    else:
                        # New key
                        new_key = f"quantum_{key}"
                        result[new_key] = self._generate_random_content()
                
                return result
                
            else:  # moderate
                # Apply multiple minor mutations
                result = content.copy()
                mutations = random.randint(2, 5)
                
                for _ in range(mutations):
                    result = self._mutate_content(result)
                    
                return result
                
        else:
            # Default: generate new content
            return self._generate_random_content()
    
    def _process_abstraction(self, inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply abstraction process to extract higher-level concepts"""
        # Get input elements
        input_elements = [self.elements[input_id] for input_id in inputs if input_id in self.elements]
        
        if not input_elements:
            return {"success": False, "reason": "No valid input elements"}
            
        # Extract common features across inputs
        common_tags = set.intersection(*[e.type_tags for e in input_elements]) if input_elements else set()
        
        # Calculate centroid coordinates
        coords_list = [e.coordinates for e in input_elements if e.coordinates is not None]
        if coords_list:
            centroid = np.mean(coords_list, axis=0)
        else:
            centroid = None
            
        # Generate abstracted content
        if all(isinstance(e.content, (int, float)) for e in input_elements):
            # Numeric abstraction - use average
            abstracted_content = sum(e.content for e in input_elements) / len(input_elements)
            
        elif all(isinstance(e.content, str) for e in input_elements):
            # Text abstraction - find common words
            all_words = []
            for e in input_elements:
                if e.content:
                    words = e.content.lower().split()
                    all_words.extend(words)
                    
            # Count word frequencies
            word_counts = {}
            for word in all_words:
                if len(word) > 2:  # Ignore short words
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
            # Find most common words
            common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if common_words:
                abstracted_content = "Abstract: " + " ".join(word for word, _ in common_words)
            else:
                abstracted_content = "Abstract concept"
                
        elif all(isinstance(e.content, (list, tuple)) for e in input_elements):
            # List abstraction - find common elements
            all_elements = []
            for e in input_elements:
                all_elements.extend(e.content)
                
            # For simplicity, just take average if numeric
            if all(isinstance(x, (int, float)) for x in all_elements):
                abstracted_content = sum(all_elements) / len(all_elements)
            else:
                # Otherwise use first few elements
                abstracted_content = all_elements[:3]
                
        elif all(isinstance(e.content, dict) for e in input_elements):
            # Dictionary abstraction - merge common keys
            common_keys = set.intersection(*[set(e.content.keys()) for e in input_elements])
            
            abstracted_content = {}
            for key in common_keys:
                # Extract values for this key
                values = [e.content[key] for e in input_elements]
                
                if all(isinstance(v, (int, float)) for v in values):
                    # Average numeric values
                    abstracted_content[key] = sum(values) / len(values)
                else:
                    # Otherwise just use first value
                    abstracted_content[key] = values[0]
                    
        else:
            # Mixed type abstraction - create a summary
            abstracted_content = f"Abstract of {len(input_elements)} mixed elements"
            
        # Create the abstracted element
        abstract_id = self.add_element(
            content=abstracted_content,
            coordinates=centroid,
            type_tags=list(common_tags) + ["abstraction"],
            fitness=0.6  # Start with moderate fitness
        )
        
        # Connect to input elements
        for input_id in inputs:
            if input_id in self.elements:
                self.connect_elements(abstract_id, input_id, 0.7)
                self.connect_elements(input_id, abstract_id, 0.7)
        
        return {
            "success": True,
            "abstract_element": abstract_id,
            "common_tags": list(common_tags),
            "abstracted_content": abstracted_content
        }
    
    def _process_association(self, inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply association process to find related concepts"""
        if not inputs:
            return {"success": False, "reason": "No input elements provided"}
            
        # Get input elements
        input_elements = [self.elements[input_id] for input_id in inputs if input_id in self.elements]
        
        if not input_elements:
            return {"success": False, "reason": "No valid input elements"}
            
        # Activation threshold
        threshold = params.get("activation_threshold", 0.3)
        
        # Activate input elements
        for element in input_elements:
            element.update_activation(1.0)
            
        # Spread activation through connections
        activated_elements = set()
        for element in input_elements:
            for conn_id, strength in element.connections.items():
                if conn_id in self.elements and conn_id not in inputs:
                    activation = strength * element.activation
                    if activation >= threshold:
                        self.elements[conn_id].update_activation(activation)
                        activated_elements.add(conn_id)
        
        # Find elements with spatial proximity
        coords_list = [e.coordinates for e in input_elements if e.coordinates is not None]
        if coords_list:
            # Calculate centroid
            centroid = np.mean(coords_list, axis=0)
            
            # Find nearest elements
            neighbors = self.get_nearest_elements(centroid, k=params.get("output_count", 3))
            
            # Activate neighboring elements
            for neighbor_id, distance in neighbors:
                if neighbor_id not in inputs and neighbor_id in self.elements:
                    # Activate based on proximity
                    proximity_activation = max(0, 1.0 - distance * 0.5)
                    if proximity_activation >= threshold:
                        self.elements[neighbor_id].update_activation(proximity_activation)
                        activated_elements.add(neighbor_id)
        
        # Get highest activated elements
        associations = []
        for element_id in activated_elements:
            element = self.elements.get(element_id)
            if element and element.activation >= threshold:
                associations.append({
                    "element_id": element_id,
                    "activation": element.activation,
                    "content": element.content,
                    "tags": list(element.type_tags)
                })
        
        # Sort by activation (highest first)
        associations.sort(key=lambda x: x["activation"], reverse=True)
        
        # Limit to requested count
        output_count = params.get("output_count", 3)
        associations = associations[:output_count]
        
        return {
            "success": True,
            "associations": associations,
            "count": len(associations)
        }
    
    def _process_inference(self, inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply inference process to draw conclusions"""
        # Get input elements
        input_elements = [self.elements[input_id] for input_id in inputs if input_id in self.elements]
        
        if len(input_elements) < 2:
            return {"success": False, "reason": "Need at least 2 valid elements for inference"}
            
        # Extract input coordinates
        coords_list = [e.coordinates for e in input_elements if e.coordinates is not None]
        if len(coords_list) < 2:
            return {"success": False, "reason": "Need at least 2 elements with coordinates"}
            
        # Calculate vector between elements (simplified to first two elements)
        vector = coords_list[1] - coords_list[0]
        
        # Project vector to infer new position
        projection_scale = params.get("projection_scale", 1.0)
        inferred_coords = coords_list[1] + vector * projection_scale
        
        # Generate inferred content based on input types
        if all(isinstance(e.content, (int, float)) for e in input_elements):
            # Numeric inference - extrapolate trend
            values = [e.content for e in input_elements]
            
            if len(values) >= 2:
                # Linear extrapolation
                diff = values[-1] - values[-2]
                inferred_content = values[-1] + diff
            else:
                inferred_content = values[0]
                
        elif all(isinstance(e.content, str) for e in input_elements):
            # Text inference - combine with "therefore"
            texts = [e.content for e in input_elements if e.content]
            if texts:
                inferred_content = f"If {texts[0]} and {texts[1] if len(texts) > 1 else ''}, then {texts[-1]}"
            else:
                inferred_content = "Inferred conclusion"
                
        elif all(isinstance(e.content, dict) for e in input_elements):
            # Dictionary inference - merge and extend
            inferred_content = {}
            
            # Merge dictionaries
            for element in input_elements:
                for k, v in element.content.items():
                    if k in inferred_content and isinstance(v, (int, float)) and isinstance(inferred_content[k], (int, float)):
                        # For numeric values, calculate trend
                        inferred_content[k] = v + (v - inferred_content[k])
                    else:
                        inferred_content[k] = v
                        
        else:
            # Default inference
            inferred_content = f"Inference from {len(input_elements)} elements"
        
        # Create inference element
        inference_id = self.add_element(
            content=inferred_content,
            coordinates=inferred_coords,
            type_tags=["inference"],
            fitness=0.5
        )
        
        # Connect to input elements
        for input_id in inputs:
            if input_id in self.elements:
                self.connect_elements(input_id, inference_id, 0.6)
        
        return {
            "success": True,
            "inference_element": inference_id,
            "inferred_content": inferred_content,
            "confidence": 0.6  # Default confidence for inferences
        }
    
    def _process_analogy(self, inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply analogy process to find analogical relationships"""
        if len(inputs) < 3:
            return {"success": False, "reason": "Need at least 3 elements for analogy (A:B::C:?)"}
            
        # Get input elements
        input_elements = [self.elements.get(input_id) for input_id in inputs[:3]]
        
        # Check if all elements exist
        if not all(input_elements):
            return {"success": False, "reason": "One or more elements not found"}
            
        # A:B::C:D analogy pattern
        # If A is to B as C is to D, then D = C + (B - A)
        # Get coordinates for A, B, C
        a, b, c = input_elements
        
        if a.coordinates is None or b.coordinates is None or c.coordinates is None:
            return {"success": False, "reason": "All elements must have coordinates"}
            
        # Calculate vector from A to B
        ab_vector = b.coordinates - a.coordinates
        
        # Apply the same vector to C
        d_coords = c.coordinates + ab_vector
        
        # Generate content for D based on the analogy
        if (isinstance(a.content, (int, float)) and 
            isinstance(b.content, (int, float)) and 
            isinstance(c.content, (int, float))):
            # Numeric analogy
            # If a:b = c:d, then d = c * (b/a)
            if a.content != 0:
                d_content = c.content * (b.content / a.content)
            else:
                d_content = c.content + (b.content - a.content)
                
        elif (isinstance(a.content, str) and 
              isinstance(b.content, str) and 
              isinstance(c.content, str)):
            # Text analogy
            d_content = f"{c.content} is to {b.content} as {a.content}"
            
        elif (isinstance(a.content, dict) and 
              isinstance(b.content, dict) and 
              isinstance(c.content, dict)):
            # Dictionary analogy
            d_content = c.content.copy()
            
            # Find changes from a to b
            for k in set(a.content.keys()).union(b.content.keys()):
                if k in a.content and k in b.content:
                    # Key exists in both a and b
                    if k in c.content:
                        # Apply similar transformation
                        a_val = a.content[k]
                        b_val = b.content[k]
                        c_val = c.content[k]
                        
                        if (isinstance(a_val, (int, float)) and 
                            isinstance(b_val, (int, float)) and 
                            isinstance(c_val, (int, float))):
                            # Numeric transformation
                            if a_val != 0:
                                d_content[k] = c_val * (b_val / a_val)
                            else:
                                d_content[k] = c_val + (b_val - a_val)
                elif k in b.content and k not in a.content:
                    # Key added in b
                    d_content[k] = b.content[k]
                elif k in a.content and k not in b.content:
                    # Key removed in b
                    if k in d_content:
                        del d_content[k]
            
        else:
            # Default analogy
            d_content = f"Analogy: {c.content} is to ? as {a.content} is to {b.content}"
        
        # Create the analogy element
        analogy_id = self.add_element(
            content=d_content,
            coordinates=d_coords,
            type_tags=["analogy"],
            fitness=0.5
        )
        
        # Connect to input elements
        for input_id in inputs[:3]:
            if input_id in self.elements:
                self.connect_elements(input_id, analogy_id, 0.7)
        
        return {
            "success": True,
            "analogy_element": analogy_id,
            "analogy_content": d_content,
            "source_elements": inputs[:3],
            "confidence": 0.7
        }
    
    def _process_synthesis(self, inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply synthesis process to combine concepts"""
        # Get input elements
        input_elements = [self.elements[input_id] for input_id in inputs if input_id in self.elements]
        
        if len(input_elements) < 2:
            return {"success": False, "reason": "Need at least 2 elements for synthesis"}
            
        # Extract coordinates
        coords_list = [e.coordinates for e in input_elements if e.coordinates is not None]
        
        if coords_list:
            # Calculate centroid for new position
            synthesis_coords = np.mean(coords_list, axis=0)
            
            # Add some randomness based on creativity parameter
            creativity = params.get("creativity", 0.5)
            if creativity > 0:
                noise_scale = creativity * 0.2
                noise = np.random.normal(0, noise_scale, self.dimensions)
                synthesis_coords += noise
        else:
            synthesis_coords = None
        
        # Synthesize content based on input types
        if all(isinstance(e.content, (int, float)) for e in input_elements):
            # Numeric synthesis - calculate weighted average
            values = [e.content for e in input_elements]
            weights = [e.fitness for e in input_elements]
            
            if sum(weights) > 0:
                synthesis_content = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            else:
                synthesis_content = sum(values) / len(values)
                
        elif all(isinstance(e.content, str) for e in input_elements):
            # Text synthesis
            texts = [e.content for e in input_elements if e.content]
            
            # Extract key words
            all_words = []
            for text in texts:
                words = text.lower().split()
                all_words.extend(words)
                
            # Count word frequencies
            word_counts = {}
            for word in all_words:
                if len(word) > 2:  # Ignore short words
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
            # Find most common words
            common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:7]
            
            if common_words:
                # Construct synthesized text
                creativity = params.get("creativity", 0.5)
                if creativity > 0.7:
                    # Creative synthesis - generate new sentence
                    random.shuffle(common_words)
                    synthesis_content = "Synthesis: " + " ".join(word for word, _ in common_words)
                else:
                    # More literal synthesis
                    synthesis_content = " ".join(f"{texts[0]}", f"{texts[1] if len(texts) > 1 else ''}")
            else:
                synthesis_content = "Synthesized concept"
                
        elif all(isinstance(e.content, dict) for e in input_elements):
            # Dictionary synthesis - merge with preference for higher fitness
            synthesis_content = {}
            
            # Collect all keys
            all_keys = set()
            for element in input_elements:
                all_keys.update(element.content.keys())
                
            # For each key, select value from element with highest fitness
            for key in all_keys:
                candidates = [(e.content.get(key), e.fitness) for e in input_elements if key in e.content]
                
                if candidates:
                    # Sort by fitness
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    synthesis_content[key] = candidates[0][0]
                    
        elif all(isinstance(e.content, list) for e in input_elements):
            # List synthesis - combine elements with deduplication
            synthesis_content = []
            
            # Collect all items
            all_items = []
            for element in input_elements:
                all_items.extend(element.content)
                
            # Try to deduplicate
            try:
                # Convert to strings for comparison
                str_items = [str(item) for item in all_items]
                unique_indices = []
                seen = set()
                
                for i, item_str in enumerate(str_items):
                    if item_str not in seen:
                        seen.add(item_str)
                        unique_indices.append(i)
                
                # Keep original items in order of first appearance
                synthesis_content = [all_items[i] for i in unique_indices]
                
                # Limit length
                max_length = 10
                if len(synthesis_content) > max_length:
                    synthesis_content = synthesis_content[:max_length]
                    
            except:
                # Fallback if deduplication fails
                synthesis_content = all_items[:10]
                
        else:
            # Mixed type synthesis - create a container
            synthesis_content = {
                "type": "synthesis",
                "components": [str(e.content)[:50] for e in input_elements],
                "timestamp": time.time()
            }
        
        # Collect tags from all input elements
        all_tags = set()
        for element in input_elements:
            all_tags.update(element.type_tags)
        
        # Create synthesis element
        synthesis_id = self.add_element(
            content=synthesis_content,
            coordinates=synthesis_coords,
            type_tags=list(all_tags) + ["synthesis"],
            fitness=0.6
        )
        
        # Connect to input elements
        for input_id in inputs:
            if input_id in self.elements:
                self.connect_elements(input_id, synthesis_id, 0.8)
                self.connect_elements(synthesis_id, input_id, 0.6)
        
        return {
            "success": True,
            "synthesis_element": synthesis_id,
            "synthesized_content": synthesis_content,
            "source_elements": inputs,
            "tags": list(all_tags)
        }

class CognitiveManifoldEvolution:
    """
    Main controller for the Cognitive Manifold Evolution System.
    
    This class provides a high-level interface to the cognitive manifold
    evolution framework, integrating all components into a cohesive system
    for advanced cognitive modeling and reasoning.
    
    Features:
    - Automatic evolution scheduling and management
    - Performance monitoring and metrics collection
    - Thread-safe concurrent operations
    - Configurable evolution strategies and parameters
    - Built-in cognitive process implementations
    - System state persistence and restoration
    
    The system manages the lifecycle of cognitive elements within a geometric
    manifold space, automatically optimizing their organization and relationships
    through various evolutionary strategies.
    
    Attributes:
        manifold (CognitiveManifold): The underlying manifold space
        evolution_history (List): Record of evolution operations
        config (Dict): System configuration parameters
        metrics (Dict): Performance and usage statistics
    """
    
    def __init__(self, dimensions: int = 8, manifold_type: ManifoldType = ManifoldType.EUCLIDEAN):
        """
        Initialize the cognitive manifold evolution system.
        
        Args:
            dimensions: Dimensionality of the manifold space (default: 8)
            manifold_type: Geometric structure type (default: Euclidean)
            
        Note:
            The system starts in manual mode. Enable auto-evolution using
            start_auto_evolution() for autonomous operation.
        """
        self.manifold = CognitiveManifold(dimensions, manifold_type)
        self.evolution_history = []
        self.creation_time = time.time()
        self.last_evolution_time = None
        self.config = {
            "auto_evolution_enabled": False,
            "default_evolution_strategy": EvolutionStrategy.GRADIENT_ASCENT,
            "evolution_interval": 60,  # seconds
            "auto_consolidation_enabled": True,
            "consolidation_interval": 300,  # seconds
            "log_level": "info"
        }
        self.evolution_thread = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            "total_evolutions": 0,
            "avg_evolution_time": 0.0,
            "improvement_rate": 0.0,
            "structural_changes": 0,
            "total_processes": 0
        }
    
    def start_evolution(self) -> bool:
        """Start the automatic evolution thread"""
        if self.evolution_thread is not None and self.evolution_thread.is_alive():
            return False
            
        self.running = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()
        
        return True
    
    def stop_evolution(self) -> bool:
        """Stop the automatic evolution thread"""
        if self.evolution_thread is None or not self.evolution_thread.is_alive():
            return False
            
        self.running = False
        self.evolution_thread.join(timeout=1.0)
        
        return True
    
    def evolve_once(self, strategy: Optional[EvolutionStrategy] = None, 
                  iterations: int = 1) -> Dict[str, Any]:
        """Perform one evolution step with the specified strategy"""
        if strategy is None:
            strategy = self.config["default_evolution_strategy"]
            
        start_time = time.time()
        
        # Perform evolution
        result = self.manifold.evolve(strategy, iterations)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.metrics["total_evolutions"] += 1
        
        # Update average evolution time
        if self.metrics["total_evolutions"] > 1:
            self.metrics["avg_evolution_time"] = (
                (self.metrics["avg_evolution_time"] * (self.metrics["total_evolutions"] - 1) + 
                execution_time) / self.metrics["total_evolutions"]
            )
        else:
            self.metrics["avg_evolution_time"] = execution_time
            
        # Update improvement rate
        self.metrics["improvement_rate"] = (
            self.metrics["improvement_rate"] * 0.9 + 
            result.get("fitness_improvement", 0) * 0.1
        )
        
        # Update structural changes
        self.metrics["structural_changes"] += result.get("structural_changes", 0)
        
        # Record evolution
        self.evolution_history.append({
            "timestamp": time.time(),
            "strategy": strategy.name,
            "iterations": iterations,
            "result": result
        })
        
        self.last_evolution_time = time.time()
        
        return {
            "success": True,
            "evolution_result": result,
            "execution_time": execution_time,
            "strategy": strategy.name
        }
    
    def apply_process(self, process_type: CognitiveProcess, 
                    inputs: List[str], 
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a cognitive process to the manifold"""
        if params is None:
            params = {}
            
        start_time = time.time()
        
        # Apply process
        result = self.manifold.process(process_type, inputs, params)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.metrics["total_processes"] += 1
        
        return {
            "success": result.get("success", False),
            "process_type": process_type.name,
            "execution_time": execution_time,
            "result": result
        }
    
    def add_concept(self, content: Any, coordinates: Optional[np.ndarray] = None, 
                  type_tags: Optional[List[str]] = None) -> str:
        """Add a new concept to the cognitive manifold"""
        if type_tags is None:
            type_tags = []
            
        # Auto-tag based on content type
        if not type_tags:
            if isinstance(content, (int, float)):
                type_tags.append("numeric")
            elif isinstance(content, str):
                type_tags.append("text")
            elif isinstance(content, (list, tuple)):
                type_tags.append("list")
            elif isinstance(content, dict):
                type_tags.append("structure")
                
        # Add element to manifold
        element_id = self.manifold.add_element(
            content=content,
            coordinates=coordinates,
            type_tags=type_tags
        )
        
        return element_id
    
    def search_concepts(self, query: Any, search_type: str = "content", 
                      limit: int = 5) -> List[Dict[str, Any]]:
        """Search for concepts in the manifold"""
        results = []
        
        if search_type == "content":
            # Search by content similarity
            # Convert query to coordinates
            if isinstance(query, np.ndarray) and query.ndim == 1:
                query_coords = query
            else:
                # Generate random coordinates as fallback
                query_coords = np.random.uniform(-1, 1, self.manifold.dimensions)
            
            # Find nearest elements
            nearest = self.manifold.get_nearest_elements(
                query_coords, k=limit
            )
            
            # Prepare results
            for element_id, distance in nearest:
                if element_id in self.manifold.elements:
                    element = self.manifold.elements[element_id]
                    results.append({
                        "element_id": element_id,
                        "content": element.content,
                        "distance": distance,
                        "similarity": 1.0 / (1.0 + distance),
                        "tags": list(element.type_tags)
                    })
                    
        elif search_type == "tag":
            # Search by tag
            if isinstance(query, str):
                matching_elements = self.manifold.get_elements_by_type(query)
                
                # Get details for each element
                for element_id in matching_elements[:limit]:
                    if element_id in self.manifold.elements:
                        element = self.manifold.elements[element_id]
                        results.append({
                            "element_id": element_id,
                            "content": element.content,
                            "fitness": element.fitness,
                            "tags": list(element.type_tags)
                        })
                        
        elif search_type == "activation":
            # Search by activation level
            activated_elements = [
                (element_id, element) 
                for element_id, element in self.manifold.elements.items()
                if element.activation > 0.1
            ]
            
            # Sort by activation (highest first)
            activated_elements.sort(key=lambda x: x[1].activation, reverse=True)
            
            # Prepare results
            for element_id, element in activated_elements[:limit]:
                results.append({
                    "element_id": element_id,
                    "content": element.content,
                    "activation": element.activation,
                    "tags": list(element.type_tags)
                })
                
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the system"""
        # Get manifold statistics
        manifold_stats = self.manifold.get_manifold_stats()
        
        # Calculate system uptime
        uptime = time.time() - self.creation_time
        
        # Prepare status report
        status = {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "manifold": manifold_stats,
            "evolution": {
                "total_evolutions": self.metrics["total_evolutions"],
                "avg_evolution_time": self.metrics["avg_evolution_time"],
                "improvement_rate": self.metrics["improvement_rate"],
                "auto_evolution": self.config["auto_evolution_enabled"],
                "last_evolution": self.last_evolution_time,
                "time_since_last_evolution": time.time() - (self.last_evolution_time or self.creation_time)
            },
            "processes": {
                "total_processes": self.metrics["total_processes"]
            },
            "config": self.config
        }
        
        return status
    
    def save_system(self, filepath: str) -> bool:
        """Save the entire system to a file"""
        try:
            # Create a system snapshot
            system_data = {
                "creation_time": self.creation_time,
                "last_evolution_time": self.last_evolution_time,
                "metrics": self.metrics,
                "config": self.config,
                "evolution_history": self.evolution_history,
                "timestamp": time.time()
            }
            
            # Save system data
            system_filepath = f"{filepath}_system.pkl"
            with open(system_filepath, 'wb') as f:
                pickle.dump(system_data, f)
                
            # Save manifold
            manifold_filepath = f"{filepath}_manifold.pkl"
            manifold_saved = self.manifold.save_to_file(manifold_filepath)
            
            return manifold_saved
            
        except Exception as e:
            print(f"Error saving system: {str(e)}")
            return False
    
    @classmethod
    def load_system(cls, filepath: str) -> 'CognitiveManifoldEvolution':
        """Load the system from a file"""
        try:
            # Load system data
            system_filepath = f"{filepath}_system.pkl"
            with open(system_filepath, 'rb') as f:
                system_data = pickle.load(f)
                
            # Load manifold
            manifold_filepath = f"{filepath}_manifold.pkl"
            manifold = CognitiveManifold.load_from_file(manifold_filepath)
            
            # Create system
            system = cls(
                dimensions=manifold.dimensions,
                manifold_type=manifold.manifold_type
            )
            
            # Restore manifold
            system.manifold = manifold
            
            # Restore system data
            system.creation_time = system_data.get("creation_time", time.time())
            system.last_evolution_time = system_data.get("last_evolution_time")
            system.metrics = system_data.get("metrics", system.metrics)
            system.config = system_data.get("config", system.config)
            system.evolution_history = system_data.get("evolution_history", [])
            
            return system
            
        except Exception as e:
            print(f"Error loading system: {str(e)}")
            return cls()  # Return new system as fallback
    
    def _evolution_loop(self) -> None:
        """Background thread for automatic evolution"""
        last_evolution = time.time()
        last_consolidation = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Check if evolution is due
            if (self.config["auto_evolution_enabled"] and 
                current_time - last_evolution >= self.config["evolution_interval"]):
                
                try:
                    # Select evolution strategy
                    strategies = list(EvolutionStrategy)
                    if random.random() < 0.7:
                        # Usually use default strategy
                        strategy = self.config["default_evolution_strategy"]
                    else:
                        # Occasionally use random strategy
                        strategy = random.choice(strategies)
                        
                    # Perform evolution
                    self.evolve_once(strategy=strategy)
                    last_evolution = time.time()
                except Exception as e:
                    print(f"Error in evolution loop: {str(e)}")
            
            # Check if consolidation is due
            if (self.config["auto_consolidation_enabled"] and 
                current_time - last_consolidation >= self.config["consolidation_interval"]):
                
                try:
                    # Perform consolidation
                    self.evolve_once(strategy=EvolutionStrategy.CONSOLIDATION)
                    last_consolidation = time.time()
                except Exception as e:
                    print(f"Error in consolidation: {str(e)}")
            
            # Sleep to avoid excessive CPU usage
            time.sleep(1.0)


# Utility functions for common operations
def create_basic_system(dimensions: int = DEFAULT_DIMENSIONS) -> CognitiveManifoldEvolution:
    """
    Create a basic cognitive manifold evolution system with default settings.
    
    Args:
        dimensions: Number of manifold dimensions (default: 8)
        
    Returns:
        Configured CognitiveManifoldEvolution system
    """
    return CognitiveManifoldEvolution(
        dimensions=dimensions,
        manifold_type=ManifoldType.EUCLIDEAN
    )


def create_advanced_system(dimensions: int = DEFAULT_DIMENSIONS, 
                         manifold_type: ManifoldType = ManifoldType.MIXED_CURVATURE) -> CognitiveManifoldEvolution:
    """
    Create an advanced cognitive manifold evolution system with sophisticated settings.
    
    Args:
        dimensions: Number of manifold dimensions (default: 8)
        manifold_type: Geometric structure type (default: Mixed Curvature)
        
    Returns:
        Configured CognitiveManifoldEvolution system with advanced features
    """
    system = CognitiveManifoldEvolution(
        dimensions=dimensions,
        manifold_type=manifold_type
    )
    
    # Configure for advanced operations
    system.config.update({
        "auto_evolution_enabled": True,
        "evolution_interval": 30,  # More frequent evolution
        "auto_consolidation_enabled": True,
        "consolidation_interval": 120,  # Regular consolidation
        "log_level": "debug"
    })
    
    return system


def analyze_manifold_health(manifold: CognitiveManifold) -> Dict[str, Any]:
    """
    Analyze the health and characteristics of a cognitive manifold.
    
    Args:
        manifold: The manifold to analyze
        
    Returns:
        Dictionary containing health metrics and recommendations
    """
    stats = manifold.get_manifold_stats()
    
    # Calculate health indicators
    element_count = stats["element_count"]
    avg_fitness = stats.get("fitness", {}).get("mean", 0.0)
    avg_connections = stats.get("avg_connections_per_element", 0.0)
    curvature = stats["manifold_properties"]["curvature"]
    
    # Determine health status
    health_score = 0.0
    issues = []
    recommendations = []
    
    # Element count assessment
    if element_count < 5:
        issues.append("Insufficient elements for meaningful evolution")
        recommendations.append("Add more cognitive elements")
    elif element_count > 1000:
        issues.append("High element count may impact performance")
        recommendations.append("Consider pruning low-fitness elements")
    else:
        health_score += 0.3
    
    # Fitness assessment
    if avg_fitness > 0.7:
        health_score += 0.3
    elif avg_fitness < 0.3:
        issues.append("Low average fitness indicates poor element quality")
        recommendations.append("Apply fitness-improving evolution strategies")
    
    # Connectivity assessment
    if avg_connections > 5:
        issues.append("High connectivity may cause over-association")
        recommendations.append("Consider connection pruning")
    elif avg_connections < 0.5:
        issues.append("Low connectivity limits cognitive processes")
        recommendations.append("Establish more conceptual connections")
    else:
        health_score += 0.2
    
    # Curvature assessment
    if abs(curvature) > 2.0:
        issues.append("Extreme curvature may indicate structural problems")
        recommendations.append("Apply structural evolution strategies")
    else:
        health_score += 0.2
    
    return {
        "health_score": min(1.0, health_score),
        "status": "healthy" if health_score > 0.7 else "needs_attention" if health_score > 0.4 else "poor",
        "issues": issues,
        "recommendations": recommendations,
        "stats": stats
    }


def export_manifold_summary(manifold: CognitiveManifold, filepath: str) -> bool:
    """
    Export a human-readable summary of the manifold to a text file.
    
    Args:
        manifold: The manifold to summarize
        filepath: Path to save the summary file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        stats = manifold.get_manifold_stats()
        health = analyze_manifold_health(manifold)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("COGNITIVE MANIFOLD SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Manifold Type: {manifold.manifold_type.name}\n")
            f.write(f"Dimensions: {manifold.dimensions}\n\n")
            
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Elements: {stats['element_count']}\n")
            f.write(f"Total Connections: {stats.get('connection_count', 0)}\n")
            f.write(f"Average Fitness: {stats.get('fitness', {}).get('mean', 0):.3f}\n")
            f.write(f"Average Complexity: {stats.get('complexity', {}).get('mean', 0):.3f}\n")
            f.write(f"Active Elements: {stats.get('activation', {}).get('active_elements', 0)}\n\n")
            
            f.write("HEALTH ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Health Score: {health['health_score']:.2f}/1.00\n")
            f.write(f"Status: {health['status'].upper()}\n\n")
            
            if health['issues']:
                f.write("IDENTIFIED ISSUES:\n")
                for issue in health['issues']:
                    f.write(f"  • {issue}\n")
                f.write("\n")
            
            if health['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in health['recommendations']:
                    f.write(f"  • {rec}\n")
                f.write("\n")
            
            f.write("TYPE TAG DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            type_counts = stats.get('type_tag_counts', {})
            for tag, count in sorted(type_counts.items()):
                f.write(f"  {tag}: {count} elements\n")
        
        return True
        
    except Exception as e:
        print(f"Error exporting summary: {e}")
        return False


def run_example():
    """
    Run a comprehensive demonstration of the cognitive manifold evolution system.
    
    This example showcases the core capabilities of the system including:
    - Creating and configuring a cognitive manifold
    - Adding concepts with different types of content
    - Establishing conceptual connections
    - Applying various cognitive processes
    - Performing evolution operations
    - Monitoring system performance and statistics
    
    The demonstration uses a mixed-curvature manifold to show how different
    geometric structures can encode cognitive relationships in sophisticated ways.
    """
    print(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: Shriram-2005")
    
    print("\n" + "="*60)
    print("COGNITIVE MANIFOLD EVOLUTION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create system with advanced configuration
    print("\nInitializing Cognitive Manifold Evolution System...")
    system = CognitiveManifoldEvolution(
        dimensions=6,
        manifold_type=ManifoldType.MIXED_CURVATURE
    )
    
    print(f"  ✓ Manifold type: {system.manifold.manifold_type.name}")
    print(f"  ✓ Dimensions: {system.manifold.dimensions}")
    print(f"  ✓ System initialized successfully")
    
    # Add diverse cognitive concepts
    print("\nAdding initial cognitive concepts...")
    
    concept1_id = system.add_concept("Quantum entanglement", type_tags=["physics", "quantum"])
    concept2_id = system.add_concept("Neural networks", type_tags=["ai", "computing"])
    concept3_id = system.add_concept("Gravitational waves", type_tags=["physics", "astronomy"])
    concept4_id = system.add_concept("Consciousness", type_tags=["philosophy", "cognition"])
    concept5_id = system.add_concept({
        "name": "Topological data analysis",
        "field": "mathematics",
        "applications": ["machine learning", "physics", "biology"]
    }, type_tags=["math", "data"])
    
    print(f"  ✓ Added 5 concepts spanning physics, AI, mathematics, and philosophy")
    
    # Establish conceptual relationships
    print("\nEstablishing conceptual connections...")
    system.manifold.connect_elements(concept1_id, concept3_id, 0.7)  # Quantum entanglement -> Gravitational waves
    system.manifold.connect_elements(concept2_id, concept4_id, 0.6)  # Neural networks -> Consciousness
    system.manifold.connect_elements(concept2_id, concept5_id, 0.8)  # Neural networks -> Topological data analysis
    system.manifold.connect_elements(concept4_id, concept1_id, 0.5)  # Consciousness -> Quantum entanglement
    
    # Perform cognitive processes
    print("\nPerforming cognitive processes...")
    
    # Association
    association_result = system.apply_process(
        CognitiveProcess.ASSOCIATION, 
        [concept1_id],  # Start from quantum entanglement
        {"output_count": 2}
    )
    
    print("  Association from 'quantum entanglement':")
    if association_result["success"] and association_result["result"]["associations"]:
        for assoc in association_result["result"]["associations"]:
            print(f"  → Associated with: {assoc['content']}")
    
    # Synthesis
    synthesis_result = system.apply_process(
        CognitiveProcess.SYNTHESIS,
        [concept2_id, concept4_id]  # Neural networks + Consciousness
    )
    
    print("\n  Synthesis of 'neural networks' and 'consciousness':")
    if synthesis_result["success"]:
        synthesis_id = synthesis_result["result"]["synthesis_element"]
        synthesis_content = synthesis_result["result"]["synthesized_content"]
        print(f"  → Synthesized concept: {synthesis_content}")
    
    # Analogy
    analogy_result = system.apply_process(
        CognitiveProcess.ANALOGY,
        [concept1_id, concept3_id, concept2_id]  # Quantum entanglement:Gravitational waves::Neural networks:?
    )
    
    print("\n  Analogy process:")
    print(f"  → If quantum entanglement is to gravitational waves, neural networks is to:")
    if analogy_result["success"]:
        analogy_id = analogy_result["result"]["analogy_element"]
        analogy_content = analogy_result["result"]["analogy_content"]
        print(f"  → {analogy_content}")
    
    # Evolve the manifold
    print("\nEvolving the cognitive manifold...")
    evolution_result = system.evolve_once(
        strategy=EvolutionStrategy.EXPLORATION,
        iterations=2
    )
    
    print(f"  Evolution completed in {evolution_result['execution_time']:.4f} seconds")
    print(f"  Elements affected: {evolution_result['evolution_result'].get('elements_affected', 0)}")
    
    if "new_elements" in evolution_result["evolution_result"]:
        print(f"  New elements created: {len(evolution_result['evolution_result']['new_elements'])}")
    
    print("\n" + "="*60)
    print("FINAL SYSTEM STATUS")
    print("="*60)
    
    # Comprehensive system analysis
    status = system.get_system_status()
    health = analyze_manifold_health(system.manifold)
    
    print(f"\nManifold Statistics:")
    print(f"  • Total elements: {status['manifold']['element_count']}")
    print(f"  • Average fitness: {status['manifold'].get('fitness', {}).get('mean', 0):.4f}")
    print(f"  • Manifold curvature: {status['manifold']['manifold_properties']['curvature']:.4f}")
    print(f"  • Effective dimensionality: {status['manifold']['manifold_properties']['dimensionality']}")
    print(f"  • Total connections: {status['manifold'].get('connection_count', 0)}")
    
    print(f"\nSystem Health:")
    print(f"  • Health score: {health['health_score']:.2f}/1.00")
    print(f"  • Status: {health['status'].upper()}")
    
    if health['recommendations']:
        print(f"\nRecommendations:")
        for rec in health['recommendations'][:3]:  # Show top 3
            print(f"  • {rec}")
    
    print(f"\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe Cognitive Manifold Evolution System has successfully demonstrated:")
    print("  ✓ Multi-dimensional cognitive representation")
    print("  ✓ Dynamic spatial organization of concepts")
    print("  ✓ Advanced cognitive process simulation")
    print("  ✓ Adaptive evolution strategies")
    print("  ✓ Real-time system monitoring and analysis")
    print("\nThis framework enables sophisticated AI reasoning through")
    print("geometric modeling of cognitive relationships.")


# Module exports for public API
__all__ = [
    # Core classes
    'CognitiveManifoldEvolution',
    'CognitiveManifold', 
    'CognitiveElement',
    
    # Enumerations
    'ManifoldType',
    'CognitiveProcess',
    'EvolutionStrategy',
    
    # Utility functions
    'create_basic_system',
    'create_advanced_system',
    'analyze_manifold_health',
    'export_manifold_summary',
    'run_example',
    
    # Constants
    'DEFAULT_DIMENSIONS',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_TEMPERATURE',
    '__version__',
    '__author__'
]


if __name__ == "__main__":
    # Run the demonstration when executed directly
    try:
        run_example()
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        print("Please check the system configuration and try again.")
    finally:
        print("\nThank you for exploring the Cognitive Manifold Evolution System!")