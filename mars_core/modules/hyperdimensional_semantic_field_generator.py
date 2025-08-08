"""
MARS Hyperdimensional Semantic Field Generator

This module implements an advanced semantic representation system using hyperdimensional 
computing (HDC) principles for sophisticated natural language understanding and reasoning.
The system creates high-dimensional vector representations of semantic concepts and
provides powerful operations for compositional semantics, analogical reasoning,
and relationship modeling.

Features:
    - 10 different encoding methods for semantic representation
    - Advanced vector operations (bind, bundle, permute, transform)
    - Compositional semantics with weighted combination
    - Analogical reasoning and pattern completion
    - Semantic relationship modeling and path finding
    - Thread-safe operations with concurrent access support
    - Adaptive and hierarchical dimensionality management
    - Comprehensive statistics and monitoring

Encoding Methods:
    - BINARY: Binary hypervectors with {0,1} values
    - BIPOLAR: Bipolar vectors with {-1,+1} values
    - HOLOGRAPHIC: Holographic reduced representations (HRR)
    - FRACTAL: Self-similar fractal encoding patterns
    - CIRCULAR: Circular holographic encoding
    - QUANTUM: Quantum-inspired superposition states
    - SPARSE: Sparse distributed representations
    - GEOMETRIC: Geometric algebra encoding
    - TENSOR: Tensor product encoding
    - CONTEXTUAL: Context-sensitive adaptive encoding

Operations:
    - BIND: Creates associative connections between concepts
    - BUNDLE: Superposition of multiple concepts
    - PERMUTE: Spatial/temporal ordering operations
    - TRANSFORM: Linear transformations and projections
    - QUERY: Similarity-based retrieval
    - ANALOGY: Analogical reasoning (A:B::C:?)
    - COMPOSE: Compositional semantic construction
    - NEGATE: Semantic negation and opposition
    - PROJECT: Subspace projection operations
    - ENTANGLE: Quantum-like semantic entanglement

Mathematical Foundation:
    The system employs vector symbolic architectures (VSA) with high-dimensional
    representations typically ranging from 1,000 to 10,000 dimensions. Operations
    preserve semantic relationships while enabling efficient computation.

Usage Example:
    ```python
    # Initialize semantic field
    field = HyperdimensionalSemanticField(
        dimensions=10000,
        encoding_method=EncodingMethod.BINARY
    )
    
    # Create semantic vectors
    cat_id = field.create_vector(label="cat")
    animal_id = field.create_vector(label="animal")
    
    # Establish relationships
    field.relate_vectors(animal_id, cat_id, SemanticRelation.HYPERNYM)
    
    # Perform operations
    analogy = field.execute_analogy("cat", "animal", "dog")
    ```

References:
    - Kanerva, P. (2009). Hyperdimensional computing
    - Plate, T. A. (2003). Holographic reduced representations
    - Gayler, R. W. (2003). Vector symbolic architectures

"""
# Standard library imports
import hashlib
import heapq
import math
import os
import pickle
import random
import re
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import numpy as np
import scipy.sparse as sparse
from scipy.spatial.distance import cityblock, cosine, euclidean

# System constants and configuration
DEFAULT_DIMENSIONS = 10000
DEFAULT_SPARSITY = 0.01  # 1% of dimensions active in sparse encoding
MAX_CACHE_SIZE = 1000
SIMILARITY_THRESHOLD = 0.7
CONFIDENCE_DECAY_RATE = 0.95
ACCESS_COUNT_THRESHOLD = 100

# Mathematical constants for vector operations
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
PI_OVER_TWO = math.pi / 2
EULER_CONSTANT = math.e
NORMALIZATION_EPSILON = 1e-12

# Threading and performance constants
MAX_THREADS = 4
OPERATION_TIMEOUT = 30.0  # seconds
MEMORY_OPTIMIZATION_INTERVAL = 3600  # 1 hour

class EncodingMethod(Enum):
    """
    Enumeration of encoding methods for hyperdimensional semantic representations.
    
    Each encoding method provides different mathematical properties and trade-offs
    for representing semantic information in high-dimensional vector spaces.
    """
    BINARY = auto()           # Binary hypervectors {0,1} - memory efficient
    BIPOLAR = auto()          # Bipolar vectors {-1,+1} - algebraically convenient
    HOLOGRAPHIC = auto()      # Holographic reduced representations - complex-valued
    FRACTAL = auto()          # Fractal encoding - self-similar patterns
    CIRCULAR = auto()         # Circular holographic encoding - phase-based
    QUANTUM = auto()          # Quantum-inspired encoding - superposition states
    SPARSE = auto()           # Sparse distributed encoding - few active dimensions
    GEOMETRIC = auto()        # Geometric algebra encoding - multivector representations
    TENSOR = auto()           # Tensor product encoding - high-order relationships
    CONTEXTUAL = auto()       # Context-sensitive encoding - adaptive representations


class SemanticOperation(Enum):
    """
    Enumeration of semantic operations in hyperdimensional vector space.
    
    These operations form the computational primitives for semantic reasoning,
    compositional semantics, and analogical thinking in the vector space.
    """
    BIND = auto()             # Binding operation - creates associative connections
    BUNDLE = auto()           # Bundling operation - superposition of concepts
    PERMUTE = auto()          # Permutation - spatial/temporal ordering
    TRANSFORM = auto()        # Linear transformation - rotations and projections
    QUERY = auto()            # Similarity query - nearest neighbor search
    ANALOGY = auto()          # Analogical reasoning - pattern completion
    COMPOSE = auto()          # Compositional semantics - meaning construction
    NEGATE = auto()           # Semantic negation - logical opposition
    PROJECT = auto()          # Semantic projection - subspace extraction
    ENTANGLE = auto()         # Semantic entanglement - quantum-like correlations

class SemanticRelation(Enum):
    """
    Enumeration of semantic relationships between concepts.
    
    These relationships define the structure of semantic knowledge and enable
    sophisticated reasoning about concept hierarchies and associations.
    """
    SYNONYM = auto()          # Similar meaning - conceptual equivalence
    ANTONYM = auto()          # Opposite meaning - semantic opposition
    HYPERNYM = auto()         # Superordinate - general category (animal -> cat)
    HYPONYM = auto()          # Subordinate - specific instance (cat -> animal)
    MERONYM = auto()          # Part-whole relationship - component (wheel -> car)
    HOLONYM = auto()          # Whole-part relationship - container (car -> wheel)
    CAUSAL = auto()           # Causal relationship - cause and effect
    TEMPORAL = auto()         # Temporal relationship - time-based ordering
    SPATIAL = auto()          # Spatial relationship - location-based
    FUNCTIONAL = auto()       # Functional relationship - purpose or role


class DimensionalityType(Enum):
    """
    Enumeration of dimensionality management strategies.
    
    Different approaches to managing the dimensional complexity of the
    semantic representation space for optimal performance and memory usage.
    """
    FIXED = auto()            # Fixed dimensionality - consistent size
    ADAPTIVE = auto()         # Adaptive dimensionality - dynamic size adjustment
    HIERARCHICAL = auto()     # Hierarchical dimensions - nested subspaces
    FRACTAL = auto()          # Fractal dimensionality - self-similar scaling
    MANIFOLD = auto()         # Manifold-based - non-linear embedding spaces
    TENSOR = auto()           # Tensor dimensionality - multi-way representations
    QUANTUM = auto()          # Quantum-inspired dimensionality - superposition states


@dataclass
class SemanticVector:
    """
    Represents a semantic concept as a vector in hyperdimensional space.
    
    This dataclass encapsulates all information about a semantic vector including
    its high-dimensional representation, metadata, and access statistics.
    
    Attributes:
        vector_id: Unique identifier for the vector
        dimensions: Number of dimensions in the vector space
        label: Human-readable label for the concept
        encoding: Encoding method used for this vector
        data: Dense numpy array representation
        sparse_data: Sparse matrix representation (optional)
        confidence: Confidence score for vector quality (0.0-1.0)
        creation_time: Timestamp when vector was created
        last_accessed: Timestamp of last access
        access_count: Number of times vector has been accessed
        importance: Importance score for optimization (0.0-1.0)
        metadata: Additional metadata dictionary
        
    Methods:
        get_data: Retrieve the appropriate data representation
        update_access: Update access statistics
        to_dict: Convert to dictionary representation
    """
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dimensions: int = 10000  # Hyperdimensional vectors are typically very high-dimensional
    encoding: EncodingMethod = EncodingMethod.BINARY
    label: Optional[str] = None
    creation_time: float = field(default_factory=time.time)
    
    # The actual vector data
    data: np.ndarray = None
    sparse_data: sparse.csr_matrix = None
    
    # Metadata about the vector
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [vector_ids]
    importance: float = 1.0  # Importance weight
    confidence: float = 1.0  # Confidence score
    
    # Usage statistics
    access_count: int = 0
    last_accessed: float = None
    
    def initialize(self, method: EncodingMethod = None) -> None:
        """Initialize the vector data based on encoding method"""
        if method is not None:
            self.encoding = method
            
        if self.data is not None or self.sparse_data is not None:
            return  # Already initialized
            
        if self.encoding == EncodingMethod.BINARY:
            # Binary vectors (0/1)
            self.data = np.random.randint(0, 2, self.dimensions, dtype=np.int8)
            
        elif self.encoding == EncodingMethod.BIPOLAR:
            # Bipolar vectors (-1/+1)
            self.data = np.random.choice([-1, 1], self.dimensions, dtype=np.int8)
            
        elif self.encoding == EncodingMethod.SPARSE:
            # Sparse distributed representation
            # Only k elements are 1, rest are 0
            k = int(self.dimensions * 0.05)  # 5% density
            indices = np.random.choice(self.dimensions, k, replace=False)
            data = np.ones(k, dtype=np.int8)
            self.sparse_data = sparse.csr_matrix((data, indices, np.array([0, k])),
                                               shape=(1, self.dimensions))
            
        elif self.encoding == EncodingMethod.HOLOGRAPHIC:
            # Complex-valued vectors for holographic encoding
            self.data = np.exp(1j * np.random.uniform(0, 2*np.pi, self.dimensions))
            
        elif self.encoding == EncodingMethod.CIRCULAR:
            # Angles for circular holographic encoding
            self.data = np.random.uniform(0, 2*np.pi, self.dimensions)
            
        elif self.encoding == EncodingMethod.FRACTAL:
            # Fractal encoding with recursive structure
            base_dim = int(np.sqrt(self.dimensions))
            base_vector = np.random.uniform(-1, 1, base_dim)
            
            # Build fractal pattern by repeating and perturbing
            self.data = np.zeros(self.dimensions)
            for i in range(0, self.dimensions, base_dim):
                end_idx = min(i + base_dim, self.dimensions)
                segment_len = end_idx - i
                perturb = np.random.normal(0, 0.1, segment_len)
                self.data[i:end_idx] = base_vector[:segment_len] + perturb
                
        elif self.encoding == EncodingMethod.GEOMETRIC:
            # Geometric algebra encoding - multivectors
            # Represent as coefficients of basis elements
            self.data = np.random.normal(0, 1, self.dimensions)
            
        elif self.encoding == EncodingMethod.QUANTUM:
            # Quantum-inspired encoding - complex amplitudes
            real_part = np.random.normal(0, 1, self.dimensions)
            imag_part = np.random.normal(0, 1, self.dimensions)
            self.data = real_part + 1j * imag_part
            
            # Normalize like a quantum state
            norm = np.linalg.norm(self.data)
            if norm > 0:
                self.data = self.data / norm
                
        else:
            # Default to random real-valued vector
            self.data = np.random.normal(0, 1, self.dimensions)
            # Normalize to unit length
            norm = np.linalg.norm(self.data)
            if norm > 0:
                self.data = self.data / norm
    
    def access(self) -> None:
        """Record vector access"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def get_data(self) -> Union[np.ndarray, sparse.csr_matrix]:
        """Get the vector data"""
        return self.sparse_data if self.sparse_data is not None else self.data
    
    def normalize(self) -> None:
        """Normalize the vector to unit length"""
        if self.sparse_data is not None:
            # Normalize sparse vector
            data = self.sparse_data.data
            norm = np.sqrt(np.sum(data**2))
            if norm > 0:
                self.sparse_data.data = data / norm
        elif self.data is not None:
            # Check if binary/bipolar encoding (don't normalize these)
            if self.encoding in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
                return
                
            # Normalize dense vector
            norm = np.linalg.norm(self.data)
            if norm > 0:
                self.data = self.data / norm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "vector_id": self.vector_id,
            "dimensions": self.dimensions,
            "encoding": self.encoding.name,
            "label": self.label,
            "creation_time": self.creation_time,
            "metadata": self.metadata,
            "relations": self.relations,
            "importance": self.importance,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
        
        # Vector data representation depends on encoding
        if self.sparse_data is not None:
            result["is_sparse"] = True
            # Store sparse matrix in a compact format
            result["indices"] = self.sparse_data.indices.tolist()
            result["data"] = self.sparse_data.data.tolist()
            result["shape"] = self.sparse_data.shape
        elif self.data is not None:
            result["is_sparse"] = False
            # Handle different data types
            if np.iscomplexobj(self.data):
                result["data_real"] = self.data.real.tolist()
                result["data_imag"] = self.data.imag.tolist()
            else:
                result["data"] = self.data.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticVector':
        """Create a SemanticVector from dictionary representation"""
        vector = cls(
            vector_id=data["vector_id"],
            dimensions=data["dimensions"],
            encoding=EncodingMethod[data["encoding"]],
            label=data["label"],
            creation_time=data["creation_time"],
            metadata=data["metadata"],
            relations=data["relations"],
            importance=data["importance"],
            confidence=data["confidence"],
            access_count=data["access_count"],
            last_accessed=data["last_accessed"]
        )
        
        # Restore vector data
        if data.get("is_sparse", False):
            indices = np.array(data["indices"])
            values = np.array(data["data"])
            shape = data["shape"]
            vector.sparse_data = sparse.csr_matrix((values, indices, [0, len(indices)]), shape=shape)
        else:
            if "data_real" in data and "data_imag" in data:
                # Complex data
                vector.data = np.array(data["data_real"]) + 1j * np.array(data["data_imag"])
            elif "data" in data:
                # Real data
                vector.data = np.array(data["data"])
        
        return vector

class HyperdimensionalSemanticField:
    """
    Main system for managing hyperdimensional semantic vector representations.
    
    This class provides a comprehensive framework for creating, storing, and manipulating
    semantic vectors in high-dimensional space. It supports multiple encoding methods,
    sophisticated operations for semantic reasoning, and efficient indexing for
    large-scale semantic knowledge bases.
    
    Key Features:
        - Multiple encoding strategies for different semantic requirements
        - Thread-safe operations for concurrent access
        - Efficient indexing and retrieval systems
        - Advanced semantic operations (binding, bundling, analogies)
        - Relationship modeling and path finding
        - Adaptive dimensionality management
        - Comprehensive statistics and monitoring
        
    Architecture:
        The system uses Vector Symbolic Architectures (VSA) principles with
        high-dimensional distributed representations. Operations are designed
        to preserve semantic relationships while enabling efficient computation.
        
    Attributes:
        dimensions: Number of dimensions in the vector space
        encoding_method: Default encoding method for new vectors
        dimensionality_type: Strategy for managing dimensionality
        vectors: Main storage for semantic vectors
        label_index: Fast lookup from labels to vector IDs
        relation_index: Structured storage for semantic relationships
        subspaces: Hierarchical or specialized subspaces
        stats: System statistics and performance metrics
    """
    
    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, 
               encoding_method: EncodingMethod = EncodingMethod.BINARY,
               dimensionality_type: DimensionalityType = DimensionalityType.FIXED):
        """
        Initialize the hyperdimensional semantic field.
        
        Args:
            dimensions: Number of dimensions for vector representations
            encoding_method: Default method for encoding semantic vectors
            dimensionality_type: Strategy for dimensionality management
        """
        self.dimensions = dimensions
        self.encoding_method = encoding_method
        self.dimensionality_type = dimensionality_type
        
        # Core storage for semantic vectors
        self.vectors: Dict[str, SemanticVector] = {}
        
        # Efficient indexing structures
        self.label_index: Dict[str, str] = {}  # label -> vector_id
        self.relation_index: Dict[SemanticRelation, Dict[str, List[str]]] = {
            relation: {} for relation in SemanticRelation
        }  # relation -> source_id -> [target_ids]
        
        # Hierarchical and specialized subspaces
        self.subspaces: Dict[str, 'HyperdimensionalSemanticField'] = {}
        
        # For ADAPTIVE dimensionality
        self.adaptive_dimensions: List[int] = []
        self.dimension_importance: np.ndarray = None
        
        # Operation cache for expensive operations
        self.operation_cache: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "vector_count": 0,
            "operation_count": 0,
            "creation_time": time.time(),
            "last_modified": time.time(),
            "most_accessed_vectors": [],
            "avg_vector_confidence": 1.0
        }
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize based on dimensionality type
        self._initialize_dimensionality()
    
    def create_vector(self, label: Optional[str] = None, 
                    data: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new semantic vector in the hyperdimensional space.
        
        Creates a new semantic vector with the specified properties and adds it
        to the semantic field. The vector can be initialized with custom data
        or generated automatically based on the field's encoding method.
        
        Args:
            label: Optional human-readable label for the concept
            data: Optional pre-computed vector data
            metadata: Optional metadata dictionary
            
        Returns:
            Unique identifier for the created vector
            
        Thread Safety:
            This method is thread-safe and can be called concurrently.
        """
        with self._lock:
            # Check if label already exists to avoid duplicates
            if label is not None and label in self.label_index:
                return self.label_index[label]
            
            # Create new semantic vector
            vector = SemanticVector(
                dimensions=self.dimensions,
                encoding=self.encoding_method,
                label=label,
                metadata=metadata or {}
            )
            
            # Initialize with provided data or generate automatically
            if data is not None:
                if len(data) != self.dimensions:
                    raise ValueError(f"Data dimensions {len(data)} don't match expected {self.dimensions}")
                    
                vector.data = np.array(data)
                if self.encoding_method in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
                    # Ensure data conforms to encoding
                    if self.encoding_method == EncodingMethod.BINARY:
                        vector.data = (vector.data > 0).astype(np.int8)
                    else:  # BIPOLAR
                        vector.data = np.sign(vector.data).astype(np.int8)
                        # Replace zeros with random -1/1
                        zero_indices = np.where(vector.data == 0)[0]
                        vector.data[zero_indices] = np.random.choice([-1, 1], size=len(zero_indices))
            else:
                vector.initialize(self.encoding_method)
            
            # Store the vector
            vector_id = vector.vector_id
            self.vectors[vector_id] = vector
            
            # Update label index if provided
            if label is not None:
                self.label_index[label] = vector_id
            
            # Update statistics
            self.stats["vector_count"] += 1
            self.stats["last_modified"] = time.time()
            self._update_vector_confidence_stats()
            
            return vector_id
    
    def get_vector(self, identifier: str) -> Optional[SemanticVector]:
        """
        Retrieve a semantic vector by ID or label.
        
        Args:
            identifier: Vector ID or human-readable label
            
        Returns:
            SemanticVector instance if found, None otherwise
            
        Thread Safety:
            This method is thread-safe and can be called concurrently.
        """
        with self._lock:
            # Check if identifier is a vector ID
            if identifier in self.vectors:
                vector = self.vectors[identifier]
                vector.access()
                return vector
            
            # Check if identifier is a label
            if identifier in self.label_index:
                vector_id = self.label_index[identifier]
                if vector_id in self.vectors:
                    vector = self.vectors[vector_id]
                    vector.access()
                    return vector
            
            return None
    
    def get_vector_data(self, identifier: str) -> Optional[np.ndarray]:
        """
        Get the raw vector data by ID or label.
        
        Args:
            identifier: Vector ID or human-readable label
            
        Returns:
            Vector data as numpy array or sparse matrix, None if not found
        """
        vector = self.get_vector(identifier)
        if vector is None:
            return None
            
        return vector.get_data()
    
    def relate_vectors(self, source_id: str, target_id: str, 
                     relation: SemanticRelation) -> bool:
        """
        Create a semantic relation between two vectors.
        
        Establishes a directed semantic relationship from source to target
        vector using the specified relation type.
        
        Args:
            source_id: ID of the source vector
            target_id: ID of the target vector
            relation: Type of semantic relationship
            
        Returns:
            True if relation was created successfully, False otherwise
        """
        with self._lock:
            # Verify both vectors exist
            if source_id not in self.vectors or target_id not in self.vectors:
                return False
            
            # Update source vector's relations
            source = self.vectors[source_id]
            relation_name = relation.name
            if relation_name not in source.relations:
                source.relations[relation_name] = []
            if target_id not in source.relations[relation_name]:
                source.relations[relation_name].append(target_id)
            
            # Update relation index
            if source_id not in self.relation_index[relation]:
                self.relation_index[relation][source_id] = []
            if target_id not in self.relation_index[relation][source_id]:
                self.relation_index[relation][source_id].append(target_id)
            
            # For certain relations, create the inverse relation
            inverse_relations = {
                SemanticRelation.HYPERNYM: SemanticRelation.HYPONYM,
                SemanticRelation.HYPONYM: SemanticRelation.HYPERNYM,
                SemanticRelation.MERONYM: SemanticRelation.HOLONYM,
                SemanticRelation.HOLONYM: SemanticRelation.MERONYM
            }
            
            if relation in inverse_relations:
                inverse_relation = inverse_relations[relation]
                target = self.vectors[target_id]
                inv_relation_name = inverse_relation.name
                
                # Update target's relations
                if inv_relation_name not in target.relations:
                    target.relations[inv_relation_name] = []
                if source_id not in target.relations[inv_relation_name]:
                    target.relations[inv_relation_name].append(source_id)
                
                # Update inverse relation index
                if target_id not in self.relation_index[inverse_relation]:
                    self.relation_index[inverse_relation][target_id] = []
                if source_id not in self.relation_index[inverse_relation][target_id]:
                    self.relation_index[inverse_relation][target_id].append(source_id)
            
            # Update statistics
            self.stats["last_modified"] = time.time()
            
            return True
    
    def find_related_vectors(self, vector_id: str, 
                           relation: SemanticRelation) -> List[str]:
        """Find vectors related to the given vector by the specified relation"""
        with self._lock:
            # Check if vector exists
            if vector_id not in self.vectors:
                return []
            
            # Check relation index
            if vector_id in self.relation_index[relation]:
                return self.relation_index[relation][vector_id].copy()
            
            # Check vector's relations
            vector = self.vectors[vector_id]
            relation_name = relation.name
            if relation_name in vector.relations:
                # Update relation index for future lookups
                for target_id in vector.relations[relation_name]:
                    if target_id in self.vectors:
                        if vector_id not in self.relation_index[relation]:
                            self.relation_index[relation][vector_id] = []
                        if target_id not in self.relation_index[relation][vector_id]:
                            self.relation_index[relation][vector_id].append(target_id)
                
                return vector.relations[relation_name].copy()
            
            return []
    
    def find_similar_vectors(self, query: Union[str, np.ndarray, sparse.csr_matrix],
                           top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Find vectors similar to the query vector or identifier"""
        with self._lock:
            # Get query vector
            query_vector = None
            
            if isinstance(query, str):
                # Get by ID or label
                vector = self.get_vector(query)
                if vector is not None:
                    query_vector = vector.get_data()
            elif isinstance(query, (np.ndarray, sparse.csr_matrix)):
                # Direct vector data
                query_vector = query
            
            if query_vector is None:
                return []
            
            # Calculate similarity to all vectors
            similarities = []
            
            for vector_id, vector in self.vectors.items():
                vector_data = vector.get_data()
                
                # Skip if data is missing
                if vector_data is None:
                    continue
                
                # Calculate similarity based on encoding
                similarity = self._calculate_similarity(query_vector, vector_data, vector.encoding)
                
                if similarity >= threshold:
                    similarities.append((vector_id, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Update statistics
            self.stats["operation_count"] += 1
            
            return similarities[:top_k]
    
    def perform_operation(self, operation: SemanticOperation, 
                        operands: List[str],
                        operation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform a semantic vector operation"""
        with self._lock:
            operation_params = operation_params or {}
            
            # Get operand vectors
            vectors = []
            for op in operands:
                vector = self.get_vector(op)
                if vector is None:
                    return {"success": False, "error": f"Vector not found: {op}"}
                vectors.append(vector)
            
            # Generate cache key for expensive operations
            cache_key = None
            use_cache = operation_params.get("use_cache", True)
            
            if use_cache and operation in [SemanticOperation.BIND, SemanticOperation.ANALOGY]:
                # Create cache key from operation and sorted operands
                cache_key = f"{operation.name}:{','.join(sorted(operands))}"
                
                # Check cache
                if cache_key in self.operation_cache:
                    return self.operation_cache[cache_key]
            
            # Perform the operation
            result = self._execute_operation(operation, vectors, operation_params)
            
            # Cache result if needed
            if cache_key is not None and result.get("success", False):
                self.operation_cache[cache_key] = result
            
            # Update statistics
            self.stats["operation_count"] += 1
            self.stats["last_modified"] = time.time()
            
            return result
    
    def create_compositional_vector(self, components: Dict[str, float],
                                  label: Optional[str] = None,
                                  method: str = "weighted_sum") -> Optional[str]:
        """Create a new vector by composing existing vectors"""
        with self._lock:
            # Get component vectors
            component_vectors = {}
            for comp_id, weight in components.items():
                vector = self.get_vector(comp_id)
                if vector is None:
                    return None
                component_vectors[comp_id] = (vector, weight)
            
            if not component_vectors:
                return None
            
            # Compose vectors based on method
            if method == "weighted_sum":
                # Create weighted sum of vectors
                result_vector = None
                total_weight = 0.0
                
                for comp_id, (vector, weight) in component_vectors.items():
                    vector_data = vector.get_data()
                    if vector_data is None:
                        continue
                    
                    if result_vector is None:
                        if isinstance(vector_data, sparse.csr_matrix):
                            result_vector = weight * vector_data.copy()
                        else:
                            result_vector = weight * vector_data.copy()
                    else:
                        result_vector += weight * vector_data
                    
                    total_weight += abs(weight)
                
                # Normalize if needed
                if total_weight > 0 and result_vector is not None:
                    if isinstance(result_vector, sparse.csr_matrix):
                        result_vector = result_vector / total_weight
                    else:
                        result_vector = result_vector / total_weight
                
            elif method == "binding":
                # Use vector binding (multiplicative composition)
                result_vector = None
                
                for comp_id, (vector, _) in component_vectors.items():
                    vector_data = vector.get_data()
                    if vector_data is None:
                        continue
                    
                    if result_vector is None:
                        result_vector = vector_data.copy()
                    else:
                        result_vector = self._bind_vectors(result_vector, vector_data, vector.encoding)
            
            else:
                # Unknown method
                return None
            
            if result_vector is None:
                return None
            
            # Create new vector
            confidence = sum(vector.confidence * weight for _, (vector, weight) in component_vectors.items()) / sum(weight for _, (_, weight) in component_vectors.items())
            
            metadata = {
                "composition_method": method,
                "components": {comp_id: weight for comp_id, weight in components.items()},
                "creation_type": "composition"
            }
            
            vector_id = self.create_vector(label=label, data=result_vector, metadata=metadata)
            
            if vector_id:
                # Set confidence
                vector = self.vectors[vector_id]
                vector.confidence = confidence
                
                # Create relations to component vectors
                for comp_id, (_, weight) in component_vectors.items():
                    if weight > 0:
                        relation = SemanticRelation.HYPERNYM if weight < 0.5 else SemanticRelation.HOLONYM
                    else:
                        relation = SemanticRelation.ANTONYM
                    self.relate_vectors(vector_id, comp_id, relation)
            
            return vector_id
    
    def execute_analogy(self, a: str, b: str, c: str, 
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """Execute an analogy query: a is to b as c is to ?"""
        with self._lock:
                        # Get vectors
            a_vector = self.get_vector(a)
            b_vector = self.get_vector(b)
            c_vector = self.get_vector(c)
            
            if any(v is None for v in [a_vector, b_vector, c_vector]):
                return []
            
            # Get vector data
            a_data = a_vector.get_data()
            b_data = b_vector.get_data()
            c_data = c_vector.get_data()
            
            # Execute analogy based on encoding method
            target = None
            
            # For binary and bipolar vectors
            if a_vector.encoding in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
                # a:b as c:? means ? = c XOR a XOR b
                if isinstance(a_data, sparse.csr_matrix):
                    # For sparse binary vectors
                    target = self._sparse_xor(self._sparse_xor(c_data, a_data), b_data)
                else:
                    # For dense binary/bipolar vectors
                    if a_vector.encoding == EncodingMethod.BINARY:
                        target = (c_data ^ a_data ^ b_data).astype(a_data.dtype)
                    else:  # BIPOLAR
                        # For bipolar, multiplication is equivalent to XOR
                        target = c_data * a_data * b_data
            
            # For holographic vectors
            elif a_vector.encoding == EncodingMethod.HOLOGRAPHIC:
                # Holographic analogies using circular convolution
                # ? = c ⊗ a⁻¹ ⊗ b where ⊗ is circular convolution
                a_inv = np.conj(a_data)  # Inverse by complex conjugate
                target = self._circular_convolution(self._circular_convolution(c_data, a_inv), b_data)
            
            # For circular vectors
            elif a_vector.encoding == EncodingMethod.CIRCULAR:
                # Circular vectors use addition/subtraction modulo 2π
                target = (c_data - a_data + b_data) % (2*np.pi)
            
            # For real-valued or complex vectors
            else:
                # Use the vector offset method: ? = c + (b - a)
                target = c_data + (b_data - a_data)
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(target)
                if norm > 0:
                    target = target / norm
            
            # Find most similar vectors to target
            similarities = []
            
            for vector_id, vector in self.vectors.items():
                # Skip input vectors
                if vector_id in [a_vector.vector_id, b_vector.vector_id, c_vector.vector_id]:
                    continue
                    
                vector_data = vector.get_data()
                if vector_data is None:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(target, vector_data, vector.encoding)
                similarities.append((vector_id, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Update statistics
            self.stats["operation_count"] += 1
            
            return similarities[:top_k]
    
    def find_semantic_path(self, start_id: str, end_id: str, 
                         max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Find a semantic path connecting two concepts"""
        with self._lock:
            # Check if vectors exist
            start_vector = self.get_vector(start_id)
            end_vector = self.get_vector(end_id)
            
            if start_vector is None or end_vector is None:
                return None
            
            # Breadth-first search for path
            visited = {start_id}
            queue = deque([(start_id, [])])  # (node_id, path_so_far)
            
            while queue:
                current_id, path = queue.popleft()
                
                # Check all semantic relations
                for relation in SemanticRelation:
                    related_ids = self.find_related_vectors(current_id, relation)
                    
                    for related_id in related_ids:
                        if related_id == end_id:
                            # Found target, return path
                            return path + [
                                {"from": current_id, "to": related_id, "relation": relation.name}
                            ]
                            
                        if related_id not in visited and len(path) < max_depth - 1:
                            visited.add(related_id)
                            new_path = path + [
                                {"from": current_id, "to": related_id, "relation": relation.name}
                            ]
                            queue.append((related_id, new_path))
            
            # If no direct path through relations, try finding a common neighbor
            if max_depth >= 2:
                # Find vectors similar to both start and end
                start_similar = self.find_similar_vectors(start_id, top_k=10, threshold=0.5)
                end_similar = self.find_similar_vectors(end_id, top_k=10, threshold=0.5)
                
                start_similar_ids = {vector_id for vector_id, _ in start_similar}
                end_similar_ids = {vector_id for vector_id, _ in end_similar}
                
                # Find common similar vectors
                common = start_similar_ids.intersection(end_similar_ids)
                
                if common:
                    bridge_id = next(iter(common))
                    return [
                        {"from": start_id, "to": bridge_id, "relation": "SIMILARITY"},
                        {"from": bridge_id, "to": end_id, "relation": "SIMILARITY"}
                    ]
            
            # No path found
            return None
    
    def create_subspace(self, name: str, dimensions: int = None,
                      encoding_method: EncodingMethod = None) -> bool:
        """Create a semantic subspace (for HIERARCHICAL dimensionality)"""
        with self._lock:
            if self.dimensionality_type != DimensionalityType.HIERARCHICAL:
                return False
                
            if name in self.subspaces:
                return False
                
            # Use parent parameters if not specified
            if dimensions is None:
                dimensions = self.dimensions // 2  # Half the parent dimensions
            if encoding_method is None:
                encoding_method = self.encoding_method
                
            # Create subspace
            subspace = HyperdimensionalSemanticField(
                dimensions=dimensions,
                encoding_method=encoding_method,
                dimensionality_type=DimensionalityType.FIXED
            )
            
            self.subspaces[name] = subspace
            return True
    
    def get_subspace(self, name: str) -> Optional['HyperdimensionalSemanticField']:
        """Get a semantic subspace by name"""
        with self._lock:
            return self.subspaces.get(name)
    
    def update_dimension_importance(self, importance: np.ndarray) -> bool:
        """Update dimension importance for adaptive dimensionality"""
        with self._lock:
            if self.dimensionality_type != DimensionalityType.ADAPTIVE:
                return False
                
            if len(importance) != self.dimensions:
                return False
                
            # Update importance
            self.dimension_importance = np.array(importance)
            
            # Normalize
            total = np.sum(self.dimension_importance)
            if total > 0:
                self.dimension_importance = self.dimension_importance / total
                
            # Update adaptive dimensions
            threshold = 1.0 / self.dimensions  # Average importance
            self.adaptive_dimensions = np.where(self.dimension_importance > threshold)[0].tolist()
            
            return True
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize vector storage based on usage patterns"""
        with self._lock:
            start_time = time.time()
            
            # Find rarely used vectors
            rarely_used = []
            for vector_id, vector in self.vectors.items():
                # Skip vectors used in the last day
                if vector.last_accessed and time.time() - vector.last_accessed < 86400:
                    continue
                    
                # Consider vectors with low importance and few accesses
                if vector.importance < 0.5 and vector.access_count < 5:
                    rarely_used.append(vector_id)
            
            # Convert dense to sparse for binary vectors with < 10% ones
            conversions = 0
            for vector_id, vector in self.vectors.items():
                if (vector.encoding == EncodingMethod.BINARY and
                    vector.data is not None and
                    vector.sparse_data is None and
                    np.mean(vector.data) < 0.1):
                    
                    # Convert to sparse
                    indices = np.where(vector.data > 0)[0]
                    data = np.ones(len(indices), dtype=np.int8)
                    vector.sparse_data = sparse.csr_matrix((data, indices, np.array([0, len(indices)])),
                                                         shape=(1, vector.dimensions))
                    vector.data = None
                    conversions += 1
            
            # Update most accessed vectors
            top_vectors = sorted(self.vectors.values(), key=lambda v: v.access_count, reverse=True)[:10]
            self.stats["most_accessed_vectors"] = [
                {"id": v.vector_id, "label": v.label, "accesses": v.access_count}
                for v in top_vectors
            ]
            
            # Clear operation cache if too large
            cache_cleared = False
            if len(self.operation_cache) > 1000:
                self.operation_cache.clear()
                cache_cleared = True
            
            # Return optimization statistics
            return {
                "rarely_used_vectors": len(rarely_used),
                "dense_to_sparse_conversions": conversions,
                "cache_cleared": cache_cleared,
                "time_taken": time.time() - start_time
            }
    
    def save_to_file(self, filepath: str) -> bool:
        """Save the semantic field to a file"""
        try:
            with self._lock:
                # Create a serializable representation
                data = {
                    "dimensions": self.dimensions,
                    "encoding_method": self.encoding_method.name,
                    "dimensionality_type": self.dimensionality_type.name,
                    "stats": self.stats,
                    "label_index": self.label_index,
                    "relation_index": {
                        relation.name: {
                            source_id: target_ids
                            for source_id, target_ids in relation_dict.items()
                        }
                        for relation, relation_dict in self.relation_index.items()
                    },
                    "vectors": {}
                }
                
                # Add vectors (but limit size by excluding large arrays)
                for vector_id, vector in self.vectors.items():
                    data["vectors"][vector_id] = vector.to_dict()
                
                # For hierarchical dimensionality, save subspaces
                if self.dimensionality_type == DimensionalityType.HIERARCHICAL:
                    data["subspaces"] = {}
                    for name, subspace in self.subspaces.items():
                        # Create subspace filepath
                        subspace_filepath = f"{filepath}_subspace_{name}"
                        # Save subspace
                        subspace.save_to_file(subspace_filepath)
                        # Store reference to subspace file
                        data["subspaces"][name] = subspace_filepath
                
                # For adaptive dimensionality, save dimension importance
                if self.dimensionality_type == DimensionalityType.ADAPTIVE:
                    if self.dimension_importance is not None:
                        data["dimension_importance"] = self.dimension_importance.tolist()
                    data["adaptive_dimensions"] = self.adaptive_dimensions
                
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                
                return True
                
        except Exception as e:
            print(f"Error saving semantic field: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HyperdimensionalSemanticField':
        """Load a semantic field from a file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Create field with same parameters
            field = cls(
                dimensions=data["dimensions"],
                encoding_method=EncodingMethod[data["encoding_method"]],
                dimensionality_type=DimensionalityType[data["dimensionality_type"]]
            )
            
            # Restore statistics
            field.stats = data["stats"]
            field.label_index = data["label_index"]
            
            # Restore relation index
            for relation_name, relation_dict in data["relation_index"].items():
                relation = SemanticRelation[relation_name]
                field.relation_index[relation] = relation_dict
            
            # Restore vectors
            for vector_id, vector_data in data["vectors"].items():
                vector = SemanticVector.from_dict(vector_data)
                field.vectors[vector_id] = vector
            
            # For hierarchical dimensionality, load subspaces
            if field.dimensionality_type == DimensionalityType.HIERARCHICAL:
                if "subspaces" in data:
                    for name, subspace_filepath in data["subspaces"].items():
                        # Check if subspace file exists
                        if os.path.exists(subspace_filepath):
                            # Load subspace
                            subspace = cls.load_from_file(subspace_filepath)
                            field.subspaces[name] = subspace
            
            # For adaptive dimensionality, restore dimension importance
            if field.dimensionality_type == DimensionalityType.ADAPTIVE:
                if "dimension_importance" in data:
                    field.dimension_importance = np.array(data["dimension_importance"])
                if "adaptive_dimensions" in data:
                    field.adaptive_dimensions = data["adaptive_dimensions"]
            
            return field
            
        except Exception as e:
            print(f"Error loading semantic field: {e}")
            return cls()  # Return new field as fallback
    
    def _initialize_dimensionality(self) -> None:
        """Initialize based on dimensionality type"""
        if self.dimensionality_type == DimensionalityType.ADAPTIVE:
            # Initialize uniform importance
            self.dimension_importance = np.ones(self.dimensions) / self.dimensions
            self.adaptive_dimensions = list(range(self.dimensions))
            
        elif self.dimensionality_type == DimensionalityType.FRACTAL:
            # For fractal dimensionality, ensure dimensions is a power of base
            base = 2
            power = int(math.log(self.dimensions, base))
            self.dimensions = base ** power
    
    def _calculate_similarity(self, vec1: Union[np.ndarray, sparse.csr_matrix],
                           vec2: Union[np.ndarray, sparse.csr_matrix],
                           encoding: EncodingMethod) -> float:
        """Calculate similarity between two vectors based on encoding"""
        # Handle sparse matrices
        if sparse.issparse(vec1) and sparse.issparse(vec2):
            if encoding == EncodingMethod.BINARY:
                # Jaccard similarity for binary sparse vectors
                intersection = vec1.multiply(vec2).sum()
                union = vec1.sum() + vec2.sum() - intersection
                return float(intersection / max(union, 1))
                
            elif encoding == EncodingMethod.BIPOLAR:
                # Convert to dense for hamming similarity
                vec1_dense = vec1.toarray().flatten()
                vec2_dense = vec2.toarray().flatten()
                matches = np.sum(vec1_dense == vec2_dense)
                return float(matches / len(vec1_dense))
                
            else:
                # Cosine similarity for other sparse vectors
                return float(vec1.dot(vec2.T).toarray()[0, 0] / 
                           (np.sqrt(vec1.power(2).sum()) * np.sqrt(vec2.power(2).sum())))
        
        # Handle case where one is sparse and one is dense
        if sparse.issparse(vec1) or sparse.issparse(vec2):
            if sparse.issparse(vec1):
                vec1 = vec1.toarray().flatten()
            else:
                vec2 = vec2.toarray().flatten()
        
        # Different similarity metrics based on encoding
        if encoding == EncodingMethod.BINARY:
            # Jaccard similarity for binary vectors
            intersection = np.sum(np.logical_and(vec1 > 0, vec2 > 0))
            union = np.sum(np.logical_or(vec1 > 0, vec2 > 0))
            return float(intersection / max(union, 1))
            
        elif encoding == EncodingMethod.BIPOLAR:
            # Normalized hamming similarity for bipolar vectors
            matches = np.sum(vec1 == vec2)
            return float(matches / len(vec1))
            
        elif encoding == EncodingMethod.HOLOGRAPHIC:
            # For complex vectors, use absolute value of hermitian product
            hermitian = np.abs(np.vdot(vec1, vec2))
            return float(hermitian / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            
        elif encoding == EncodingMethod.CIRCULAR:
            # For circular vectors, use angular similarity
            angular_diff = np.minimum(np.abs(vec1 - vec2), 2*np.pi - np.abs(vec1 - vec2))
            return float(1 - np.mean(angular_diff / np.pi))
            
        else:
            # Default to cosine similarity
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _execute_operation(self, operation: SemanticOperation,
                        vectors: List[SemanticVector],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a semantic vector operation"""
        if operation == SemanticOperation.BIND:
            return self._operation_bind(vectors, params)
            
        elif operation == SemanticOperation.BUNDLE:
            return self._operation_bundle(vectors, params)
            
        elif operation == SemanticOperation.PERMUTE:
            return self._operation_permute(vectors, params)
            
        elif operation == SemanticOperation.TRANSFORM:
            return self._operation_transform(vectors, params)
            
        elif operation == SemanticOperation.QUERY:
            return self._operation_query(vectors, params)
            
        elif operation == SemanticOperation.ANALOGY:
            return self._operation_analogy(vectors, params)
            
        elif operation == SemanticOperation.COMPOSE:
            return self._operation_compose(vectors, params)
            
        elif operation == SemanticOperation.NEGATE:
            return self._operation_negate(vectors, params)
            
        elif operation == SemanticOperation.PROJECT:
            return self._operation_project(vectors, params)
            
        elif operation == SemanticOperation.ENTANGLE:
            return self._operation_entangle(vectors, params)
            
        else:
            return {"success": False, "error": "Unknown operation"}
    
    def _operation_bind(self, vectors: List[SemanticVector],
                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Binding operation - creates associative connections between vectors"""
        if len(vectors) < 2:
            return {"success": False, "error": "Need at least 2 vectors for binding"}
            
        # Get encoding from first vector
        encoding = vectors[0].encoding
        
        # Get vector data
        vector_data = [v.get_data() for v in vectors]
        if any(d is None for d in vector_data):
            return {"success": False, "error": "Missing vector data"}
        
        # Perform binding based on encoding
        result = vector_data[0]
        for data in vector_data[1:]:
            result = self._bind_vectors(result, data, encoding)
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "bind",
            "operands": [v.vector_id for v in vectors],
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relations to operands
        if vector_id:
            for vector in vectors:
                self.relate_vectors(vector_id, vector.vector_id, SemanticRelation.HOLONYM)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "encoding": encoding.name
        }
    
    def _operation_bundle(self, vectors: List[SemanticVector],
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Bundling operation - creates a superposition of vectors"""
        if not vectors:
            return {"success": False, "error": "No vectors provided"}
            
        # Get encoding from first vector
        encoding = vectors[0].encoding
        
        # Get vector data and weights
        weights = params.get("weights")
        if weights is None:
            # Default to equal weights
            weights = [1.0] * len(vectors)
        elif len(weights) != len(vectors):
            return {"success": False, "error": "Number of weights doesn't match vectors"}
        
        vector_data = [v.get_data() for v in vectors]
        if any(d is None for d in vector_data):
            return {"success": False, "error": "Missing vector data"}
        
        # Perform weighted bundle
        result = None
        total_weight = 0.0
        
        for i, data in enumerate(vector_data):
            weight = weights[i]
            total_weight += abs(weight)
            
            if result is None:
                if isinstance(data, sparse.csr_matrix):
                    result = weight * data.copy()
                else:
                    result = weight * data.copy()
            else:
                result += weight * data
        
        # Normalize if needed
        if params.get("normalize", True) and total_weight > 0:
            if isinstance(result, sparse.csr_matrix):
                result = result / total_weight
            else:
                result = result / total_weight
        
        # Special handling for binary/bipolar vectors
        if encoding == EncodingMethod.BINARY:
            # Threshold for binary vectors
            threshold = params.get("threshold", 0.5)
            if isinstance(result, sparse.csr_matrix):
                # For sparse matrices, convert to dense, threshold, then back to sparse
                result_dense = result.toarray()[0]
                result_binary = (result_dense > threshold).astype(np.int8)
                indices = np.where(result_binary > 0)[0]
                data = np.ones(len(indices), dtype=np.int8)
                result = sparse.csr_matrix((data, indices, np.array([0, len(indices)])),
                                         shape=(1, len(result_dense)))
            else:
                result = (result > threshold).astype(np.int8)
                
        elif encoding == EncodingMethod.BIPOLAR:
            # Sign threshold for bipolar vectors
            if isinstance(result, sparse.csr_matrix):
                # Convert to dense for sign operation
                result = np.sign(result.toarray()[0]).astype(np.int8)
            else:
                result = np.sign(result).astype(np.int8)
                # Replace zeros with random -1/1
                zero_indices = np.where(result == 0)[0]
                result[zero_indices] = np.random.choice([-1, 1], size=len(zero_indices))
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "bundle",
            "operands": [v.vector_id for v in vectors],
            "weights": weights,
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relations to operands
        if vector_id:
            for i, vector in enumerate(vectors):
                relation = (SemanticRelation.HYPERNYM if weights[i] > 0 
                          else SemanticRelation.ANTONYM)
                self.relate_vectors(vector_id, vector.vector_id, relation)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "encoding": encoding.name
        }
    
    def _operation_permute(self, vectors: List[SemanticVector],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Permutation operation - shifts elements in the vector"""
        if len(vectors) != 1:
            return {"success": False, "error": "Permutation requires exactly one vector"}
            
        vector = vectors[0]
        data = vector.get_data()
        if data is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Get permutation parameters
        shift = params.get("shift", 1)
        direction = params.get("direction", "right")
        
        # Apply permutation
        if isinstance(data, sparse.csr_matrix):
            # Convert to dense for permutation
            dense_data = data.toarray()[0]
            if direction == "right":
                result = np.roll(dense_data, shift)
            else:
                result = np.roll(dense_data, -shift)
                
            # Convert back to sparse if binary encoding
            if vector.encoding == EncodingMethod.BINARY:
                indices = np.where(result > 0)[0]
                values = np.ones(len(indices), dtype=np.int8)
                result = sparse.csr_matrix((values, indices, np.array([0, len(indices)])),
                                         shape=(1, len(dense_data)))
        else:
            # Direct permutation on dense array
            if direction == "right":
                result = np.roll(data, shift)
            else:
                result = np.roll(data, -shift)
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "permute",
            "operand": vector.vector_id,
            "shift": shift,
            "direction": direction,
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relation to operand
        if vector_id:
            self.relate_vectors(vector_id, vector.vector_id, SemanticRelation.FUNCTIONAL)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "encoding": vector.encoding.name,
            "shift": shift,
            "direction": direction
        }
    
    def _operation_transform(self, vectors: List[SemanticVector],
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform operation - applies a linear transformation to a vector"""
        if len(vectors) != 1:
            return {"success": False, "error": "Transform requires exactly one vector"}
            
        vector = vectors[0]
        data = vector.get_data()
        if data is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Get transformation parameters
        transform_type = params.get("transform_type", "rotate")
        
        # Convert sparse to dense for transformation
        if isinstance(data, sparse.csr_matrix):
            data = data.toarray()[0]
        
        # Apply transformation
        if transform_type == "rotate":
            # Rotate in first two dimensions
            angle = params.get("angle", np.pi/4)  # Default to 45 degrees
            
            # Need at least 2D for rotation
            if len(data) < 2:
                return {"success": False, "error": "Vector dimension too small for rotation"}
                
            # Create rotation matrix
            rot_matrix = np.eye(len(data))
            rot_matrix[0, 0] = np.cos(angle)
            rot_matrix[0, 1] = -np.sin(angle)
            rot_matrix[1, 0] = np.sin(angle)
            rot_matrix[1, 1] = np.cos(angle)
            
            # Apply rotation
            result = rot_matrix @ data
            
        elif transform_type == "scale":
            # Scale vector
            scale = params.get("scale", 2.0)
            result = data * scale
            
        elif transform_type == "reflect":
            # Reflect across axis
            axis = params.get("axis", 0)
            if axis >= len(data):
                return {"success": False, "error": "Reflection axis out of bounds"}
                
            # Create reflection matrix
            refl_matrix = np.eye(len(data))
            refl_matrix[axis, axis] = -1
            
            # Apply reflection
            result = refl_matrix @ data
            
        else:
            return {"success": False, "error": "Unknown transformation type"}
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "transform",
            "operand": vector.vector_id,
            "transform_type": transform_type,
            "params": {k: v for k, v in params.items() 
                      if k not in ["transform_type", "label"]},
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relation to operand
        if vector_id:
            self.relate_vectors(vector_id, vector.vector_id, SemanticRelation.FUNCTIONAL)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "encoding": vector.encoding.name,
            "transform_type": transform_type
        }
    
    def _operation_query(self, vectors: List[SemanticVector],
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Query operation - finds vectors similar to the query"""
        if len(vectors) != 1:
            return {"success": False, "error": "Query requires exactly one vector"}
            
        vector = vectors[0]
        data = vector.get_data()
        if data is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Get query parameters
        top_k = params.get("top_k", 10)
        threshold = params.get("threshold", 0.0)
        exclude_self = params.get("exclude_self", True)
        
        # Find similar vectors
        similarities = []
        
        for vector_id, other_vector in self.vectors.items():
            # Skip self if requested
            if exclude_self and vector_id == vector.vector_id:
                continue
                
            other_data = other_vector.get_data()
            if other_data is None:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(data, other_data, vector.encoding)
            
            if similarity >= threshold:
                similarities.append((vector_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for vector_id, similarity in top_results:
            other_vector = self.vectors[vector_id]
            results.append({
                "vector_id": vector_id,
                "label": other_vector.label,
                "similarity": similarity
            })
        
        return {
            "success": True,
            "query_vector": vector.vector_id,
            "results": results,
            "count": len(results)
        }
    
    def _operation_analogy(self, vectors: List[SemanticVector],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Analogy operation - solves analogy problems a:b::c:?"""
        if len(vectors) != 3:
            return {"success": False, "error": "Analogy requires exactly three vectors (a, b, c)"}
            
        # Get vectors
        a_vector, b_vector, c_vector = vectors
        
        # Get vector data
        a_data = a_vector.get_data()
        b_data = b_vector.get_data()
        c_data = c_vector.get_data()
        
        if any(d is None for d in [a_data, b_data, c_data]):
            return {"success": False, "error": "Missing vector data"}
        
        # Get parameters
        top_k = params.get("top_k", 5)
        
        # Execute analogy based on encoding method
        target = None
        
        # For binary and bipolar vectors
        if a_vector.encoding in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
            # a:b as c:? means ? = c XOR a XOR b
            if isinstance(a_data, sparse.csr_matrix):
                # For sparse binary vectors
                target = self._sparse_xor(self._sparse_xor(c_data, a_data), b_data)
            else:
                # For dense binary/bipolar vectors
                if a_vector.encoding == EncodingMethod.BINARY:
                    target = (c_data ^ a_data ^ b_data).astype(a_data.dtype)
                else:  # BIPOLAR
                    # For bipolar, multiplication is equivalent to XOR
                    target = c_data * a_data * b_data
        
        # For holographic vectors
        elif a_vector.encoding == EncodingMethod.HOLOGRAPHIC:
            # Holographic analogies using circular convolution
            # ? = c ⊗ a⁻¹ ⊗ b where ⊗ is circular convolution
            a_inv = np.conj(a_data)  # Inverse by complex conjugate
            target = self._circular_convolution(self._circular_convolution(c_data, a_inv), b_data)
        
        # For circular vectors
        elif a_vector.encoding == EncodingMethod.CIRCULAR:
            # Circular vectors use addition/subtraction modulo 2π
            target = (c_data - a_data + b_data) % (2*np.pi)
        
        # For real-valued or complex vectors
        else:
            # Use the vector offset method: ? = c + (b - a)
            target = c_data + (b_data - a_data)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(target)
            if norm > 0:
                target = target / norm
        
        # Find most similar vectors to target
        similarities = []
        
        for vector_id, vector in self.vectors.items():
            # Skip input vectors
            if vector_id in [a_vector.vector_id, b_vector.vector_id, c_vector.vector_id]:
                continue
                
            vector_data = vector.get_data()
            if vector_data is None:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(target, vector_data, a_vector.encoding)
            similarities.append((vector_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for vector_id, similarity in top_results:
            other_vector = self.vectors[vector_id]
            results.append({
                "vector_id": vector_id,
                "label": other_vector.label,
                "similarity": similarity
            })
        
        # Create target vector if requested
        target_vector_id = None
        if params.get("create_target", False):
            label = params.get("label", f"analogy_{a_vector.label}:{b_vector.label}::{c_vector.label}")
            metadata = {
                "operation": "analogy",
                "a": a_vector.vector_id,
                "b": b_vector.vector_id,
                "c": c_vector.vector_id,
                "creation_type": "operation"
            }
            
            target_vector_id = self.create_vector(label=label, data=target, metadata=metadata)
        
        return {
            "success": True,
            "a": a_vector.vector_id,
            "b": b_vector.vector_id,
            "c": c_vector.vector_id,
            "results": results,
            "target_vector_id": target_vector_id
        }
    
    def _operation_compose(self, vectors: List[SemanticVector],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Compositional semantics operation"""
        if len(vectors) < 2:
            return {"success": False, "error": "Composition requires at least two vectors"}
            
        # Get composition method
        method = params.get("method", "weighted_sum")
        
        # Get weights if provided, otherwise use equal weights
        weights = params.get("weights")
        if weights is None:
            weights = [1.0] * len(vectors)
        elif len(weights) != len(vectors):
            return {"success": False, "error": "Number of weights doesn't match vectors"}
        
        # Create components dictionary
        components = {vectors[i].vector_id: weights[i] for i in range(len(vectors))}
        
        # Create compositional vector
        label = params.get("label")
        vector_id = self.create_compositional_vector(components, label, method)
        
        if not vector_id:
            return {"success": False, "error": "Failed to create compositional vector"}
        
        return {
            "success": True,
            "vector_id": vector_id,
            "method": method,
            "components": [{"id": v.vector_id, "weight": weights[i]} 
                         for i, v in enumerate(vectors)]
        }
    
    def _operation_negate(self, vectors: List[SemanticVector],
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Negation operation - creates semantic opposite"""
        if len(vectors) != 1:
            return {"success": False, "error": "Negation requires exactly one vector"}
            
        vector = vectors[0]
        data = vector.get_data()
        if data is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Negate based on encoding
        if vector.encoding == EncodingMethod.BINARY:
            # For binary, flip bits
            if isinstance(data, sparse.csr_matrix):
                # For sparse binary, complement = all indices not in original
                dense = data.toarray()[0]
                negated = 1 - dense
                indices = np.where(negated > 0)[0]
                values = np.ones(len(indices), dtype=np.int8)
                result = sparse.csr_matrix((values, indices, np.array([0, len(indices)])),
                                         shape=(1, len(dense)))
            else:
                result = 1 - data
                
        elif vector.encoding == EncodingMethod.BIPOLAR:
            # For bipolar, negate all values
            if isinstance(data, sparse.csr_matrix):
                result = data.copy()
                result.data = -result.data
            else:
                result = -data
                
        elif vector.encoding == EncodingMethod.HOLOGRAPHIC:
            # For holographic, take complex conjugate
            result = np.conj(data)
            
        elif vector.encoding == EncodingMethod.CIRCULAR:
            # For circular, add π (mod 2π)
            result = (data + np.pi) % (2*np.pi)
            
        else:
            # For other encodings, negate vector
            result = -data
            
            # Normalize if real-valued
            if not np.iscomplex(data).any():
                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm
        
        # Create result vector
        label = params.get("label", f"not_{vector.label}" if vector.label else None)
        metadata = {
            "operation": "negate",
            "operand": vector.vector_id,
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create antonym relation
        if vector_id:
            self.relate_vectors(vector_id, vector.vector_id, SemanticRelation.ANTONYM)
            self.relate_vectors(vector.vector_id, vector_id, SemanticRelation.ANTONYM)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "original_vector": vector.vector_id,
            "encoding": vector.encoding.name
        }
    
    def _operation_project(self, vectors: List[SemanticVector],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Projection operation - projects one vector onto another's subspace"""
        if len(vectors) != 2:
            return {"success": False, "error": "Projection requires exactly two vectors"}
            
        # Get source and target vectors
        source = vectors[0]
        target = vectors[1]
        
        source_data = source.get_data()
        target_data = target.get_data()
        
        if source_data is None or target_data is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Convert sparse to dense for projection
        if isinstance(source_data, sparse.csr_matrix):
            source_data = source_data.toarray()[0]
        if isinstance(target_data, sparse.csr_matrix):
            target_data = target_data.toarray()[0]
        
        # Project source onto target
        if source.encoding in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
            # For binary/bipolar, use dot product as similarity measure
            similarity = np.dot(source_data, target_data) / (np.linalg.norm(target_data)**2)
            result = similarity * target_data
            
            # Convert back to binary/bipolar
            if source.encoding == EncodingMethod.BINARY:
                threshold = params.get("threshold", 0.5)
                result = (result > threshold).astype(np.int8)
            else:  # BIPOLAR
                result = np.sign(result).astype(np.int8)
                
        elif source.encoding == EncodingMethod.HOLOGRAPHIC or source.encoding == EncodingMethod.QUANTUM:
            # For complex vectors, use hermitian product
            similarity = np.vdot(source_data, target_data) / np.vdot(target_data, target_data)
            result = similarity * target_data
            
        elif source.encoding == EncodingMethod.CIRCULAR:
            # For circular vectors, use angular similarity
            # Project by minimizing angular distance
            angular_diff = (source_data - target_data) % (2*np.pi)
            angular_diff = np.minimum(angular_diff, 2*np.pi - angular_diff)
            similarity = 1 - np.mean(angular_diff / np.pi)
            
            result = target_data.copy()
            
        else:
            # For real vectors, standard projection
            # proj_u(v) = (v·u / u·u) * u
            similarity = np.dot(source_data, target_data) / np.dot(target_data, target_data)
            result = similarity * target_data
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "project",
            "source": source.vector_id,
            "target": target.vector_id,
            "similarity": float(similarity) if isinstance(similarity, (int, float, np.number)) else float(np.real(similarity)),
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relations
        if vector_id:
            self.relate_vectors(vector_id, source.vector_id, SemanticRelation.FUNCTIONAL)
            self.relate_vectors(vector_id, target.vector_id, SemanticRelation.FUNCTIONAL)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "source_vector": source.vector_id,
            "target_vector": target.vector_id,
            "similarity": float(similarity) if isinstance(similarity, (int, float, np.number)) else float(np.real(similarity)),
            "encoding": source.encoding.name
        }
    
    def _operation_entangle(self, vectors: List[SemanticVector],
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Entanglement operation - creates a quantum-like entangled state"""
        if len(vectors) != 2:
            return {"success": False, "error": "Entanglement requires exactly two vectors"}
            
        # Get vectors
        vector1 = vectors[0]
        vector2 = vectors[1]
        
        data1 = vector1.get_data()
        data2 = vector2.get_data()
        
        if data1 is None or data2 is None:
            return {"success": False, "error": "Missing vector data"}
        
        # Convert sparse to dense
        if isinstance(data1, sparse.csr_matrix):
            data1 = data1.toarray()[0]
        if isinstance(data2, sparse.csr_matrix):
            data2 = data2.toarray()[0]
        
        # Create entangled state based on encoding
        if vector1.encoding in [EncodingMethod.BINARY, EncodingMethod.BIPOLAR]:
            # For binary/bipolar, use XOR for entanglement
            if vector1.encoding == EncodingMethod.BINARY:
                result = np.logical_xor(data1 > 0, data2 > 0).astype(np.int8)
            else:
                result = data1 * data2  # Multiplication is XOR for {-1, 1}
                
        elif vector1.encoding == EncodingMethod.HOLOGRAPHIC or vector1.encoding == EncodingMethod.QUANTUM:
            # For complex vectors, create a tensor-like representation
            # Simplified tensor product representation
            n = len(data1)
            result = np.zeros(n, dtype=complex)
            
            # Create entangled state using phase correlations
            for i in range(n):
                phase1 = np.angle(data1[i]) if np.abs(data1[i]) > 0 else 0
                phase2 = np.angle(data2[i]) if np.abs(data2[i]) > 0 else 0
                
                # Create correlated phases
                if i % 2 == 0:
                    # Even indices: phases add
                    result[i] = np.exp(1j * (phase1 + phase2))
                else:
                    # Odd indices: phases subtract
                    result[i] = np.exp(1j * (phase1 - phase2))
            
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
                
        elif vector1.encoding == EncodingMethod.CIRCULAR:
            # For circular vectors, create correlated angles
            result = np.zeros_like(data1)
            for i in range(len(data1)):
                if i % 2 == 0:
                    # Even indices: average angles
                    result[i] = ((data1[i] + data2[i]) / 2) % (2*np.pi)
                else:
                    # Odd indices: difference of angles
                    result[i] = np.abs(data1[i] - data2[i]) % (2*np.pi)
                    
        else:
            # For real vectors, create a combined representation
            # Use Hadamard product for entanglement-like effects
            result = data1 * data2
            
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
        
        # Create result vector
        label = params.get("label")
        metadata = {
            "operation": "entangle",
            "vector1": vector1.vector_id,
            "vector2": vector2.vector_id,
            "creation_type": "operation"
        }
        
        vector_id = self.create_vector(label=label, data=result, metadata=metadata)
        
        # Create relations
        if vector_id:
            relation_type = SemanticRelation.FUNCTIONAL
            self.relate_vectors(vector_id, vector1.vector_id, relation_type)
            self.relate_vectors(vector_id, vector2.vector_id, relation_type)
        
        return {
            "success": True,
            "vector_id": vector_id,
            "vector1": vector1.vector_id,
            "vector2": vector2.vector_id,
            "encoding": vector1.encoding.name
        }
    
    def _bind_vectors(self, vec1: Union[np.ndarray, sparse.csr_matrix],
                   vec2: Union[np.ndarray, sparse.csr_matrix],
                   encoding: EncodingMethod) -> Union[np.ndarray, sparse.csr_matrix]:
        """Bind two vectors based on encoding method"""
        # Handle sparse matrices
        if sparse.issparse(vec1) and sparse.issparse(vec2):
            if encoding == EncodingMethod.BINARY:
                # XOR for binary sparse vectors
                return self._sparse_xor(vec1, vec2)
            else:
                # Convert to dense for other operations
                vec1 = vec1.toarray()[0]
                vec2 = vec2.toarray()[0]
        elif sparse.issparse(vec1):
            vec1 = vec1.toarray()[0]
        elif sparse.issparse(vec2):
            vec2 = vec2.toarray()[0]
        
        # Binding operation based on encoding
        if encoding == EncodingMethod.BINARY:
            # XOR for binary vectors
            return np.logical_xor(vec1, vec2).astype(np.int8)
            
        elif encoding == EncodingMethod.BIPOLAR:
            # Element-wise multiplication for bipolar vectors
            return (vec1 * vec2).astype(np.int8)
            
        elif encoding == EncodingMethod.HOLOGRAPHIC:
            # Circular convolution for holographic representations
            return self._circular_convolution(vec1, vec2)
            
        elif encoding == EncodingMethod.CIRCULAR:
            # Addition modulo 2π for circular vectors
            return (vec1 + vec2) % (2*np.pi)
            
        else:
            # Hadamard product for other vectors
            result = vec1 * vec2
            
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
                
            return result
    
    def _sparse_xor(self, a: sparse.csr_matrix, b: sparse.csr_matrix) -> sparse.csr_matrix:
        """Compute XOR of two sparse binary matrices"""
        # Convert to boolean arrays
        a_dense = a.toarray()[0].astype(bool)
        b_dense = b.toarray()[0].astype(bool)
        
        # Calculate XOR
        result_dense = np.logical_xor(a_dense, b_dense)
        
        # Convert back to sparse
        indices = np.where(result_dense)[0]
        data = np.ones(len(indices), dtype=np.int8)
        result = sparse.csr_matrix((data, indices, np.array([0, len(indices)])),
                                 shape=a.shape)
        
        return result
    
    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute circular convolution using FFT"""
        # FFT-based circular convolution: ifft(fft(a) * fft(b))
        n = len(a)
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        
        # Element-wise product in frequency domain
        fft_prod = fft_a * fft_b
        
        # Back to time domain
        result = np.fft.ifft(fft_prod)
        
        # Return real part if result should be real
        if not np.iscomplex(a).any() and not np.iscomplex(b).any():
            result = np.real(result)
            
        return result
    
    def _update_vector_confidence_stats(self) -> None:
        """Update vector confidence statistics"""
        if not self.vectors:
            self.stats["avg_vector_confidence"] = 1.0
            return
            
        total_confidence = sum(v.confidence for v in self.vectors.values())
        self.stats["avg_vector_confidence"] = total_confidence / len(self.vectors)

def run_example():
    """
    Run a comprehensive demonstration of the hyperdimensional semantic field generator.
    
    This demonstration showcases the key features of the semantic field system:
    
    1. **Field Initialization**: Creates a high-dimensional semantic field with binary encoding
    2. **Vector Creation**: Demonstrates creation of semantic vectors for various concepts
    3. **Relation Establishment**: Shows how to create hierarchical and synonymous relations
    4. **Compositional Vectors**: Illustrates creation of complex concepts from simpler ones
    5. **Semantic Operations**: Demonstrates binding and other vector operations
    6. **Similarity Search**: Shows how to find semantically similar concepts
    7. **Analogical Reasoning**: Performs analogy completion tasks
    8. **Path Finding**: Finds semantic paths between concepts
    9. **Statistics**: Reports field metrics and performance data
    
    The example uses a practical semantic network with concepts like animals, pets,
    and homes to demonstrate real-world applicability of hyperdimensional computing
    principles in semantic representation.
    
    Returns:
        None: Prints comprehensive demonstration output to console
        
    Example Output:
        Creates semantic field with 10,000 dimensions
        Establishes relations between concepts like cat, dog, animal, pet
        Performs analogical reasoning: "cat is to animal as dog is to ?"
        Finds semantic paths between related concepts
        Reports field statistics and performance metrics
    """
    print(f"Current Date/Time: 2025-08-02 03:44:26")
    print(f"User: Shriram-2005")
    
    print("\n===== Hyperdimensional Semantic Field Generator Example =====")
    
    # Create semantic field
    print("\nInitializing Hyperdimensional Semantic Field...")
    field = HyperdimensionalSemanticField(
        dimensions=10000,
        encoding_method=EncodingMethod.BINARY,
        dimensionality_type=DimensionalityType.FIXED
    )
    
    print(f"Dimensionality: {field.dimensions}")
    print(f"Encoding method: {field.encoding_method.name}")
    
    # Create some semantic vectors
    print("\nCreating semantic vectors...")
    
    cat_id = field.create_vector(label="cat")
    dog_id = field.create_vector(label="dog")
    animal_id = field.create_vector(label="animal")
    pet_id = field.create_vector(label="pet")
    house_id = field.create_vector(label="house")
    home_id = field.create_vector(label="home")
    
    print(f"Created vectors: cat, dog, animal, pet, house, home")
    
    # Create semantic relations
    print("\nEstablishing semantic relations...")
    
    field.relate_vectors(animal_id, cat_id, SemanticRelation.HYPERNYM)  # animal is hypernym of cat
    field.relate_vectors(animal_id, dog_id, SemanticRelation.HYPERNYM)  # animal is hypernym of dog
    field.relate_vectors(pet_id, cat_id, SemanticRelation.HYPERNYM)     # pet is hypernym of cat
    field.relate_vectors(pet_id, dog_id, SemanticRelation.HYPERNYM)     # pet is hypernym of dog
    field.relate_vectors(house_id, home_id, SemanticRelation.SYNONYM)   # house is synonym of home
    
    print(f"Established hypernym and synonym relations")
    
    # Find related concepts
    print("\nFinding related concepts...")
    
    animal_hyponyms = field.find_related_vectors(animal_id, SemanticRelation.HYPONYM)
    print(f"Concepts under 'animal': {len(animal_hyponyms)} found")
    for hyponym_id in animal_hyponyms:
        hyponym = field.get_vector(hyponym_id)
        print(f"  → {hyponym.label}")
    
    # Create compositional vector
    print("\nCreating compositional vector...")
    
    components = {pet_id: 0.7, house_id: 0.3}
    pet_house_id = field.create_compositional_vector(components, label="pet_house", method="weighted_sum")
    
    print(f"Created compositional concept 'pet_house'")
    
    # Perform semantic operations
    print("\nPerforming semantic operations...")
    
    # Bind cat and home
    bind_result = field.perform_operation(
        operation=SemanticOperation.BIND,
        operands=[cat_id, home_id],
        operation_params={"label": "cat_home"}
    )
    
    cat_home_id = bind_result.get("vector_id")
    print(f"Bound 'cat' and 'home' to create 'cat_home'")
    
    # Find similar vectors
    print("\nFinding similar vectors to 'pet'...")
    
    similar_to_pet = field.find_similar_vectors(pet_id, top_k=3)
    print(f"Found {len(similar_to_pet)} similar vectors:")
    
    for vector_id, similarity in similar_to_pet:
        vector = field.get_vector(vector_id)
        print(f"  → {vector.label}: similarity = {similarity:.4f}")
    
    # Execute analogy
    print("\nExecuting analogy: cat is to animal as dog is to ?")
    
    analogy_results = field.execute_analogy("cat", "animal", "dog", top_k=2)
    
    print(f"Analogy results:")
    for vector_id, similarity in analogy_results:
        vector = field.get_vector(vector_id)
        print(f"  → {vector.label}: similarity = {similarity:.4f}")
    
    # Find semantic path
    print("\nFinding semantic path from 'cat' to 'home'...")
    
    path = field.find_semantic_path(cat_id, home_id)
    
    if path:
        print(f"Found path with {len(path)} steps:")
        for step in path:
            from_vector = field.get_vector(step["from"])
            to_vector = field.get_vector(step["to"])
            relation = step["relation"]
            print(f"  → {from_vector.label} --[{relation}]--> {to_vector.label}")
    else:
        print("No path found")
    
    # Get field statistics
    stats = field.stats
    print("\nSemantic field statistics:")
    print(f"  Total vectors: {stats['vector_count']}")
    print(f"  Operations performed: {stats['operation_count']}")
    print(f"  Average vector confidence: {stats['avg_vector_confidence']:.4f}")
    
    print("\nHyperdimensional Semantic Field Generator demonstration complete!")
    print("The system successfully implements advanced semantic representations using hyperdimensional computing principles.")


# Module exports for public API
__all__ = [
    # Core classes
    'HyperdimensionalSemanticField',
    'SemanticVector',
    
    # Enumerations
    'EncodingMethod',
    'SemanticOperation', 
    'SemanticRelation',
    'DimensionalityType',
    
    # Demonstration function
    'run_example'
]


if __name__ == "__main__":
    run_example()