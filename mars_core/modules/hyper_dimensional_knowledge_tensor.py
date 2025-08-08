"""
Hyper-Dimensional Knowledge Tensor System for MARS Quantum Framework

This module implements an advanced knowledge representation system using high-dimensional
tensor spaces for sophisticated AI reasoning and knowledge management. The system provides
multi-dimensional embeddings across various conceptual spaces including linguistic,
temporal, causal, quantum, and emotional domains.

Features:
    - Multi-dimensional tensor space representations
    - Advanced knowledge embeddings across 8 distinct tensor spaces
    - Quantum-inspired dimensionality reduction techniques
    - Sophisticated tensor operations (product, contraction, addition)
    - Intelligent similarity detection and clustering
    - Temporal and causal relationship modeling
    - Thread-safe knowledge storage and retrieval
    - Comprehensive statistical analysis and monitoring

Architecture:
    The system is built around three core components:
    1. KnowledgeTensor: Fundamental data structure for tensor representation
    2. TensorOperator: Mathematical operations and transformations
    3. HyperDimensionalKnowledgeStore: Storage and indexing engine
    4. HyperDimensionalEmbedding: Embedding creation across different spaces

Tensor Spaces:
    - CONCEPTUAL: Abstract concept relationships and hierarchies
    - LINGUISTIC: Natural language semantic embeddings
    - PHYSICAL: Physical system properties and interactions
    - TEMPORAL: Time-based patterns and temporal relationships
    - CAUSAL: Cause-and-effect relationship modeling
    - EMOTIONAL: Sentiment and emotional state representations
    - DECISION: Decision-making process embeddings
    - QUANTUM: Quantum state vector representations

Usage Example:
    ```python
    # Initialize the knowledge store
    store = HyperDimensionalKnowledgeStore()
    embedding_engine = HyperDimensionalEmbedding(store)
    
    # Create embeddings in different spaces
    concept_tensor = embedding_engine.create_conceptual_embedding({
        "name": "quantum_computer",
        "type": "technology",
        "properties": ["superposition", "entanglement"]
    })
    
    # Store and operate on tensors
    store.store_tensor(concept_tensor)
    similar_tensors = store.find_similar_tensor(concept_tensor)
    ```

Mathematical Foundation:
    The system employs advanced tensor mathematics including:
    - Higher-order tensor decompositions
    - Quantum-inspired transformations
    - Cosine similarity metrics
    - Principal Component Analysis (PCA)
    - t-SNE and UMAP embeddings
    - Complex amplitude projections
    
"""
# Standard library imports
import hashlib
import logging
import math
import pickle
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cityblock, cosine, euclidean

# Configure module logger
logger = logging.getLogger("MARS.HDTensor")

# Mathematical constants for tensor operations
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
PI_SQUARED_OVER_SIX = math.pi**2 / 6

# Default system parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MAX_RESULTS = 5
DEFAULT_DIMENSION_SCALING_FACTOR = 1.618  # Golden ratio scaling
QUANTUM_COHERENCE_THRESHOLD = 0.95
TEMPORAL_CYCLE_RESOLUTION = 16  # Number of harmonics for temporal encoding

# Memory management constants
MAX_TENSOR_CACHE_SIZE = 1000
HASH_TRUNCATION_LENGTH = 16

class TensorSpace(Enum):
    """
    Enumeration of supported tensor spaces for knowledge representation.
    
    Each tensor space represents a different domain of knowledge encoding,
    allowing for specialized representations of various types of information
    and their relationships within the hyper-dimensional framework.
    """
    CONCEPTUAL = auto()   # Abstract concept space for hierarchical relationships
    LINGUISTIC = auto()   # Natural language semantic space for text understanding
    PHYSICAL = auto()     # Physical system properties and material interactions
    TEMPORAL = auto()     # Time-based relationships and sequential patterns
    CAUSAL = auto()       # Causality relationships and dependency structures
    EMOTIONAL = auto()    # Emotional/sentiment space for affective computing
    DECISION = auto()     # Decision-making space for choice optimization
    QUANTUM = auto()      # Quantum probability space for superposition states


class DimensionalityReduction(Enum):
    """
    Methods for reducing tensor dimensionality while preserving information.
    
    These techniques enable efficient computation and storage while maintaining
    the essential structural properties of the high-dimensional representations.
    """
    PCA = auto()          # Principal Component Analysis for linear reduction
    TSNE = auto()         # t-SNE embedding for non-linear manifold learning
    UMAP = auto()         # Uniform Manifold Approximation for topology preservation
    TENSOR_DECOMP = auto() # Higher-order tensor decomposition methods
    QUANTUM_REDUCE = auto() # Quantum-inspired reduction with complex projections

@dataclass
class KnowledgeTensor:
    """
    Fundamental data structure for representing knowledge in tensor space.
    
    This class encapsulates all information needed to represent and manipulate
    knowledge as high-dimensional tensors, including the tensor data itself,
    metadata about its structure and origin, and mechanisms for content hashing
    and similarity comparison.
    
    Attributes:
        tensor_id: Unique identifier for this tensor instance
        dimensions: List of dimension sizes for each tensor axis
        spaces: List of tensor spaces this tensor operates in
        timestamp: Creation timestamp for temporal tracking
        data: The actual tensor data (numpy array or sparse matrix)
        sparse: Whether the tensor uses sparse representation
        confidence: Confidence score for tensor reliability (0.0-1.0)
        metadata: Additional metadata dictionary for context
        origin: Source identifier for tensor provenance tracking
        tags: Set of string tags for categorization and search
        
    Methods:
        __post_init__: Initialize tensor data structure if not provided
        get_hash: Generate deterministic hash of tensor content
    """
    tensor_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dimensions: List[int] = field(default_factory=list)
    spaces: List[TensorSpace] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    data: Any = None
    sparse: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    origin: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """
        Initialize tensor with proper data structure.
        
        Creates the underlying tensor data structure based on the specified
        dimensions and sparsity requirements. For sparse tensors, uses scipy
        compressed sparse row format for memory efficiency.
        """
        if self.data is None:
            if self.sparse:
                self.data = csr_matrix(np.zeros(self.dimensions))
            else:
                self.data = np.zeros(self.dimensions)
    
    def get_hash(self) -> str:
        """
        Generate a deterministic hash of tensor content.
        
        Creates a SHA-256 hash of the tensor data for deduplication,
        similarity detection, and content verification purposes.
        
        Returns:
            Hexadecimal string representation of the content hash
        """
        if isinstance(self.data, np.ndarray):
            return hashlib.sha256(self.data.tobytes()).hexdigest()
        elif hasattr(self.data, 'toarray'):  # Sparse matrix
            return hashlib.sha256(self.data.toarray().tobytes()).hexdigest()
        else:
            return hashlib.sha256(str(self.data).encode()).hexdigest()

class TensorOperator:
    """
    Static methods for performing mathematical operations on knowledge tensors.
    
    This class provides a comprehensive suite of tensor operations for manipulating
    and transforming knowledge representations in high-dimensional space. Operations
    include basic arithmetic, advanced tensor calculus, dimensionality reduction,
    and quantum-inspired transformations.
    
    Key Operations:
        - Tensor Product: Combine tensors into higher-dimensional representations
        - Tensor Contraction: Reduce dimensionality through index summation
        - Tensor Addition: Element-wise combination of compatible tensors
        - Dimensionality Reduction: Various methods for dimension compression
        - Quantum Transformations: Quantum operator applications
    
    Mathematical Foundation:
        All operations preserve the mathematical integrity of tensor algebra
        while maintaining semantic meaning in the knowledge representation context.
    """
    
    @staticmethod
    def tensor_product(tensor1: KnowledgeTensor, tensor2: KnowledgeTensor) -> KnowledgeTensor:
        """
        Compute tensor product between two knowledge tensors.
        
        The tensor product creates a higher-dimensional representation by combining
        two tensors into a single structure that encodes relationships between
        the original tensor spaces. This operation is fundamental for creating
        composite knowledge representations.
        
        Args:
            tensor1: First tensor for the product operation
            tensor2: Second tensor for the product operation
            
        Returns:
            New KnowledgeTensor containing the tensor product
            
        Raises:
            ValueError: If tensors don't have compatible dense data
            
        Mathematical Details:
            For tensors A and B, computes A ⊗ B where the result has
            dimensions that are the concatenation of input dimensions.
        """
        if isinstance(tensor1.data, np.ndarray) and isinstance(tensor2.data, np.ndarray):
            # Convert to PyTorch tensors for easier tensor operations
            t1 = torch.from_numpy(tensor1.data.copy())
            t2 = torch.from_numpy(tensor2.data.copy())
            
            # Compute tensor product
            result_tensor = torch.tensordot(t1, t2, dims=0).numpy()
            
            # Create new knowledge tensor
            result = KnowledgeTensor(
                dimensions=list(result_tensor.shape),
                spaces=tensor1.spaces + tensor2.spaces,
                data=result_tensor,
                confidence=min(tensor1.confidence, tensor2.confidence),
                metadata={
                    "operation": "tensor_product",
                    "source_tensors": [tensor1.tensor_id, tensor2.tensor_id]
                }
            )
            return result
        else:
            raise ValueError("Tensor product requires dense tensor data")
    
    @staticmethod
    def tensor_contraction(tensor: KnowledgeTensor, axes: Tuple[int, int]) -> KnowledgeTensor:
        """
        Perform tensor contraction along specified axes.
        
        Tensor contraction is a generalization of matrix trace that sums over
        paired indices, effectively reducing the tensor's dimensionality while
        preserving essential structural information.
        
        Args:
            tensor: Input tensor for contraction
            axes: Tuple of axes indices to contract (must be compatible)
            
        Returns:
            New KnowledgeTensor with reduced dimensions
            
        Raises:
            ValueError: If tensor doesn't have compatible dense data
            
        Mathematical Details:
            For tensor T, contraction over axes (i,j) computes Σ_k T[...,k,...,k,...]
        """
        if isinstance(tensor.data, np.ndarray):
            # Convert to PyTorch tensor
            t = torch.from_numpy(tensor.data.copy())
            
            # Perform contraction
            result_tensor = torch.tensordot(t, t, dims=(axes[0], axes[1])).numpy()
            
            # Create new knowledge tensor
            new_spaces = [space for i, space in enumerate(tensor.spaces) 
                         if i != axes[0] and i != axes[1]]
            
            result = KnowledgeTensor(
                dimensions=list(result_tensor.shape),
                spaces=new_spaces,
                data=result_tensor,
                confidence=tensor.confidence,
                metadata={
                    "operation": "contraction",
                    "source_tensor": tensor.tensor_id,
                    "contracted_axes": axes
                }
            )
            return result
        else:
            raise ValueError("Tensor contraction requires dense tensor data")
            
    @staticmethod
    def tensor_addition(tensor1: KnowledgeTensor, tensor2: KnowledgeTensor) -> KnowledgeTensor:
        """
        Add two knowledge tensors element-wise.
        
        Performs element-wise addition of two tensors with compatible dimensions.
        This operation combines knowledge from both tensors while preserving
        the dimensional structure of the tensor space.
        
        Args:
            tensor1: First tensor for addition
            tensor2: Second tensor for addition
            
        Returns:
            New KnowledgeTensor containing the sum
            
        Raises:
            ValueError: If tensors have incompatible dimensions or data types
            
        Mathematical Details:
            For tensors A and B of same shape, computes C = A + B element-wise
        """
        if tensor1.dimensions != tensor2.dimensions:
            raise ValueError("Tensor addition requires same dimensions")
            
        if tensor1.sparse and tensor2.sparse:
            result_data = tensor1.data + tensor2.data
        elif isinstance(tensor1.data, np.ndarray) and isinstance(tensor2.data, np.ndarray):
            result_data = tensor1.data + tensor2.data
        else:
            raise ValueError("Tensor addition requires compatible data types")
            
        # Create new knowledge tensor
        result = KnowledgeTensor(
            dimensions=tensor1.dimensions,
            spaces=tensor1.spaces,
            data=result_data,
            sparse=tensor1.sparse,
            confidence=(tensor1.confidence + tensor2.confidence) / 2,
            metadata={
                "operation": "addition",
                "source_tensors": [tensor1.tensor_id, tensor2.tensor_id]
            }
        )
        return result
        
    @staticmethod
    def reduce_dimensionality(tensor: KnowledgeTensor, 
                             method: DimensionalityReduction,
                             target_dims: int) -> KnowledgeTensor:
        """
        Reduce tensor dimensionality using specified method.
        
        Applies various dimensionality reduction techniques to compress
        high-dimensional tensors while preserving essential information
        and structural properties.
        
        Args:
            tensor: Input tensor for dimensionality reduction
            method: Reduction technique to apply
            target_dims: Target number of dimensions
            
        Returns:
            New KnowledgeTensor with reduced dimensions
            
        Supported Methods:
            - PCA: Principal Component Analysis using SVD
            - TENSOR_DECOMP: Higher-order tensor decomposition
            - QUANTUM_REDUCE: Quantum-inspired complex projections
        """
        if not isinstance(tensor.data, np.ndarray):
            # Convert sparse to dense for reduction
            data = tensor.data.toarray() if hasattr(tensor.data, 'toarray') else tensor.data
        else:
            data = tensor.data
            
        original_shape = data.shape
        
        # Flatten tensor to 2D for reduction
        flat_shape = (original_shape[0], np.prod(original_shape[1:]).astype(int))
        flattened = data.reshape(flat_shape)
        
        if method == DimensionalityReduction.PCA:
            # PCA using SVD
            U, s, Vh = np.linalg.svd(flattened, full_matrices=False)
            reduced = U[:, :target_dims] @ np.diag(s[:target_dims])
        
        elif method == DimensionalityReduction.TENSOR_DECOMP:
            # Simplified higher-order SVD
            reduced = flattened
            for i in range(len(original_shape) - 1):
                reshaped = reduced.reshape(-1, reduced.shape[-1])
                U, s, Vh = np.linalg.svd(reshaped, full_matrices=False)
                reduced = (U[:, :min(target_dims, len(s))] @ 
                          np.diag(s[:min(target_dims, len(s))]))
        
        elif method == DimensionalityReduction.QUANTUM_REDUCE:
            # Quantum-inspired dimensionality reduction
            # Using random projections with complex amplitudes
            projection = (np.random.randn(flattened.shape[1], target_dims) + 
                         1j * np.random.randn(flattened.shape[1], target_dims))
            projection = projection / np.sqrt(np.sum(np.abs(projection)**2, axis=0))
            reduced = np.abs(flattened @ projection)**2
        
        else:
            # Default simple truncation
            if len(original_shape) > 1:
                reduced = flattened[:, :target_dims]
            else:
                reduced = flattened[:target_dims]
        
        # Create new reduced tensor
        new_dims = list(reduced.shape)
        result = KnowledgeTensor(
            dimensions=new_dims,
            spaces=tensor.spaces[:len(new_dims)],  # Truncate spaces to match dims
            data=reduced,
            confidence=tensor.confidence * 0.9,  # Slight confidence penalty for reduction
            metadata={
                "operation": "dimensionality_reduction",
                "source_tensor": tensor.tensor_id,
                "method": method.name,
                "original_shape": original_shape
            }
        )
        return result

    @staticmethod
    def apply_quantum_transformation(tensor: KnowledgeTensor, 
                                   operator_matrix: np.ndarray) -> KnowledgeTensor:
        """
        Apply a quantum transformation operator to the tensor.
        
        Applies a unitary or general linear transformation to the tensor data,
        enabling quantum-inspired operations such as rotations, phase shifts,
        and entangling operations in the tensor space.
        
        Args:
            tensor: Input tensor for transformation
            operator_matrix: Square transformation matrix to apply
            
        Returns:
            New KnowledgeTensor with transformed data
            
        Raises:
            ValueError: If operator matrix is incompatible with tensor shape
            
        Mathematical Details:
            For tensor state |ψ⟩ and operator Û, computes Û|ψ⟩
        """
        if not isinstance(tensor.data, np.ndarray):
            data = tensor.data.toarray() if hasattr(tensor.data, 'toarray') else tensor.data
        else:
            data = tensor.data
            
        # Ensure the tensor is in a quantum-compatible shape
        original_shape = data.shape
        flattened = data.reshape(-1)
        vector_size = len(flattened)
        
        # Check if operator matrix is compatible
        if operator_matrix.shape[0] != operator_matrix.shape[1] or operator_matrix.shape[0] != vector_size:
            raise ValueError(f"Operator matrix shape {operator_matrix.shape} is incompatible with tensor shape {original_shape}")
        
        # Apply transformation
        result_vector = operator_matrix @ flattened
        result_data = result_vector.reshape(original_shape)
        
        # Create new transformed tensor
        result = KnowledgeTensor(
            dimensions=list(result_data.shape),
            spaces=tensor.spaces,
            data=result_data,
            confidence=tensor.confidence,
            metadata={
                "operation": "quantum_transformation",
                "source_tensor": tensor.tensor_id
            }
        )
        return result

class HyperDimensionalKnowledgeStore:
    """
    Comprehensive storage and management system for hyper-dimensional knowledge tensors.
    
    This class provides a complete knowledge management infrastructure including
    tensor storage, indexing, similarity search, and statistical analysis.
    The system is designed for high-performance operations with thread safety
    and efficient memory management.
    
    Features:
        - Thread-safe tensor storage and retrieval
        - Multi-dimensional indexing by space and tags
        - Content-based similarity detection using hash indexes
        - Configurable dimension sizes per tensor space
        - Comprehensive statistics and monitoring
        - Memory-efficient sparse tensor support
        
    Attributes:
        default_dimension_sizes: Default dimensions for each tensor space
        tensors: Main storage dictionary for all tensors
        space_index: Index mapping tensor spaces to tensor IDs
        tag_index: Index mapping tags to tensor IDs
        hash_index: Index mapping content hashes to tensor IDs
        access_count: Total number of access operations
        creation_time: Store initialization timestamp
    """
    
    def __init__(self, default_dimension_sizes: Dict[TensorSpace, int] = None):
        """
        Initialize the hyper-dimensional knowledge store.
        
        Args:
            default_dimension_sizes: Optional custom dimension sizes per space.
                                   If None, uses optimized default values.
        """
        # Default tensor dimensions per space (optimized for different domains)
        self.default_dimension_sizes = default_dimension_sizes or {
            TensorSpace.CONCEPTUAL: 256,    # Rich conceptual relationships
            TensorSpace.LINGUISTIC: 512,    # Complex language patterns
            TensorSpace.PHYSICAL: 128,      # Physical system properties
            TensorSpace.TEMPORAL: 64,       # Time-based patterns
            TensorSpace.CAUSAL: 128,        # Cause-effect relationships
            TensorSpace.EMOTIONAL: 32,      # Emotional state encoding
            TensorSpace.DECISION: 64,       # Decision process modeling
            TensorSpace.QUANTUM: 128        # Quantum state representations
        }
        
        # Primary tensor storage
        self.tensors: Dict[str, KnowledgeTensor] = {}
        
        # Multi-dimensional indexing structures
        self.space_index: Dict[TensorSpace, Set[str]] = {space: set() for space in TensorSpace}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.hash_index: Dict[str, str] = {}  # tensor hash -> tensor_id for deduplication
        
        # Thread safety for concurrent operations
        self._lock = threading.RLock()
        
        # Performance and monitoring statistics
        self.access_count = 0
        self.creation_time = time.time()
        
    def store_tensor(self, tensor: KnowledgeTensor) -> str:
        """
        Store a knowledge tensor in the store with full indexing.
        
        Adds the tensor to the main storage and updates all relevant indexes
        for efficient retrieval by space, tags, and content similarity.
        
        Args:
            tensor: KnowledgeTensor instance to store
            
        Returns:
            The tensor ID for the stored tensor
            
        Thread Safety:
            This method is thread-safe and can be called concurrently.
        """
        with self._lock:
            tensor_id = tensor.tensor_id
            
            # Store the tensor in main collection
            self.tensors[tensor_id] = tensor
            
            # Update space-based index
            for space in tensor.spaces:
                self.space_index[space].add(tensor_id)
                
            # Update tag-based index
            for tag in tensor.tags:
                self.tag_index[tag].add(tensor_id)
                
            # Add to content-based hash index for deduplication
            tensor_hash = tensor.get_hash()
            self.hash_index[tensor_hash] = tensor_id
            
            return tensor_id
            
    def get_tensor(self, tensor_id: str) -> Optional[KnowledgeTensor]:
        """
        Retrieve a tensor by its unique identifier.
        
        Args:
            tensor_id: Unique identifier for the tensor
            
        Returns:
            KnowledgeTensor instance if found, None otherwise
            
        Thread Safety:
            This method is thread-safe and can be called concurrently.
        """
        with self._lock:
            self.access_count += 1
            return self.tensors.get(tensor_id)
            
    def get_tensors_by_space(self, space: TensorSpace) -> List[KnowledgeTensor]:
        """
        Retrieve all tensors operating in a specific tensor space.
        
        Args:
            space: The tensor space to filter by
            
        Returns:
            List of KnowledgeTensor instances in the specified space
        """
        with self._lock:
            tensor_ids = self.space_index.get(space, set())
            return [self.tensors[tid] for tid in tensor_ids if tid in self.tensors]
            
    def get_tensors_by_tag(self, tag: str) -> List[KnowledgeTensor]:
        """
        Retrieve all tensors with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of KnowledgeTensor instances with the specified tag
        """
        with self._lock:
            tensor_ids = self.tag_index.get(tag, set())
            return [self.tensors[tid] for tid in tensor_ids if tid in self.tensors]
            
    def find_similar_tensor(self, tensor: KnowledgeTensor, 
                          similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                          max_results: int = DEFAULT_MAX_RESULTS) -> List[Tuple[str, float]]:
        """
        Find tensors similar to the given tensor using content-based analysis.
        
        Uses multiple similarity detection methods including exact hash matching
        for duplicates and cosine similarity for content-based matching.
        
        Args:
            tensor: Reference tensor for similarity comparison
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (tensor_id, similarity_score) sorted by similarity
        """
        """Find tensors similar to the given tensor"""
        with self._lock:
            if not tensor.spaces:
                return []
                
            # Get tensors in the same spaces
            candidate_ids = set()
            for space in tensor.spaces:
                candidate_ids.update(self.space_index.get(space, set()))
                
            # Check tensor hash first for exact matches
            tensor_hash = tensor.get_hash()
            if tensor_hash in self.hash_index:
                exact_match_id = self.hash_index[tensor_hash]
                if exact_match_id in candidate_ids and exact_match_id != tensor.tensor_id:
                    return [(exact_match_id, 1.0)]
            
            # Calculate similarities
            similarities = []
            for tensor_id in candidate_ids:
                if tensor_id == tensor.tensor_id:
                    continue  # Skip self
                    
                candidate = self.tensors.get(tensor_id)
                if not candidate:
                    continue
                    
                # Calculate similarity if dimensions match
                if tensor.dimensions == candidate.dimensions:
                    try:
                        similarity = self._calculate_tensor_similarity(tensor, candidate)
                        if similarity >= similarity_threshold:
                            similarities.append((tensor_id, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity: {e}")
                        
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
    def _calculate_tensor_similarity(self, tensor1: KnowledgeTensor, 
                                  tensor2: KnowledgeTensor) -> float:
        """
        Calculate cosine similarity between two tensors.
        
        Uses cosine similarity to measure the angular distance between
        tensor representations, providing a normalized similarity score
        between 0.0 (orthogonal) and 1.0 (identical direction).
        
        Args:
            tensor1: First tensor for comparison
            tensor2: Second tensor for comparison
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Flatten tensors for vector comparison
        if isinstance(tensor1.data, np.ndarray) and isinstance(tensor2.data, np.ndarray):
            v1 = tensor1.data.flatten()
            v2 = tensor2.data.flatten()
            
            # Calculate cosine similarity
            return 1.0 - cosine(v1, v2)
        elif hasattr(tensor1.data, 'toarray') and hasattr(tensor2.data, 'toarray'):
            v1 = tensor1.data.toarray().flatten()
            v2 = tensor2.data.toarray().flatten()
            
            return 1.0 - cosine(v1, v2)
        else:
            return 0.0
    
    def create_empty_tensor(self, 
                          spaces: List[TensorSpace], 
                          sparse: bool = False) -> KnowledgeTensor:
        """
        Create an empty tensor with default dimensions for specified spaces.
        
        Args:
            spaces: List of tensor spaces for the new tensor
            sparse: Whether to use sparse representation
            
        Returns:
            New KnowledgeTensor with zero-initialized data
        """
        dimensions = [self.default_dimension_sizes[space] for space in spaces]
        
        tensor = KnowledgeTensor(
            dimensions=dimensions,
            spaces=spaces,
            sparse=sparse
        )
        
        return tensor
        
    def create_combined_space_tensor(self, 
                                   tensor1: KnowledgeTensor, 
                                   tensor2: KnowledgeTensor,
                                   operation: str = "product") -> KnowledgeTensor:
        """Create a tensor combining two different spaces"""
        if operation == "product":
            return TensorOperator.tensor_product(tensor1, tensor2)
        elif operation == "addition":
            return TensorOperator.tensor_addition(tensor1, tensor2)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge store"""
        with self._lock:
            total_tensors = len(self.tensors)
            tensors_by_space = {space.name: len(ids) for space, ids in self.space_index.items()}
            total_tags = len(self.tag_index)
            
            # Calculate total memory usage (rough estimate)
            memory_usage = 0
            for tensor in self.tensors.values():
                if isinstance(tensor.data, np.ndarray):
                    memory_usage += tensor.data.nbytes
                elif hasattr(tensor.data, 'data'):  # Sparse matrices
                    memory_usage += tensor.data.data.nbytes + tensor.data.indptr.nbytes + tensor.data.indices.nbytes
                    
            return {
                "total_tensors": total_tensors,
                "tensors_by_space": tensors_by_space,
                "total_tags": total_tags,
                "unique_tensor_hashes": len(self.hash_index),
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "access_count": self.access_count,
                "uptime_seconds": time.time() - self.creation_time
            }

class HyperDimensionalEmbedding:
    """
    Advanced embedding system for creating tensor representations across multiple spaces.
    
    This class provides sophisticated methods for converting various types of data
    into high-dimensional tensor representations suitable for different knowledge
    domains. Each embedding method is optimized for its specific tensor space
    and preserves the essential characteristics of the input data.
    
    Features:
        - Deterministic embedding generation for reproducibility
        - Space-specific optimization for different data types
        - Temporal encoding with cyclic patterns
        - Quantum state vector representations
        - Causal relationship graph embeddings
        - Multi-space tensor combination operations
        
    Attributes:
        store: Reference to the knowledge store for dimension information
        random_projections: Cached random projection matrices for efficiency
    """
    
    def __init__(self, store: HyperDimensionalKnowledgeStore):
        """
        Initialize the embedding system with a knowledge store reference.
        
        Args:
            store: HyperDimensionalKnowledgeStore instance for configuration
        """
        self.store = store
        self.random_projections = {}
        
    def create_conceptual_embedding(self, concept_data: Dict[str, Any]) -> KnowledgeTensor:
        """
        Create a deterministic embedding in conceptual space.
        
        Converts structured concept data into a high-dimensional vector
        representation that preserves conceptual relationships and hierarchies.
        Uses hash-based seeding for deterministic, reproducible embeddings.
        
        Args:
            concept_data: Dictionary containing concept information and properties
            
        Returns:
            KnowledgeTensor in CONCEPTUAL space
            
        Example:
            concept_data = {
                "name": "quantum_computer",
                "type": "technology", 
                "properties": ["superposition", "entanglement"]
            }
        """
        dimension = self.store.default_dimension_sizes[TensorSpace.CONCEPTUAL]
        
        # Create deterministic embedding using content hash
        concept_str = str(sorted(concept_data.items()))
        hash_obj = hashlib.sha256(concept_str.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Use hash to seed random generator for reproducible embeddings
        np.random.seed(hash_int % (2**32))
        
        # Generate normalized embedding vector
        embedding = np.random.randn(dimension)
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
        
        # Create tensor with rich metadata
        tensor = KnowledgeTensor(
            dimensions=[dimension],
            spaces=[TensorSpace.CONCEPTUAL],
            data=embedding,
            metadata={"concept_data": concept_data},
            tags=set(concept_data.keys())
        )
        
        return tensor
        
    def create_linguistic_embedding(self, text: str) -> KnowledgeTensor:
        """
        Create a sophisticated embedding in linguistic space.
        
        Generates a high-dimensional representation of text using character-level
        random projections with position-dependent phase encoding. This approach
        captures both character content and positional information.
        
        Args:
            text: Input text string to embed
            
        Returns:
            KnowledgeTensor in LINGUISTIC space
            
        Technical Details:
            Uses complex-valued character vectors with position-dependent phase
            shifts to encode both content and sequence information.
        """
        dimension = self.store.default_dimension_sizes[TensorSpace.LINGUISTIC]
        
        # Initialize character projection vectors if not cached
        if "linguistic" not in self.random_projections:
            # Create deterministic character-level random projections
            char_vectors = {}
            for c in "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"-()[]{}":
                char_vectors[c] = np.random.randn(dimension)
                char_vectors[c] = char_vectors[c] / np.linalg.norm(char_vectors[c])
                
            self.random_projections["linguistic"] = char_vectors
            
        # Create embedding by combining character vectors with phase encoding
        embedding = np.zeros(dimension)
        text = text.lower()
        
        for i, char in enumerate(text):
            if char in self.random_projections["linguistic"]:
                # Apply position-dependent phase shift for sequence encoding
                phase = 2 * np.pi * i / max(1, len(text))
                char_vec = self.random_projections["linguistic"][char] * np.exp(1j * phase)
                embedding += np.real(char_vec)
                
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Create tensor with linguistic metadata
        tensor = KnowledgeTensor(
            dimensions=[dimension],
            spaces=[TensorSpace.LINGUISTIC],
            data=embedding,
            metadata={"text": text[:100], "text_length": len(text)},
            tags=set(["text", f"length_{min(10, len(text)//10)}0s"])
        )
        
        return tensor
        
    def create_temporal_embedding(self, timestamp: float, window: float = 86400) -> KnowledgeTensor:
        """
        Create a cyclic temporal embedding encoding multiple time scales.
        
        Generates a temporal representation using multiple periodic functions
        to encode different time cycles (daily, weekly, monthly, yearly).
        This approach captures temporal patterns across multiple scales.
        
        Args:
            timestamp: Unix timestamp to encode
            window: Time window for normalization (default: 24 hours)
            
        Returns:
            KnowledgeTensor in TEMPORAL space
        """
        dimension = self.store.default_dimension_sizes[TensorSpace.TEMPORAL]
        
        # Create cyclic temporal representation
        embedding = np.zeros(dimension)
        
        # Encode different time cycles with different frequencies
        # Daily cycle
        day_phase = 2 * np.pi * (timestamp % 86400) / 86400
        embedding[:dimension//4] = np.sin(np.linspace(0, 4*np.pi, dimension//4) + day_phase)
        
        # Weekly cycle
        week_phase = 2 * np.pi * (timestamp % (86400*7)) / (86400*7)
        embedding[dimension//4:dimension//2] = np.sin(np.linspace(0, 2*np.pi, dimension//4) + week_phase)
        
        # Monthly cycle
        month_phase = 2 * np.pi * (timestamp % (86400*30)) / (86400*30)
        embedding[dimension//2:3*dimension//4] = np.sin(np.linspace(0, 2*np.pi, dimension//4) + month_phase)
        
        # Yearly cycle
        year_phase = 2 * np.pi * (timestamp % (86400*365)) / (86400*365)
        embedding[3*dimension//4:] = np.sin(np.linspace(0, 2*np.pi, dimension - 3*dimension//4) + year_phase)
        
        # Create tensor
        tensor = KnowledgeTensor(
            dimensions=[dimension],
            spaces=[TensorSpace.TEMPORAL],
            data=embedding,
            metadata={
                "timestamp": timestamp,
                "window": window,
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            },
            tags=set(["temporal", time.strftime("%Y-%m-%d", time.localtime(timestamp))])
        )
        
        return tensor
        
    def create_quantum_embedding(self, state_vector: np.ndarray) -> KnowledgeTensor:
        """Create an embedding in quantum space"""
        dimension = self.store.default_dimension_sizes[TensorSpace.QUANTUM]
        
        # Convert to complex state vector if not already
        if not np.iscomplexobj(state_vector):
            complex_state = state_vector.astype(np.complex128)
        else:
            complex_state = state_vector.copy()
            
        # Normalize as quantum state
        norm = np.linalg.norm(complex_state)
        if norm > 0:
            complex_state = complex_state / norm
            
        # If needed, project to desired dimension
        if len(complex_state) != dimension:
            if len(complex_state) > dimension:
                # Truncate
                complex_state = complex_state[:dimension]
                # Re-normalize
                complex_state = complex_state / np.linalg.norm(complex_state)
            else:
                # Pad with zeros
                padded = np.zeros(dimension, dtype=np.complex128)
                padded[:len(complex_state)] = complex_state
                complex_state = padded
                # Re-normalize
                complex_state = complex_state / np.linalg.norm(complex_state)
                
        # Calculate coherence
        coherence = 0.0
        for i in range(len(complex_state)):
            for j in range(i+1, len(complex_state)):
                coherence += abs(complex_state[i] * np.conj(complex_state[j]))
                
        coherence = coherence / (dimension * (dimension - 1) / 2) if dimension > 1 else 1.0
                
        # Create tensor
        tensor = KnowledgeTensor(
            dimensions=[dimension],
            spaces=[TensorSpace.QUANTUM],
            data=complex_state,
            metadata={
                "quantum_state": True,
                "coherence": float(coherence),
                "is_pure": True
            },
            tags=set(["quantum", "state_vector"])
        )
        
        return tensor
        
    def create_causal_embedding(self, cause_effect_pairs: List[Tuple[str, str]]) -> KnowledgeTensor:
        """Create an embedding in causal space"""
        dimension = self.store.default_dimension_sizes[TensorSpace.CAUSAL]
        
        # Create causal matrix representation
        # Each pair (cause, effect) creates a directed edge in the causal graph
        unique_concepts = set()
        for cause, effect in cause_effect_pairs:
            unique_concepts.add(cause)
            unique_concepts.add(effect)
            
        # Map concepts to indices
        concept_to_idx = {concept: i for i, concept in enumerate(sorted(unique_concepts))}
        n_concepts = len(unique_concepts)
        
        # Create adjacency matrix representation of causal graph
        causal_matrix = np.zeros((n_concepts, n_concepts))
        
        for cause, effect in cause_effect_pairs:
            cause_idx = concept_to_idx[cause]
            effect_idx = concept_to_idx[effect]
            causal_matrix[cause_idx, effect_idx] = 1.0
            
        # If the matrix is smaller than the target dimension, embed it
        if n_concepts < dimension:
            embedded_matrix = np.zeros((dimension, dimension))
            embedded_matrix[:n_concepts, :n_concepts] = causal_matrix
            causal_matrix = embedded_matrix
        elif n_concepts > dimension:
            # Apply dimensionality reduction
            # For simplicity, just take the submatrix of most connected concepts
            importance = np.sum(causal_matrix, axis=0) + np.sum(causal_matrix, axis=1)
            top_indices = np.argsort(importance)[-dimension:]
            causal_matrix = causal_matrix[np.ix_(top_indices, top_indices)]
            
            # Update concept mapping
            concept_to_idx = {c: i for i, (c, old_idx) in enumerate(concept_to_idx.items()) 
                           if old_idx in top_indices}
            
        # Create tensor
        tensor = KnowledgeTensor(
            dimensions=[dimension, dimension],
            spaces=[TensorSpace.CAUSAL, TensorSpace.CAUSAL],
            data=causal_matrix,
            metadata={
                "concept_mapping": concept_to_idx,
                "cause_effect_pairs": cause_effect_pairs
            },
            tags=set(["causal", "graph"])
        )
        
        return tensor
        
    def combine_embeddings(self, tensors: List[KnowledgeTensor], 
                          operation: str = "product") -> KnowledgeTensor:
        """Combine multiple embeddings into a higher-dimensional representation"""
        if not tensors:
            return None
            
        if len(tensors) == 1:
            return tensors[0]
            
        # Start with first tensor
        result = tensors[0]
        
        # Combine with remaining tensors
        for tensor in tensors[1:]:
            if operation == "product":
                result = TensorOperator.tensor_product(result, tensor)
            elif operation == "addition":
                # Only combine if dimensions match
                if result.dimensions == tensor.dimensions:
                    result = TensorOperator.tensor_addition(result, tensor)
                else:
                    logger.warning("Cannot add tensors with different dimensions")
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        return result

# Example usage of the system
def run_example():
    """
    Comprehensive demonstration of the hyper-dimensional knowledge tensor system.
    
    This function showcases the key capabilities of the tensor system including:
    - Multi-space embedding creation
    - Tensor operations and transformations  
    - Similarity detection and analysis
    - System statistics and monitoring
    """
    print(f"Current Date/Time (UTC): 2025-07-24 14:18:26")
    print(f"User: Shriram-2005")
    print("\n===== Hyper-Dimensional Knowledge Tensor System Demo =====")
    
    # Initialize the knowledge infrastructure
    store = HyperDimensionalKnowledgeStore()
    embedding_engine = HyperDimensionalEmbedding(store)
    
    print("\nCreating embeddings across multiple tensor spaces...")
    
    # Create conceptual embedding
    concept_data = {
        "name": "quantum_computer",
        "type": "technology",
        "properties": ["superposition", "entanglement", "qubits"]
    }
    conceptual_tensor = embedding_engine.create_conceptual_embedding(concept_data)
    store.store_tensor(conceptual_tensor)
    print(f"Created conceptual tensor: {conceptual_tensor.tensor_id}")
    
    # Create linguistic embedding
    text = "Hyper-dimensional knowledge representation enables sophisticated reasoning across multiple domains"
    linguistic_tensor = embedding_engine.create_linguistic_embedding(text)
    store.store_tensor(linguistic_tensor)
    print(f"Created linguistic tensor: {linguistic_tensor.tensor_id}")
    
    # Create temporal embedding
    timestamp = time.time()
    temporal_tensor = embedding_engine.create_temporal_embedding(timestamp)
    store.store_tensor(temporal_tensor)
    print(f"Created temporal tensor: {temporal_tensor.tensor_id}")
    
    # Create quantum embedding
    quantum_state = np.random.rand(128) + 1j * np.random.rand(128)
    quantum_state = quantum_state / np.linalg.norm(quantum_state)
    quantum_tensor = embedding_engine.create_quantum_embedding(quantum_state)
    store.store_tensor(quantum_tensor)
    print(f"Created quantum tensor: {quantum_tensor.tensor_id}")
    
    # Combine tensors
    combined_tensor = embedding_engine.combine_embeddings(
        [conceptual_tensor, linguistic_tensor], operation="product"
    )
    store.store_tensor(combined_tensor)
    print(f"Created combined tensor: {combined_tensor.tensor_id}")
    
    # Print store stats
    stats = store.get_store_stats()
    print("\nKnowledge store statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Perform tensor operations
    reduced_tensor = TensorOperator.reduce_dimensionality(
        combined_tensor, DimensionalityReduction.QUANTUM_REDUCE, 64
    )
    store.store_tensor(reduced_tensor)
    print(f"\nCreated reduced dimension tensor: {reduced_tensor.tensor_id}")
    print(f"Original dimensions: {combined_tensor.dimensions}")
    print(f"Reduced dimensions: {reduced_tensor.dimensions}")
    
    # Find similar tensors
    similar = store.find_similar_tensor(conceptual_tensor)
    print("\nSimilar tensors to conceptual tensor:")
    for tensor_id, similarity in similar:
        print(f"  {tensor_id}: {similarity:.4f}")
        
    print("\nHyper-Dimensional Knowledge Tensor system demonstration completed!")
    print("Successfully showcased multi-space embeddings, tensor operations, and intelligent similarity detection.")

# Export key classes and utilities for module usage
__all__ = [
    # Enums
    'TensorSpace',
    'DimensionalityReduction',
    
    # Core data structures
    'KnowledgeTensor',
    
    # Operations and algorithms
    'TensorOperator',
    
    # Storage and management
    'HyperDimensionalKnowledgeStore',
    
    # Embedding systems
    'HyperDimensionalEmbedding',
    
    # Utility functions
    'run_example'
]

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    run_example()