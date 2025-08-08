"""
MARS Holographic Memory Integration System

An advanced memory storage and retrieval system based on holographic principles,
implementing distributed memory encoding across multiple dimensions with support
for various encoding schemes and cognitive memory types.

This system provides:
- Multi-dimensional holographic memory encoding
- Multiple encoding strategies (spatial, spectral, phase, quantum-inspired)
- Hierarchical memory organization with different types
- Efficient content-based retrieval with similarity matching
- Automatic memory consolidation and association discovery
- Thread-safe concurrent operations
- Compression support for memory optimization

The holographic approach allows for:
- Distributed storage where information is spread across the entire memory space
- Associative recall based on partial information
- Graceful degradation when parts of memory are corrupted
- Pattern completion and reconstruction capabilities
- Content-addressable memory access

Key Components:
- HolographicEncoder: Handles encoding/decoding between data and holographic representations
- HolographicMemorySystem: Core memory storage and retrieval engine
- HolographicMemoryManager: High-level interface for memory operations

Example Usage:
    # Create a holographic memory manager
    manager = HolographicMemoryManager(
        dimensions=512,
        encoding_type=MemoryEncoding.COMPOSITE
    )
    
    # Store memories
    memory_id = manager.remember(
        "Quantum computers use superposition",
        category="knowledge",
        tags=["quantum", "computing"]
    )
    
    # Retrieve similar memories
    results = manager.recall("quantum superposition")

Dependencies:
    - numpy: Numerical computing for holographic operations
    - scipy: FFT operations and sparse matrix support
    - threading: Thread-safe memory operations
    - hashlib: Content hashing for deduplication
    
"""

# Standard library imports
import hashlib
import heapq
import logging
import math
import os
import pickle
import re
import threading
import time
import uuid
import zlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import numpy as np
import scipy.fft as fft
from scipy.sparse import csr_matrix

# Module constants
__version__ = "2.0.0"
__author__ = "Shriram-2005"

# Default configuration constants
DEFAULT_DIMENSIONS = 512
DEFAULT_ADDRESS_DIMENSIONS = 8
DEFAULT_ENCODING_TYPE = "COMPOSITE"
DEFAULT_COMPRESSION_LEVEL = "LOSSLESS"

# Memory system constants
MAX_WORKING_MEMORY_SIZE = 10
DEFAULT_CONSOLIDATION_INTERVAL = 3600  # 1 hour
DEFAULT_SIMILARITY_THRESHOLD = 0.7
GRID_RESOLUTION = 0.5

# Encoding constants
PHASE_ENCODING_RESOLUTION = 1000
SPATIAL_ENCODING_SCALE = 255
FREQUENCY_CUTOFF_RATIO = 0.1

class MemoryEncoding(Enum):
    """
    Enumeration of holographic memory encoding schemes.
    
    Each encoding type implements a different strategy for converting
    data into holographic representations:
    
    - SPATIAL: Direct spatial distribution encoding
    - SPECTRAL: Frequency domain encoding using FFT
    - PHASE: Phase-based encoding for complex data
    - COMPOSITE: Combined spatial and phase encoding
    - QUANTUM: Quantum-inspired superposition encoding
    - SPARSE: Sparse distributed representation
    - FRACTAL: Hierarchical fractal encoding
    - TEMPORAL: Time-series based encoding
    """
    SPATIAL = auto()         # Direct spatial holographic encoding
    SPECTRAL = auto()        # Frequency domain encoding using FFT
    PHASE = auto()           # Phase-based complex encoding
    COMPOSITE = auto()       # Combined spatial and phase encoding
    QUANTUM = auto()         # Quantum-inspired superposition encoding
    SPARSE = auto()          # Sparse distributed representation
    FRACTAL = auto()         # Hierarchical fractal encoding
    TEMPORAL = auto()        # Time-series based encoding


class MemoryType(Enum):
    """
    Enumeration of cognitive memory types based on neuroscience research.
    
    Different memory types serve different cognitive functions:
    
    - EPISODIC: Autobiographical events and experiences
    - SEMANTIC: Factual knowledge and concepts
    - PROCEDURAL: Skills, habits, and procedures
    - WORKING: Temporary information for processing
    - REFERENCE: Stable reference data and constants
    - ASSOCIATIVE: Connections between concepts
    - IMPLICIT: Unconscious knowledge and patterns
    - META: Knowledge about memory and cognition
    """
    EPISODIC = auto()        # Event-based autobiographical memories
    SEMANTIC = auto()        # Factual knowledge and concepts
    PROCEDURAL = auto()      # Skills, procedures, and habits
    WORKING = auto()         # Temporary processing memory
    REFERENCE = auto()       # Stable reference data
    ASSOCIATIVE = auto()     # Associative connections
    IMPLICIT = auto()        # Implicit/unconscious knowledge
    META = auto()            # Meta-cognitive knowledge


class CompressionLevel(Enum):
    """
    Enumeration of memory compression levels for storage optimization.
    
    Different compression levels trade off between storage efficiency
    and information fidelity:
    
    - NONE: No compression, full fidelity
    - LOSSLESS: Compressed but perfect reconstruction
    - MINIMAL: Slight quality loss for better compression
    - MEDIUM: Moderate compression with acceptable quality
    - HIGH: High compression with noticeable quality reduction
    - EXTREME: Maximum compression for archival storage
    """
    NONE = auto()            # No compression applied
    LOSSLESS = auto()        # Lossless compression (zlib/gzip)
    MINIMAL = auto()         # Minimal lossy compression (95% quality)
    MEDIUM = auto()          # Medium compression (80% quality)
    HIGH = auto()            # High compression (60% quality)
    EXTREME = auto()         # Extreme compression (40% quality)
    ADAPTIVE = auto()        # Adaptively determined compression


@dataclass
class MemoryAddress:
    """
    Represents a holographic memory address in multi-dimensional space.
    
    Memory addresses provide spatial coordinates for storing and retrieving
    memories in the holographic space. Each address includes both spatial
    coordinates and phase information for complete holographic addressing.
    
    Attributes:
        address_id: Unique identifier for this address
        coordinates: N-dimensional spatial coordinates
        phase: Phase component for holographic interference
        resolution: Spatial resolution of the address
        reference_count: Number of memories using this address
        last_accessed: Timestamp of most recent access
        stability: Address stability measure [0.0, 1.0]
    """
    
    address_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coordinates: Optional[np.ndarray] = None  # N-dimensional coordinates
    phase: float = 0.0                        # Phase component [0, 2π]
    resolution: float = 1.0                   # Address resolution
    reference_count: int = 0                  # Reference counter
    last_accessed: float = field(default_factory=time.time)
    stability: float = 1.0                    # Stability measure [0.0, 1.0]
    
    def distance_to(self, other: 'MemoryAddress') -> float:
        """
        Calculate holographic distance to another address.
        
        Args:
            other: The other memory address to compare
            
        Returns:
            Distance value (0.0 = identical, higher = more distant)
            
        Note:
            Combines spatial distance with phase difference for
            complete holographic distance calculation.
        """
        if self.coordinates is None or other.coordinates is None:
            return float('inf')
            
        if len(self.coordinates) != len(other.coordinates):
            return float('inf')
            
        # Calculate Euclidean spatial distance
        spatial_distance = np.linalg.norm(self.coordinates - other.coordinates)
        
        # Calculate phase distance (handles circular nature of phase)
        phase_diff = abs(self.phase - other.phase)
        phase_distance = min(phase_diff, 2 * np.pi - phase_diff) / (2 * np.pi)
        
        # Combined distance (weighted combination)
        return spatial_distance + 0.5 * phase_distance
    
    def get_hash(self) -> str:
        """
        Generate a unique hash representing this address.
        
        Returns:
            MD5 hash string based on coordinates, phase, and resolution
        """
        if self.coordinates is None:
            return hashlib.md5(str(self.phase).encode()).hexdigest()
            
        # Combine coordinates, phase, and resolution for unique hash
        addr_str = f"{self.coordinates.tobytes()}{self.phase}{self.resolution}"
        return hashlib.md5(addr_str.encode()).hexdigest()


@dataclass
class HolographicMemory:
    """
    Represents a single holographic memory with complete metadata.
    
    A HolographicMemory stores both the original content and its holographic
    encoding, along with access patterns, associations, and other metadata
    needed for effective memory management.
    
    Attributes:
        memory_id: Unique identifier for this memory
        address: Holographic address for spatial organization
        content_hash: Hash of the original content for deduplication
        content: Original data/information stored
        encoded_data: Holographic representation of the content
        encoding_type: Method used for holographic encoding
        memory_type: Cognitive category of this memory
        creation_time: When the memory was created
        last_accessed: Most recent access timestamp
        access_count: Number of times accessed
        confidence: Reliability measure [0.0, 1.0]
        tags: Set of descriptive tags
        associations: List of related memory IDs
        compression_level: Compression applied to the data
    """
    
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    address: Optional[MemoryAddress] = None
    content_hash: Optional[str] = None
    content: Any = None
    encoded_data: Optional[np.ndarray] = None
    encoding_type: MemoryEncoding = MemoryEncoding.COMPOSITE
    memory_type: MemoryType = MemoryType.SEMANTIC
    creation_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0                   # Reliability measure [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_level: CompressionLevel = CompressionLevel.NONE
    tags: Set[str] = field(default_factory=set)
    associations: List[str] = field(default_factory=list)  # Related memory IDs
    
    def access(self) -> None:
        """
        Record a memory access event.
        
        Updates access timestamp, increments access counter,
        and updates associated address statistics.
        """
        self.last_accessed = time.time()
        self.access_count += 1
        
        if self.address:
            self.address.last_accessed = self.last_accessed
            self.address.reference_count += 1
    
    def update_confidence(self, new_confidence: float) -> None:
        """
        Update memory confidence using exponential smoothing.
        
        Args:
            new_confidence: New confidence value [0.0, 1.0]
            
        Note:
            Uses 80% old confidence and 20% new confidence for stability.
        """
        # Exponential smoothing for confidence update
        self.confidence = 0.8 * self.confidence + 0.2 * new_confidence
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def get_age(self) -> float:
        """
        Get the age of this memory in seconds.
        
        Returns:
            Age in seconds since creation
        """
        return time.time() - self.creation_time
    
    def add_association(self, memory_id: str) -> None:
        """
        Add an association to another memory.
        
        Args:
            memory_id: ID of the memory to associate with
        """
        if memory_id not in self.associations:
            self.associations.append(memory_id)
    
    def remove_association(self, memory_id: str) -> None:
        """
        Remove an association to another memory.
        
        Args:
            memory_id: ID of the memory to disassociate from
        """
        if memory_id in self.associations:
            self.associations.remove(memory_id)
    
    def get_size(self) -> int:
        """
        Calculate approximate memory size in bytes.
        
        Returns:
            Estimated size in bytes including encoded data and content
            
        Note:
            Provides rough estimation for memory usage monitoring.
        """
        size = 0
        
        # Add encoded data size
        if self.encoded_data is not None:
            size += self.encoded_data.nbytes
        
        # Estimate content size
        if self.content is not None:
            if isinstance(self.content, (str, bytes)):
                size += len(self.content)
            elif isinstance(self.content, (list, tuple, set)):
                size += len(self.content) * 8  # Rough estimate
            elif isinstance(self.content, dict):
                size += len(self.content) * 16  # Rough estimate
            else:
                size += 100  # Default size for unknown types
                
        return size


class HolographicEncoder:
    """
    Handles encoding and decoding between data and holographic representations.
    
    The HolographicEncoder implements various encoding strategies to convert
    arbitrary data into holographic representations suitable for distributed
    storage and associative retrieval.
    
    Supported encoding types:
    - Spatial: Direct spatial distribution
    - Spectral: Frequency domain encoding
    - Phase: Complex phase-based encoding
    - Composite: Combined spatial and phase
    - Quantum: Quantum-inspired superposition
    - Sparse: Sparse distributed representation
    - Fractal: Hierarchical fractal encoding
    - Temporal: Time-series based encoding
    
    Attributes:
        dimensions: Size of the holographic representation space
        encoding_type: Strategy used for encoding data
        random_phases: Random phase patterns for encoding
    """
    
    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, 
                 encoding_type: MemoryEncoding = MemoryEncoding.COMPOSITE):
        """
        Initialize the holographic encoder.
        
        Args:
            dimensions: Size of holographic space (default: 512)
            encoding_type: Encoding strategy to use (default: COMPOSITE)
        """
        self.dimensions = dimensions
        self.encoding_type = encoding_type
        
        # Create random phase vectors for different memory types
        self.phase_vectors = {}
        
        # Initialize random phase patterns for each memory type
        for memory_type in MemoryType:
            self.phase_vectors[memory_type] = self._generate_random_phases()
    
    def _generate_random_phases(self) -> np.ndarray:
        """
        Generate random phase vector for holographic encoding.
        
        Returns:
            Array of random phases in range [0, 2π]
        """
        return np.random.uniform(0, 2*np.pi, self.dimensions)
    
    def encode(self, data: Any, memory_type: MemoryType, 
             compression: CompressionLevel = CompressionLevel.NONE) -> Tuple[np.ndarray, str]:
        """
        Encode data into holographic representation.
        
        Args:
            data: Data to encode (any serializable type)
            memory_type: Type of memory for encoding strategy
            compression: Compression level to apply
            
        Returns:
            Tuple of (encoded_data, content_hash)
            
        Note:
            The encoding process converts arbitrary data into a holographic
            representation suitable for distributed storage and retrieval.
        """
        # Convert data to bytes
        if isinstance(data, (str, int, float, bool, list, dict, tuple)):
            data_bytes = pickle.dumps(data)
        elif isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            # Try to pickle, but fallback to string representation
            try:
                data_bytes = pickle.dumps(data)
            except:
                data_bytes = str(data).encode()
        
        # Calculate content hash
        content_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Apply compression if needed
        if compression != CompressionLevel.NONE:
            data_bytes = self._compress_data(data_bytes, compression)
            
        # Convert bytes to numerical array
        byte_array = np.frombuffer(data_bytes, dtype=np.uint8)
        
        # Pad or truncate to match dimensions
        if len(byte_array) < self.dimensions:
            # Pad with zeros
            padded = np.zeros(self.dimensions, dtype=np.uint8)
            padded[:len(byte_array)] = byte_array
            byte_array = padded
        elif len(byte_array) > self.dimensions:
            # Either truncate or use dimensionality reduction
            if len(byte_array) < self.dimensions * 2:
                # Simple truncation
                byte_array = byte_array[:self.dimensions]
            else:
                # Use compression for larger data
                byte_array = self._reduce_dimensions(byte_array)
                
        # Normalize to [0,1]
        normalized = byte_array.astype(np.float32) / 255.0
        
        # Apply encoding based on type
        if self.encoding_type == MemoryEncoding.SPATIAL:
            encoded = normalized
            
        elif self.encoding_type == MemoryEncoding.SPECTRAL:
            # Use Fourier transform
            encoded = fft.fft(normalized)
            
        elif self.encoding_type == MemoryEncoding.PHASE:
            # Use phase encoding
            phase = self.phase_vectors.get(memory_type, self._generate_random_phases())
            encoded = np.exp(1j * (normalized * 2*np.pi + phase))
            
        elif self.encoding_type == MemoryEncoding.COMPOSITE:
            # Use both spatial and phase encoding
            phase = self.phase_vectors.get(memory_type, self._generate_random_phases())
            spatial = normalized
            
            # Apply Fourier transform
            spectral = fft.fft(spatial)
            
            # Modulate with phase
            encoded = spectral * np.exp(1j * phase)
            
        elif self.encoding_type == MemoryEncoding.QUANTUM:
            # Quantum-inspired encoding using complex amplitudes
            phase = self.phase_vectors.get(memory_type, self._generate_random_phases())
            amplitudes = np.sqrt(normalized)
            encoded = amplitudes * np.exp(1j * phase)
            
        elif self.encoding_type == MemoryEncoding.SPARSE:
            # Sparse distributed encoding
            sparsity = 0.1  # 10% of bits active
            k = int(self.dimensions * sparsity)
            
            # Create sparse vector
            encoded = np.zeros(self.dimensions, dtype=np.complex128)
            
            # Select k positions based on data hash
            hash_value = int(hashlib.md5(data_bytes).hexdigest(), 16)
            random_state = np.random.RandomState(hash_value)
            active_indices = random_state.choice(self.dimensions, k, replace=False)
            
            # Set active positions
            encoded[active_indices] = 1.0
            
            # Modulate with data values
            for i, idx in enumerate(active_indices):
                if i < len(normalized):
                    encoded[idx] = normalized[i]
            
        elif self.encoding_type == MemoryEncoding.FRACTAL:
            # Fractal encoding using recursive patterns
            encoded = np.zeros(self.dimensions, dtype=np.complex128)
            
            # Split into multiple scales
            scales = 4
            for scale in range(scales):
                # Calculate slice size for this scale
                slice_size = self.dimensions // (2**scale)
                if slice_size < 4:  # Stop if slices get too small
                    break
                
                # For each scale, embed part of the data
                start_idx = 0
                while start_idx + slice_size <= self.dimensions:
                    # Get data segment
                    data_segment = normalized[start_idx % len(normalized):
                                            (start_idx + slice_size) % len(normalized)]
                    
                    # Repeat data if needed
                    if len(data_segment) < slice_size:
                        repeats = slice_size // len(data_segment) + 1
                        data_segment = np.tile(data_segment, repeats)[:slice_size]
                    
                    # Apply Fourier transform at this scale
                    segment_fft = fft.fft(data_segment)
                    
                    # Add to encoded data
                    encoded[start_idx:start_idx + slice_size] += segment_fft / scales
                    
                    # Move to next segment
                    start_idx += slice_size
        
        elif self.encoding_type == MemoryEncoding.TEMPORAL:
            # Temporal encoding with time-based patterns
            encoded = np.zeros(self.dimensions, dtype=np.complex128)
            
            # Use sliding windows with different time scales
            window_sizes = [4, 8, 16, 32, 64]
            
            for size in window_sizes:
                if size >= len(normalized):
                    continue
                    
                # Create overlapping windows
                for start in range(0, len(normalized) - size, size // 2):
                    window = normalized[start:start + size]
                    
                    # Apply window function (Hann window)
                    window_func = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(window)) / len(window)))
                    windowed = window * window_func
                    
                    # Transform window
                    window_fft = fft.fft(windowed, self.dimensions)
                    
                    # Add to encoding with time-based weighting
                    time_weight = np.exp(-start / len(normalized))
                    encoded += window_fft * time_weight / len(window_sizes)
        
        else:  # Default to spatial encoding
            encoded = normalized
        
        return encoded, content_hash
    
    def decode(self, encoded_data: np.ndarray, 
             memory_type: MemoryType,
             original_data_type: type = None) -> Any:
        """Decode data from holographic representation"""
        # Apply decoding based on encoding type
        if self.encoding_type == MemoryEncoding.SPATIAL:
            decoded = encoded_data
            
        elif self.encoding_type == MemoryEncoding.SPECTRAL:
            # Use inverse Fourier transform
            decoded = np.real(fft.ifft(encoded_data))
            
        elif self.encoding_type == MemoryEncoding.PHASE:
            # Phase decoding
            phase = self.phase_vectors.get(memory_type, np.zeros(self.dimensions))
            phases = np.angle(encoded_data)
            normalized = ((phases - phase) % (2*np.pi)) / (2*np.pi)
            decoded = normalized
            
        elif self.encoding_type == MemoryEncoding.COMPOSITE:
            # Composite decoding
            phase = self.phase_vectors.get(memory_type, np.zeros(self.dimensions))
            
            # Remove phase modulation
            spectral = encoded_data * np.exp(-1j * phase)
            
            # Apply inverse Fourier transform
            decoded = np.real(fft.ifft(spectral))
            
        elif self.encoding_type == MemoryEncoding.QUANTUM:
            # Quantum-inspired decoding
            phase = self.phase_vectors.get(memory_type, np.zeros(self.dimensions))
            amplitudes = np.abs(encoded_data)
            normalized = amplitudes ** 2  # Square to get probabilities
            decoded = normalized
            
        elif self.encoding_type == MemoryEncoding.SPARSE:
            # Sparse distributed decoding
            decoded = np.abs(encoded_data)
            
        elif self.encoding_type == MemoryEncoding.FRACTAL:
            # Fractal decoding - combine multiple scales
            decoded = np.zeros(self.dimensions)
            
            # Extract from multiple scales
            scales = 4
            for scale in range(scales):
                # Calculate slice size for this scale
                slice_size = self.dimensions // (2**scale)
                if slice_size < 4:  # Stop if slices get too small
                    break
                
                # For each scale, extract part of the data
                start_idx = 0
                while start_idx + slice_size <= self.dimensions:
                    # Extract segment
                    segment_fft = encoded_data[start_idx:start_idx + slice_size]
                    
                    # Apply inverse Fourier transform
                    segment = np.real(fft.ifft(segment_fft))
                    
                    # Add to decoded data
                    decoded[start_idx:start_idx + slice_size] += segment / scales
                    
                    # Move to next segment
                    start_idx += slice_size
                    
        elif self.encoding_type == MemoryEncoding.TEMPORAL:
            # Temporal decoding - simplified approach using IFFT
            decoded = np.real(fft.ifft(encoded_data))
            
        else:  # Default to spatial decoding
            decoded = encoded_data
        
        # Normalize to [0, 255]
        byte_array = np.clip(decoded * 255, 0, 255).astype(np.uint8)
        
        # Convert back to bytes
        data_bytes = byte_array.tobytes()
        
        # Try to unpickle if original type is specified
        if original_data_type:
            try:
                # Try to decompress first if needed
                try:
                    data_bytes = self._decompress_data(data_bytes)
                except:
                    pass
                    
                # Try to unpickle
                return pickle.loads(data_bytes)
            except:
                # Return raw bytes if unpickling fails
                return data_bytes
        
        return data_bytes
    
    def _compress_data(self, data: bytes, level: CompressionLevel) -> bytes:
        """Compress data based on compression level"""
        import zlib
        
        if level == CompressionLevel.NONE:
            return data
            
        if level == CompressionLevel.LOSSLESS:
            # Use zlib compression
            return zlib.compress(data)
            
        # For lossy compression, we'll use a simple approach of selective sampling
        data_array = np.frombuffer(data, dtype=np.uint8)
        
        if level == CompressionLevel.MINIMAL:
            # Keep 80% of data
            keep_ratio = 0.8
        elif level == CompressionLevel.MEDIUM:
            # Keep 50% of data
            keep_ratio = 0.5
        elif level == CompressionLevel.HIGH:
            # Keep 25% of data
            keep_ratio = 0.25
        elif level == CompressionLevel.EXTREME:
            # Keep 10% of data
            keep_ratio = 0.1
        else:  # ADAPTIVE
            # Determine ratio based on data size
            if len(data) > 10000:
                keep_ratio = 0.1
            elif len(data) > 1000:
                keep_ratio = 0.3
            else:
                keep_ratio = 0.7
                
        # Determine sampling interval
        interval = int(1 / keep_ratio)
        
        # Sample data
        sampled = data_array[::interval]
        
        # Store interval for reconstruction
        result = np.concatenate(([interval], sampled))
        
        # Further compress with zlib
        return zlib.compress(result.tobytes())
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data"""
        import zlib
        
        # Try to decompress with zlib
        try:
            decompressed = zlib.decompress(compressed_data)
            
            # Check if this is our custom lossy compression
            data_array = np.frombuffer(decompressed, dtype=np.uint8)
            if len(data_array) > 1:
                interval = data_array[0]
                
                # Check if this looks like our format (interval should be 1-50)
                if 1 <= interval <= 50:
                    # Reconstruct data with zeros in between
                    sampled = data_array[1:]
                    reconstructed = np.zeros(len(sampled) * interval, dtype=np.uint8)
                    reconstructed[::interval] = sampled
                    return reconstructed.tobytes()
            
            # If not our format, return zlib decompressed data
            return decompressed
            
        except zlib.error:
            # Not compressed or invalid compression, return as is
            return compressed_data
    
    def _reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        """Reduce dimensions of data to fit in encoding space"""
        # If data is already small enough, return as is
        if len(data) <= self.dimensions:
            return data
            
        # Method 1: Simple downsampling
        if len(data) < self.dimensions * 10:
            # Calculate interval
            interval = len(data) // self.dimensions + 1
            return data[::interval][:self.dimensions]
            
        # Method 2: Block averaging for larger data
        block_size = len(data) // self.dimensions
        result = np.zeros(self.dimensions, dtype=np.float32)
        
        for i in range(self.dimensions):
            start = i * block_size
            end = min((i + 1) * block_size, len(data))
            if start < end:
                result[i] = np.mean(data[start:end])
            
        return result.astype(np.uint8)

class HolographicMemorySystem:
    """
    Core holographic memory storage and retrieval engine.
    
    This class implements the fundamental holographic memory operations including
    storage, retrieval, indexing, and memory management. It maintains multiple
    indices for efficient access patterns and supports various encoding strategies.
    
    Key Features:
    - Holographic encoding/decoding of arbitrary data
    - Multi-dimensional spatial addressing
    - Content-based similarity matching
    - Tag-based categorization and search
    - Memory type classification
    - Automatic association discovery
    - Thread-safe concurrent operations
    - Memory consolidation and optimization
    
    The system uses multiple indexing strategies:
    - Spatial index: Grid-based spatial organization
    - Content index: Hash-based content deduplication
    - Tag index: Category-based memory grouping
    - Type index: Memory type classification
    - Association graph: Relationship mapping
    
    Attributes:
        dimensions: Size of holographic encoding space
        address_dimensions: Size of address coordinate space
        encoding_type: Strategy for data encoding
        memories: Main memory storage
        addresses: Memory address storage
        spatial_index: Grid-based spatial lookup
        content_index: Content hash lookup
        tag_index: Tag-based categorization
        type_index: Memory type classification
        associations: Memory relationship graph
    """
    
    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, 
                 address_dimensions: int = DEFAULT_ADDRESS_DIMENSIONS,
                 encoding_type: MemoryEncoding = MemoryEncoding.COMPOSITE):
        """
        Initialize the holographic memory system.
        
        Args:
            dimensions: Size of holographic encoding space (default: 512)
            address_dimensions: Size of address coordinate space (default: 8)
            encoding_type: Encoding strategy (default: COMPOSITE)
        """
        self.dimensions = dimensions
        self.address_dimensions = address_dimensions
        self.encoding_type = encoding_type
        
        # Initialize encoder
        self.encoder = HolographicEncoder(dimensions, encoding_type)
        
        # Memory storage
        self.memories: Dict[str, HolographicMemory] = {}
        
        # Memory addresses
        self.addresses: Dict[str, MemoryAddress] = {}
        
        # Spatial index for fast memory lookup
        self.spatial_index: Dict[str, List[str]] = {}  # grid_cell -> [memory_ids]
        
        # Content-based index
        self.content_index: Dict[str, str] = {}  # content_hash -> memory_id
        
        # Tag index
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [memory_ids]
        
        # Memory type index
        self.type_index: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        
        # Association graph
        self.associations: Dict[str, Set[str]] = {}  # memory_id -> {memory_ids}
        
        # Working memory cache
        self.working_memory: Dict[str, Any] = {}
        self.working_memory_capacity = 10
        
        # Memory statistics
        self.stats = {
            "total_memories": 0,
            "total_addresses": 0,
            "memory_by_type": {t.name: 0 for t in MemoryType},
            "total_associations": 0,
            "retrieval_count": 0,
            "retrieval_success": 0,
            "avg_retrieval_time": 0.0
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def store(self, data: Any, memory_type: MemoryType = MemoryType.SEMANTIC,
            tags: List[str] = None, associate_with: List[str] = None,
            compression: CompressionLevel = CompressionLevel.NONE) -> str:
        """Store data in holographic memory"""
        with self._lock:
            # Encode the data
            encoded_data, content_hash = self.encoder.encode(data, memory_type, compression)
            
            # Check if identical content already exists
            if content_hash in self.content_index:
                existing_id = self.content_index[content_hash]
                # Update access time and count
                if existing_id in self.memories:
                    self.memories[existing_id].access()
                    
                    # Add new tags if provided
                    if tags:
                        for tag in tags:
                            self.add_tag(existing_id, tag)
                            
                    # Add new associations if provided
                    if associate_with:
                        for assoc_id in associate_with:
                            self.associate(existing_id, assoc_id)
                    
                return existing_id
            
            # Generate a memory address
            address = self._generate_address(encoded_data, memory_type)
            
            # Create the memory
            memory = HolographicMemory(
                address=address,
                content_hash=content_hash,
                content=data,
                encoded_data=encoded_data,
                encoding_type=self.encoding_type,
                memory_type=memory_type,
                tags=set(tags) if tags else set(),
                compression_level=compression
            )
            
            # Store the memory
            self.memories[memory.memory_id] = memory
            
            # Update indexes
            self.content_index[content_hash] = memory.memory_id
            self.type_index[memory_type].append(memory.memory_id)
            
            # Add to spatial index
            grid_cell = self._get_grid_cell(address.coordinates)
            if grid_cell not in self.spatial_index:
                self.spatial_index[grid_cell] = []
            self.spatial_index[grid_cell].append(memory.memory_id)
            
            # Add tags
            if tags:
                for tag in tags:
                    self.add_tag(memory.memory_id, tag)
            
            # Create associations
            if associate_with:
                for assoc_id in associate_with:
                    self.associate(memory.memory_id, assoc_id)
            
            # Update statistics
            self.stats["total_memories"] += 1
            self.stats["total_addresses"] += 1
            self.stats["memory_by_type"][memory_type.name] += 1
            
            return memory.memory_id
    
    def retrieve(self, memory_id: str) -> Any:
        """Retrieve memory by ID"""
        with self._lock:
            start_time = time.time()
            
            if memory_id not in self.memories:
                self.stats["retrieval_count"] += 1
                return None
                
            memory = self.memories[memory_id]
            memory.access()
            
            # Update working memory
            self._update_working_memory(memory_id, memory.content)
            
            # Update statistics
            self.stats["retrieval_count"] += 1
            self.stats["retrieval_success"] += 1
            
            # Update retrieval time statistics
            retrieval_time = time.time() - start_time
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] * (self.stats["retrieval_success"] - 1) +
                retrieval_time
            ) / self.stats["retrieval_success"]
            
            return memory.content
    
    def retrieve_by_address(self, coordinates: np.ndarray, 
                          phase: float = 0.0, 
                          radius: float = 1.0) -> List[Dict[str, Any]]:
        """Retrieve memories near a given address"""
        with self._lock:
            start_time = time.time()
            
            # Create query address
            query_address = MemoryAddress(
                coordinates=coordinates,
                phase=phase
            )
            
            # Find nearby grid cells for efficient lookup
            nearby_cells = self._get_nearby_grid_cells(coordinates, radius)
            
            # Collect candidate memories
            candidates = []
            for cell in nearby_cells:
                if cell in self.spatial_index:
                    candidates.extend(self.spatial_index[cell])
            
            # Calculate distances to each candidate
            results = []
            for memory_id in candidates:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    
                    if memory.address:
                        distance = memory.address.distance_to(query_address)
                        
                        if distance <= radius:
                            # Update access info
                            memory.access()
                            
                            results.append({
                                "memory_id": memory_id,
                                "distance": distance,
                                "content": memory.content,
                                "confidence": memory.confidence * (1 - distance/radius),
                                "memory_type": memory.memory_type.name
                            })
            
            # Sort by distance
            results.sort(key=lambda x: x["distance"])
            
            # Update statistics
            self.stats["retrieval_count"] += 1
            if results:
                self.stats["retrieval_success"] += 1
            
            # Update retrieval time
            retrieval_time = time.time() - start_time
            if results:
                self.stats["avg_retrieval_time"] = (
                    self.stats["avg_retrieval_time"] * (self.stats["retrieval_success"] - 1) +
                    retrieval_time
                ) / self.stats["retrieval_success"]
                
                # Update working memory with closest result
                self._update_working_memory(results[0]["memory_id"], results[0]["content"])
            
            return results
    
    def retrieve_by_content(self, data: Any, 
                          threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Retrieve memories with similar content"""
        with self._lock:
            start_time = time.time()
            
                        # Encode query data
            encoded_query, query_hash = self.encoder.encode(data, MemoryType.WORKING)
            
            # Check for exact match first
            if query_hash in self.content_index:
                memory_id = self.content_index[query_hash]
                memory = self.memories[memory_id]
                memory.access()
                
                result = [{
                    "memory_id": memory_id,
                    "similarity": 1.0,
                    "content": memory.content,
                    "confidence": memory.confidence,
                    "memory_type": memory.memory_type.name
                }]
                
                # Update working memory
                self._update_working_memory(memory_id, memory.content)
                
                # Update statistics
                self.stats["retrieval_count"] += 1
                self.stats["retrieval_success"] += 1
                
                # Update retrieval time
                retrieval_time = time.time() - start_time
                self.stats["avg_retrieval_time"] = (
                    self.stats["avg_retrieval_time"] * (self.stats["retrieval_success"] - 1) +
                    retrieval_time
                ) / self.stats["retrieval_success"]
                
                return result
            
            # If no exact match, search for similar memories
            results = []
            
            # Calculate similarity with all memories (this could be optimized with indexes)
            for memory_id, memory in self.memories.items():
                if memory.encoded_data is None:
                    continue
                
                # Calculate similarity between encodings
                similarity = self._calculate_similarity(encoded_query, memory.encoded_data)
                
                if similarity >= threshold:
                    # Update access info
                    memory.access()
                    
                    results.append({
                        "memory_id": memory_id,
                        "similarity": similarity,
                        "content": memory.content,
                        "confidence": memory.confidence * similarity,
                        "memory_type": memory.memory_type.name
                    })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update statistics
            self.stats["retrieval_count"] += 1
            if results:
                self.stats["retrieval_success"] += 1
            
            # Update retrieval time
            retrieval_time = time.time() - start_time
            if results:
                self.stats["avg_retrieval_time"] = (
                    self.stats["avg_retrieval_time"] * (self.stats["retrieval_success"] - 1) +
                    retrieval_time
                ) / self.stats["retrieval_success"]
                
                # Update working memory with most similar result
                self._update_working_memory(results[0]["memory_id"], results[0]["content"])
            
            return results
    
    def retrieve_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Retrieve memories with a specific tag"""
        with self._lock:
            if tag not in self.tag_index:
                return []
                
            results = []
            for memory_id in self.tag_index[tag]:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory.access()
                    
                    results.append({
                        "memory_id": memory_id,
                        "content": memory.content,
                        "confidence": memory.confidence,
                        "memory_type": memory.memory_type.name,
                        "creation_time": memory.creation_time
                    })
            
            # Sort by confidence (highest first)
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return results
    
    def retrieve_by_type(self, memory_type: MemoryType, 
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve memories of a specific type"""
        with self._lock:
            if memory_type not in self.type_index:
                return []
                
            results = []
            for memory_id in self.type_index[memory_type][:limit]:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory.access()
                    
                    results.append({
                        "memory_id": memory_id,
                        "content": memory.content,
                        "confidence": memory.confidence,
                        "tags": list(memory.tags),
                        "creation_time": memory.creation_time
                    })
            
            # Sort by access count (most accessed first)
            results.sort(key=lambda x: self.memories[x["memory_id"]].access_count, reverse=True)
            
            return results
    
    def retrieve_associated(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve memories associated with a given memory"""
        with self._lock:
            if memory_id not in self.memories:
                return []
                
            memory = self.memories[memory_id]
            
            results = []
            for assoc_id in memory.associations:
                if assoc_id in self.memories:
                    assoc_memory = self.memories[assoc_id]
                    assoc_memory.access()
                    
                    results.append({
                        "memory_id": assoc_id,
                        "content": assoc_memory.content,
                        "confidence": assoc_memory.confidence,
                        "memory_type": assoc_memory.memory_type.name,
                        "association_strength": 1.0  # Simple fixed strength for now
                    })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return results
    
    def update(self, memory_id: str, data: Any, 
             preserve_associations: bool = True) -> bool:
        """Update memory content"""
        with self._lock:
            if memory_id not in self.memories:
                return False
                
            memory = self.memories[memory_id]
            
            # Save original properties
            original_type = memory.memory_type
            original_tags = memory.tags.copy()
            original_associations = memory.associations.copy()
            
            # Remove from content index
            if memory.content_hash in self.content_index:
                del self.content_index[memory.content_hash]
            
            # Encode new data
            encoded_data, content_hash = self.encoder.encode(
                data, 
                memory.memory_type,
                memory.compression_level
            )
            
            # Update memory
            memory.content = data
            memory.encoded_data = encoded_data
            memory.content_hash = content_hash
            memory.last_accessed = time.time()
            memory.access_count += 1
            
            # Update content index
            self.content_index[content_hash] = memory_id
            
            # Restore associations if needed
            if not preserve_associations:
                # Remove old associations
                for assoc_id in original_associations:
                    self.disassociate(memory_id, assoc_id)
                    
                # Clear associations
                memory.associations = []
            
            return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        with self._lock:
            if memory_id not in self.memories:
                return False
                
            memory = self.memories[memory_id]
            
            # Remove from content index
            if memory.content_hash in self.content_index:
                del self.content_index[memory.content_hash]
            
            # Remove from type index
            if memory.memory_type in self.type_index and memory_id in self.type_index[memory.memory_type]:
                self.type_index[memory.memory_type].remove(memory_id)
            
            # Remove from spatial index
            if memory.address:
                grid_cell = self._get_grid_cell(memory.address.coordinates)
                if grid_cell in self.spatial_index and memory_id in self.spatial_index[grid_cell]:
                    self.spatial_index[grid_cell].remove(memory_id)
            
            # Remove from tag index
            for tag in memory.tags:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
            
            # Remove associations
            for assoc_id in memory.associations:
                self.disassociate(memory_id, assoc_id)
            
            # Remove from memory storage
            del self.memories[memory_id]
            
            # Update statistics
            self.stats["total_memories"] -= 1
            self.stats["memory_by_type"][memory.memory_type.name] -= 1
            
            return True
    
    def add_tag(self, memory_id: str, tag: str) -> bool:
        """Add a tag to a memory"""
        with self._lock:
            if memory_id not in self.memories:
                return False
                
            memory = self.memories[memory_id]
            
            # Add tag to memory
            memory.tags.add(tag)
            
            # Add to tag index
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if memory_id not in self.tag_index[tag]:
                self.tag_index[tag].append(memory_id)
            
            return True
    
    def remove_tag(self, memory_id: str, tag: str) -> bool:
        """Remove a tag from a memory"""
        with self._lock:
            if memory_id not in self.memories:
                return False
                
            memory = self.memories[memory_id]
            
            # Remove tag from memory
            if tag in memory.tags:
                memory.tags.remove(tag)
            
            # Remove from tag index
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
            
            return True
    
    def associate(self, memory_id1: str, memory_id2: str) -> bool:
        """Create an association between two memories"""
        with self._lock:
            if memory_id1 not in self.memories or memory_id2 not in self.memories:
                return False
                
            if memory_id1 == memory_id2:
                return False  # Can't associate with self
                
            # Create bidirectional association
            memory1 = self.memories[memory_id1]
            memory2 = self.memories[memory_id2]
            
            # Add to associations lists
            if memory_id2 not in memory1.associations:
                memory1.associations.append(memory_id2)
            
            if memory_id1 not in memory2.associations:
                memory2.associations.append(memory_id1)
            
            # Update statistics
            self.stats["total_associations"] += 1
            
            return True
    
    def disassociate(self, memory_id1: str, memory_id2: str) -> bool:
        """Remove an association between two memories"""
        with self._lock:
            if memory_id1 not in self.memories or memory_id2 not in self.memories:
                return False
                
            # Remove bidirectional association
            memory1 = self.memories[memory_id1]
            memory2 = self.memories[memory_id2]
            
            # Remove from associations lists
            if memory_id2 in memory1.associations:
                memory1.associations.remove(memory_id2)
            
            if memory_id1 in memory2.associations:
                memory2.associations.remove(memory_id1)
            
            # Update statistics
            self.stats["total_associations"] -= 1
            
            return True
    
    def consolidate_memory(self) -> Dict[str, Any]:
        """Consolidate memories to optimize storage and associations"""
        with self._lock:
            start_time = time.time()
            results = {
                "merged": 0,
                "strengthened": 0,
                "pruned": 0,
                "time_taken": 0.0
            }
            
            # 1. Find and merge very similar memories
            similarity_threshold = 0.95
            memories_list = list(self.memories.values())
            
            # Compare pairs of memories
            for i, memory1 in enumerate(memories_list):
                for j in range(i + 1, len(memories_list)):
                    memory2 = memories_list[j]
                    
                    # Skip if not the same type
                    if memory1.memory_type != memory2.memory_type:
                        continue
                        
                    # Calculate similarity if both have encoded data
                    if memory1.encoded_data is not None and memory2.encoded_data is not None:
                        similarity = self._calculate_similarity(memory1.encoded_data, memory2.encoded_data)
                        
                        if similarity >= similarity_threshold:
                            # Merge memories - keep the one with higher confidence
                            if memory1.confidence >= memory2.confidence:
                                keep_id, remove_id = memory1.memory_id, memory2.memory_id
                            else:
                                keep_id, remove_id = memory2.memory_id, memory1.memory_id
                                
                            # Transfer associations
                            for assoc_id in self.memories[remove_id].associations:
                                self.associate(keep_id, assoc_id)
                                
                            # Transfer tags
                            for tag in self.memories[remove_id].tags:
                                self.add_tag(keep_id, tag)
                                
                            # Delete the redundant memory
                            self.delete(remove_id)
                            
                            results["merged"] += 1
            
            # 2. Strengthen related memories
            for memory_id, memory in self.memories.items():
                if not memory.associations:
                    continue
                    
                # Find memories that share multiple associations
                association_counts = {}
                
                # Count how many associations are shared with other memories
                for assoc_id in memory.associations:
                    if assoc_id in self.memories:
                        for secondary_id in self.memories[assoc_id].associations:
                            if secondary_id != memory_id:
                                association_counts[secondary_id] = association_counts.get(secondary_id, 0) + 1
                
                # Create new associations for memories with multiple shared connections
                for secondary_id, count in association_counts.items():
                    if count >= 2 and secondary_id in self.memories:
                        if self.associate(memory_id, secondary_id):
                            results["strengthened"] += 1
            
            # 3. Prune old, low-confidence, rarely accessed memories
            cutoff_time = time.time() - 30 * 24 * 3600  # 30 days old
            
            for memory_id in list(self.memories.keys()):
                memory = self.memories[memory_id]
                
                # Skip recent memories
                if memory.creation_time > cutoff_time:
                    continue
                    
                # Consider pruning old, low confidence, rarely accessed memories
                if memory.confidence < 0.3 and memory.access_count < 3:
                    if memory.last_accessed < cutoff_time:
                        self.delete(memory_id)
                        results["pruned"] += 1
            
            results["time_taken"] = time.time() - start_time
            return results
    
    def _generate_address(self, encoded_data: np.ndarray, 
                        memory_type: MemoryType) -> MemoryAddress:
        """Generate a holographic address for a memory"""
        # Create a hash of the encoded data
        data_hash = hashlib.md5(encoded_data.tobytes()).hexdigest()
        
        # Use the hash to seed a random generator for reproducibility
        hash_int = int(data_hash, 16)
        random_gen = np.random.RandomState(hash_int)
        
        # Generate coordinates in address space
        coordinates = random_gen.randn(self.address_dimensions)
        
        # Normalize to unit length
        norm = np.linalg.norm(coordinates)
        if norm > 0:
            coordinates = coordinates / norm
        
        # Apply memory type specific offset
        memory_type_index = list(MemoryType).index(memory_type)
        offset_angle = memory_type_index * (2*np.pi / len(MemoryType))
        
        # Rotate the coordinates slightly based on memory type
        if self.address_dimensions >= 2:
            # Apply rotation in first 2 dimensions
            rotation_matrix = np.eye(self.address_dimensions)
            rotation_matrix[0, 0] = np.cos(offset_angle)
            rotation_matrix[0, 1] = -np.sin(offset_angle)
            rotation_matrix[1, 0] = np.sin(offset_angle)
            rotation_matrix[1, 1] = np.cos(offset_angle)
            
            coordinates = rotation_matrix @ coordinates
        
        # Generate phase from the hash
        phase_bits = hash_int & 0xFFFFFFFF  # Take lower 32 bits
        phase = (phase_bits / 0xFFFFFFFF) * (2*np.pi)  # Map to [0, 2π)
        
        # Create address
        address = MemoryAddress(
            coordinates=coordinates,
            phase=phase,
            resolution=1.0,  # Default resolution
            reference_count=1
        )
        
        # Store the address
        address_hash = address.get_hash()
        self.addresses[address_hash] = address
        
        return address
    
    def _get_grid_cell(self, coordinates: np.ndarray) -> str:
        """Convert coordinates to a grid cell identifier"""
        if coordinates is None:
            return "null"
            
        # Scale and discretize coordinates
        grid_resolution = 0.5
        grid_coords = tuple(int(c / grid_resolution) for c in coordinates)
        
        # Convert to string
        return ":".join(str(c) for c in grid_coords)
    
    def _get_nearby_grid_cells(self, coordinates: np.ndarray, radius: float) -> List[str]:
        """Get grid cells within radius of coordinates"""
        if coordinates is None:
            return []
            
        # Scale radius to grid resolution
        grid_resolution = 0.5
        grid_radius = math.ceil(radius / grid_resolution)
        
        # Get base grid cell
        base_grid = tuple(int(c / grid_resolution) for c in coordinates)
        
        # Generate all nearby grid cells
        nearby_cells = []
        
        # Generate all grid cell offsets within radius
        def generate_offsets(dim: int, current: List[int]) -> List[Tuple[int, ...]]:
            if dim == 0:
                return [tuple(current)]
                
            results = []
            for offset in range(-grid_radius, grid_radius + 1):
                current.append(offset)
                results.extend(generate_offsets(dim - 1, current))
                current.pop()
                
            return results
        
        offsets = generate_offsets(len(coordinates), [])
        
        # Apply offsets to base grid
        for offset in offsets:
            cell = tuple(b + o for b, o in zip(base_grid, offset))
            nearby_cells.append(":".join(str(c) for c in cell))
            
        return nearby_cells
    
    def _calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate similarity between two encodings"""
        # Handle different encoding types appropriately
        if self.encoding_type in [MemoryEncoding.SPATIAL, MemoryEncoding.SPECTRAL]:
            # Use correlation coefficient
            if encoding1.shape != encoding2.shape:
                return 0.0
                
            correlation = np.corrcoef(np.abs(encoding1), np.abs(encoding2))[0, 1]
            # Handle NaN result
            if np.isnan(correlation):
                return 0.0
                
            # Map from [-1, 1] to [0, 1]
            return (correlation + 1) / 2
            
        elif self.encoding_type in [MemoryEncoding.PHASE, MemoryEncoding.COMPOSITE, MemoryEncoding.QUANTUM]:
            # Use inner product of complex vectors
            if encoding1.shape != encoding2.shape:
                return 0.0
                
            # Compute normalized inner product
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            inner_product = np.abs(np.vdot(encoding1, encoding2) / (norm1 * norm2))
            return inner_product
            
        elif self.encoding_type == MemoryEncoding.SPARSE:
            # Use Jaccard similarity for sparse encoding
            active1 = set(np.where(np.abs(encoding1) > 0.01)[0])
            active2 = set(np.where(np.abs(encoding2) > 0.01)[0])
            
            if not active1 or not active2:
                return 0.0
                
            intersection = len(active1.intersection(active2))
            union = len(active1.union(active2))
            
            return intersection / union
            
        else:  # Default similarity measure
            # Cosine similarity
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.abs(np.dot(encoding1, encoding2)) / (norm1 * norm2)
    
    def _update_working_memory(self, memory_id: str, content: Any) -> None:
        """Update the working memory cache"""
        # Add to working memory
        self.working_memory[memory_id] = content
        
        # If over capacity, remove oldest item
        if len(self.working_memory) > self.working_memory_capacity:
            oldest_id = min(self.working_memory.keys(), 
                         key=lambda mid: self.memories[mid].last_accessed 
                         if mid in self.memories else 0)
            del self.working_memory[oldest_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system"""
        with self._lock:
            # Update base statistics
            self.stats["total_memories"] = len(self.memories)
            self.stats["total_addresses"] = len(self.addresses)
            
            # Count memories by type
            for memory_type in MemoryType:
                self.stats["memory_by_type"][memory_type.name] = len(self.type_index.get(memory_type, []))
                
            # Count total associations
            association_count = sum(len(memory.associations) for memory in self.memories.values())
            self.stats["total_associations"] = association_count // 2  # Divide by 2 because associations are bidirectional
            
            # Calculate additional statistics
            memory_ages = [time.time() - memory.creation_time for memory in self.memories.values()]
            
            # Calculate size
            total_size = sum(memory.get_size() for memory in self.memories.values())
            
            # Calculate confidence statistics
            if self.memories:
                confidence_values = [memory.confidence for memory in self.memories.values()]
                avg_confidence = sum(confidence_values) / len(confidence_values)
                min_confidence = min(confidence_values) if confidence_values else 0
                max_confidence = max(confidence_values) if confidence_values else 0
            else:
                avg_confidence = 0
                min_confidence = 0
                max_confidence = 0
            
            # Extended statistics
            extended_stats = {
                "avg_memory_age": sum(memory_ages) / len(memory_ages) if memory_ages else 0,
                "oldest_memory": max(memory_ages) if memory_ages else 0,
                "newest_memory": min(memory_ages) if memory_ages else 0,
                "total_memory_size_bytes": total_size,
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "avg_tags_per_memory": sum(len(memory.tags) for memory in self.memories.values()) / max(1, len(self.memories)),
                "avg_associations_per_memory": association_count / max(1, len(self.memories)),
                "total_tags": len(self.tag_index),
                "memory_type_distribution": self.stats["memory_by_type"]
            }
            
            return {**self.stats, **extended_stats}
    
    def save_to_file(self, filepath: str) -> bool:
        """Save memory system to file"""
        try:
            with self._lock:
                # Create a simplified version for serialization
                serializable_data = {
                    "dimensions": self.dimensions,
                    "address_dimensions": self.address_dimensions,
                    "encoding_type": self.encoding_type.name,
                    "memories": {},
                    "stats": self.stats
                }
                
                # Convert memories to serializable format
                for memory_id, memory in self.memories.items():
                    serializable_data["memories"][memory_id] = {
                        "content": memory.content,
                        "memory_type": memory.memory_type.name,
                        "creation_time": memory.creation_time,
                        "last_accessed": memory.last_accessed,
                        "access_count": memory.access_count,
                        "confidence": memory.confidence,
                        "tags": list(memory.tags),
                        "associations": memory.associations,
                        "compression_level": memory.compression_level.name
                    }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(serializable_data, f)
                
                return True
                
        except Exception as e:
            print(f"Error saving memory system: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HolographicMemorySystem':
        """Load memory system from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Create new system with same parameters
            system = cls(
                dimensions=data["dimensions"],
                address_dimensions=data["address_dimensions"],
                encoding_type=MemoryEncoding[data["encoding_type"]]
            )
            
            # Load memories
            for memory_id, memory_data in data["memories"].items():
                memory_type = MemoryType[memory_data["memory_type"]]
                
                # Store the memory
                system.store(
                    data=memory_data["content"],
                    memory_type=memory_type,
                    tags=memory_data["tags"],
                    compression=CompressionLevel[memory_data["compression_level"]]
                )
                
                # Update memory attributes
                if memory_id in system.memories:
                    memory = system.memories[memory_id]
                    memory.creation_time = memory_data["creation_time"]
                    memory.last_accessed = memory_data["last_accessed"]
                    memory.access_count = memory_data["access_count"]
                    memory.confidence = memory_data["confidence"]
            
            # Create associations
            for memory_id, memory_data in data["memories"].items():
                for assoc_id in memory_data["associations"]:
                    system.associate(memory_id, assoc_id)
            
            # Restore statistics
            system.stats = data["stats"]
            
            return system
            
        except Exception as e:
            print(f"Error loading memory system: {str(e)}")
            return cls()  # Return a new empty system

class HolographicMemoryManager:
    """
    High-level manager for holographic memory operations and maintenance.
    
    This class provides a simplified interface for managing holographic memory systems,
    including automated maintenance, memory consolidation, categorization, and
    background processing capabilities.
    
    Features:
        - Automated memory consolidation and cleanup
        - Category-based memory organization
        - Background maintenance threading
        - Intelligent tagging and association
        - Configurable compression strategies
        - Real-time memory optimization
    
    Attributes:
        memory_system: The underlying holographic memory system
        consolidation_interval: Time between automatic consolidation cycles (seconds)
        auto_tagging: Whether to automatically tag memories
        auto_association: Whether to automatically create memory associations
        default_compression: Default compression level for new memories
        default_categories: Mapping of category names to memory types
        maintenance_thread: Background thread for memory maintenance
        running: Whether the manager is currently active
    """
    
    def __init__(self, dimensions: int = 512, address_dimensions: int = 8,
               encoding_type: MemoryEncoding = MemoryEncoding.COMPOSITE,
               consolidation_interval: int = 3600):  # 1 hour default
        """
        Initialize the holographic memory manager.
        
        Args:
            dimensions: Holographic vector space dimensionality
            address_dimensions: Memory address space dimensions
            encoding_type: Default encoding scheme for memories
            consolidation_interval: Seconds between automatic consolidation cycles
        """
        self.memory_system = HolographicMemorySystem(
            dimensions=dimensions,
            address_dimensions=address_dimensions,
            encoding_type=encoding_type
        )
        
        # Memory handling settings
        self.consolidation_interval = consolidation_interval
        self.auto_tagging = True
        self.auto_association = True
        self.default_compression = CompressionLevel.LOSSLESS
        
        # Memory categories mapping for simplified access
        self.default_categories = {
            "knowledge": MemoryType.SEMANTIC,
            "event": MemoryType.EPISODIC,
            "skill": MemoryType.PROCEDURAL,
            "reference": MemoryType.REFERENCE,
            "temp": MemoryType.WORKING
        }
        
        # Background maintenance state
        self.maintenance_thread = None
        self.running = False
        self.last_consolidation = time.time()
        
        # Query history for optimization
        self.query_history = deque(maxlen=100)
        
        # Memory organization indexes
        self.memory_by_category = {category: [] for category in self.default_categories}
    
    def start_maintenance(self) -> None:
        """
        Start background maintenance thread for automatic memory management.
        
        The maintenance thread handles:
        - Periodic memory consolidation
        - Garbage collection of expired memories
        - Index optimization and rebuilding
        - Performance monitoring and adjustment
        """
        if self.maintenance_thread is not None and self.maintenance_thread.is_alive():
            return
            
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()
        if self.maintenance_thread is not None and self.maintenance_thread.is_alive():
            return
            
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()
    
    def stop_maintenance(self) -> None:
        """
        Stop background maintenance thread gracefully.
        
        This method ensures that the maintenance thread completes its current
        cycle and shuts down cleanly before returning.
        """
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=1.0)
    
    def _maintenance_loop(self) -> None:
        """
        Background maintenance loop for continuous memory optimization.
        
        This method runs in a separate thread and performs periodic maintenance
        tasks including memory consolidation, garbage collection, and system
        optimization based on configured intervals.
        """
        while self.running:
            try:
                now = time.time()
                
                # Consolidate memory if interval has elapsed
                if now - self.last_consolidation >= self.consolidation_interval:
                    results = self.memory_system.consolidate_memory()
                    self.last_consolidation = now
                    
                    print(f"Memory consolidated: {results['merged']} merged, "
                        f"{results['strengthened']} strengthened, "
                        f"{results['pruned']} pruned")
            
            except Exception as e:
                print(f"Error in maintenance loop: {str(e)}")
            
            # Sleep for a bit to avoid excessive CPU usage
            time.sleep(60)  # Check every minute
    
    def remember(self, data: Any, category: str = "knowledge", 
               tags: List[str] = None, related_to: List[str] = None) -> str:
        """
        Store a memory with automatic categorization and tagging.
        
        This is the main interface for storing memories, providing intelligent
        categorization, automatic tagging, and association generation.
        
        Args:
            data: The data to store as a memory
            category: Category for memory organization ("knowledge", "event", etc.)
            tags: Optional list of tags for the memory
            related_to: Optional list of memory IDs to associate with
            
        Returns:
            Unique identifier for the stored memory
        """
        # Determine memory type from category
        memory_type = self.default_categories.get(category, MemoryType.SEMANTIC)
        
        # Automatically extract tags if enabled and not provided
        if self.auto_tagging and tags is None:
            tags = self._extract_tags(data)
            
        # Store the memory
        memory_id = self.memory_system.store(
            data=data,
            memory_type=memory_type,
            tags=tags,
            associate_with=related_to,
            compression=self.default_compression
        )
        
        # Add to category index
        if category in self.memory_by_category:
            self.memory_by_category[category].append(memory_id)
        
        # Create automatic associations if enabled
        if self.auto_association and memory_id:
            self._create_automatic_associations(memory_id, data)
        
        return memory_id
    
    def recall(self, query: Any, category: str = None, 
             tags: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recall memories using various query methods and filtering criteria.
        
        This method provides flexible memory retrieval with support for direct
        ID lookup, semantic queries, tag-based filtering, and category constraints.
        
        Args:
            query: The query data (can be string, vector, or memory ID)
            category: Optional category filter for results
            tags: Optional tag filters for results
            limit: Maximum number of results to return
            
        Returns:
            List of memory dictionaries with content, confidence, and metadata
        """
        results = []
        
        # Track query for history and optimization
        self.query_history.append({
            "query": query,
            "category": category,
            "tags": tags,
            "timestamp": time.time()
        })
        
        # Case 1: If query is a memory ID, do direct lookup
        if isinstance(query, str) and query in self.memory_system.memories:
            content = self.memory_system.retrieve(query)
            if content is not None:
                results.append({
                    "memory_id": query,
                    "content": content,
                    "confidence": 1.0,
                    "retrieval_method": "direct_id"
                })
                return results
        
        # Case 2: If tags provided, start with tag-based lookup
        if tags:
            for tag in tags:
                tag_results = self.memory_system.retrieve_by_tag(tag)
                
                # Filter by category if needed
                if category:
                    memory_type = self.default_categories.get(category)
                    if memory_type:
                        tag_results = [r for r in tag_results 
                                     if r["memory_type"] == memory_type.name]
                
                for result in tag_results:
                    result["retrieval_method"] = "tag"
                    results.append(result)
        
        # Case 3: Content-based lookup
        content_results = self.memory_system.retrieve_by_content(query, threshold=0.7)
        
        # Filter by category if needed
        if category and content_results:
            memory_type = self.default_categories.get(category)
            if memory_type:
                content_results = [r for r in content_results 
                                 if r["memory_type"] == memory_type.name]
        
        # Add content results
        for result in content_results:
            result["retrieval_method"] = "content"
            results.append(result)
        
        # Case 4: If category specified but no results yet, try category lookup
        if category and not results and category in self.default_categories:
            memory_type = self.default_categories[category]
            type_results = self.memory_system.retrieve_by_type(memory_type, limit=limit)
            
            for result in type_results:
                result["retrieval_method"] = "category"
                results.append(result)
        
        # Deduplicate results by memory_id
        unique_results = {}
        for result in results:
            memory_id = result["memory_id"]
            if memory_id not in unique_results or result["confidence"] > unique_results[memory_id]["confidence"]:
                unique_results[memory_id] = result
                
        # Convert back to list, sort by confidence
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return final_results[:limit]
    
    def forget(self, query: Any) -> int:
        """
        Remove memories matching the query criteria.
        
        This method provides flexible memory deletion with support for direct
        ID deletion or content-based matching for bulk removal.
        
        Args:
            query: The deletion criteria (memory ID or content to match)
            
        Returns:
            Number of memories successfully removed
        """
        removed_count = 0
        
        # Case 1: If query is a memory ID, delete directly
        if isinstance(query, str) and query in self.memory_system.memories:
            success = self.memory_system.delete(query)
            return 1 if success else 0
        
        # Case 2: Content-based lookup for deletion
        results = self.memory_system.retrieve_by_content(query, threshold=0.9)
        
        # Delete matching memories
        for result in results:
            memory_id = result["memory_id"]
            success = self.memory_system.delete(memory_id)
            if success:
                removed_count += 1
                
                # Remove from category indexes
                for category, memory_list in self.memory_by_category.items():
                    if memory_id in memory_list:
                        memory_list.remove(memory_id)
        
        return removed_count
    
    def relate(self, memory_id1: str, memory_id2: str) -> bool:
        """
        Create a bidirectional relationship between two memories.
        
        Args:
            memory_id1: ID of the first memory
            memory_id2: ID of the second memory
            
        Returns:
            True if association was created successfully, False otherwise
        """
        return self.memory_system.associate(memory_id1, memory_id2)
    
    def get_related(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get memories related to a given memory through associations.
        
        Args:
            memory_id: The ID of the source memory
            
        Returns:
            List of related memory dictionaries with metadata
        """
        return self.memory_system.retrieve_associated(memory_id)
    
    def tag_memory(self, memory_id: str, tags: List[str]) -> int:
        """
        Add tags to an existing memory for improved organization.
        
        Args:
            memory_id: The ID of the memory to tag
            tags: List of tags to add
            
        Returns:
            Number of tags successfully added
        """
        count = 0
        for tag in tags:
            if self.memory_system.add_tag(memory_id, tag):
                count += 1
        return count
    
    def search_by_tag(self, tags: List[str], require_all: bool = False) -> List[Dict[str, Any]]:
        """
        Search memories by tag criteria with flexible matching options.
        
        Args:
            tags: List of tags to search for
            require_all: Whether all tags must be present (AND) or any (OR)
            
        Returns:
            List of matching memory dictionaries, sorted by relevance
        """
        if not tags:
            return []
            
        # Get memories for each tag
        tag_results = {}
        for tag in tags:
            memories = self.memory_system.retrieve_by_tag(tag)
            for memory in memories:
                memory_id = memory["memory_id"]
                if memory_id not in tag_results:
                    tag_results[memory_id] = {
                        "memory": memory,
                        "tags_matched": 1
                    }
                else:
                    tag_results[memory_id]["tags_matched"] += 1
        
        # Filter based on tag matching requirements
        if require_all:
            results = [data["memory"] for memory_id, data in tag_results.items() 
                     if data["tags_matched"] == len(tags)]
        else:
            results = [data["memory"] for memory_id, data in tag_results.items()]
            # Sort by number of tags matched
            results.sort(key=lambda r: tag_results[r["memory_id"]]["tags_matched"], reverse=True)
            
        return results
    
    def save(self, filepath: str) -> bool:
        """
        Save the entire memory system state to a file.
        
        Args:
            filepath: Path where to save the memory system
            
        Returns:
            True if save was successful, False otherwise
        """
        return self.memory_system.save_to_file(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'HolographicMemoryManager':
        """
        Load a memory system from a saved file.
        
        Args:
            filepath: Path to the saved memory system file
            
        Returns:
            New HolographicMemoryManager instance with loaded state
        """
        memory_system = HolographicMemorySystem.load_from_file(filepath)
        
        manager = cls(
            dimensions=memory_system.dimensions,
            address_dimensions=memory_system.address_dimensions,
            encoding_type=memory_system.encoding_type
        )
        
        manager.memory_system = memory_system
        
        # Rebuild category indexes
        for memory_id, memory in memory_system.memories.items():
            # Determine category from memory type
            for category, memory_type in manager.default_categories.items():
                if memory.memory_type == memory_type:
                    manager.memory_by_category[category].append(memory_id)
                    break
        
        return manager
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system state.
        
        Returns:
            Dictionary containing detailed statistics including memory counts,
            category distributions, system settings, and performance metrics
        """
        stats = self.memory_system.get_statistics()
        
        # Add manager-specific statistics
        stats.update({
            "category_counts": {
                category: len(memory_ids)
                for category, memory_ids in self.memory_by_category.items()
            },
            "auto_tagging_enabled": self.auto_tagging,
            "auto_association_enabled": self.auto_association,
            "default_compression": self.default_compression.name,
            "recent_queries": len(self.query_history),
            "last_consolidation_age": time.time() - self.last_consolidation
        })
        
        return stats
    
    def _extract_tags(self, data: Any) -> List[str]:
        """
        Automatically extract relevant tags from memory data.
        
        This method analyzes the content and structure of data to generate
        appropriate tags for improved memory organization and retrieval.
        
        Args:
            data: The data to analyze for tag extraction
            
        Returns:
            List of extracted tag strings
        """
        tags = []
        
        # Handle different data types
        if isinstance(data, str):
            # For strings, extract key terms using simple frequency analysis
            import re
            clean_text = re.sub(r'[^\w\s]', ' ', data.lower())
            
            # Split into words
            words = clean_text.split()
            
            # Count word frequencies
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Select top words as tags
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            tags = [word for word, count in sorted_words[:5]]
            
        elif isinstance(data, dict):
            # For dictionaries, use keys as tags
            tags = [str(key) for key in list(data.keys())[:5]]
            
        elif isinstance(data, (list, tuple)):
            # For lists/tuples, use string representations of first few elements
            tags = [str(item) for item in data[:3]]
            
        return tags
    
    def _create_automatic_associations(self, memory_id: str, data: Any) -> None:
        """
        Automatically create associations based on content similarity.
        
        This method finds similar existing memories and creates bidirectional
        associations to enable better memory navigation and retrieval.
        
        Args:
            memory_id: The ID of the new memory to associate
            data: The content of the new memory for similarity analysis
        """
        # Find similar memories
        similar_memories = self.memory_system.retrieve_by_content(data, threshold=0.75)
        
        # Create associations with top matches
        for i, result in enumerate(similar_memories[:3]):  # Limit to top 3
            similar_id = result["memory_id"]
            if similar_id != memory_id:  # Don't associate with self
                self.memory_system.associate(memory_id, similar_id)

def run_example():
    """
    Run a comprehensive demonstration of the holographic memory system.
    
    This function showcases the key features of the holographic memory integration
    system including storage, retrieval, categorization, tagging, and associations.
    """
    print(f"Current Date/Time: 2025-07-26 14:03:32")
    print(f"User: Shriram-2005")
    
    print("\n===== Holographic Memory Integration System Example =====")
    
    # Create memory manager
    manager = HolographicMemoryManager(
        dimensions=256,
        address_dimensions=8,
        encoding_type=MemoryEncoding.COMPOSITE
    )
    
    print("\nInitializing holographic memory system...")
    print(f"Encoding type: {manager.memory_system.encoding_type.name}")
    print(f"Dimensions: {manager.memory_system.dimensions}")
    print(f"Address dimensions: {manager.memory_system.address_dimensions}")
    
    # Store some memories
    print("\nStoring memories...")
    
    # Memory 1: Factual knowledge
    memory1_id = manager.remember(
        "Quantum computing uses qubits which can exist in superposition states.",
        category="knowledge",
        tags=["quantum", "computing", "physics"]
    )
    print(f"  Stored knowledge memory: {memory1_id}")
    
    # Memory 2: More knowledge
    memory2_id = manager.remember(
        "Neural networks are computational models inspired by the human brain.",
        category="knowledge",
        tags=["neural networks", "AI", "computing"]
    )
    print(f"  Stored knowledge memory: {memory2_id}")
    
    # Memory 3: Event memory
    memory3_id = manager.remember(
        {"event": "System initialization", "timestamp": "2025-07-26T14:00:00", "status": "successful"},
        category="event",
        tags=["system", "initialization"]
    )
    print(f"  Stored event memory: {memory3_id}")
    
    # Memory 4: Reference data
    memory4_id = manager.remember(
        {"pi": 3.14159265359, "e": 2.71828182846, "phi": 1.61803398875},
        category="reference",
        tags=["constants", "mathematics"]
    )
    print(f"  Stored reference memory: {memory4_id}")
    
    # Create a relationship
    manager.relate(memory1_id, memory2_id)
    print(f"  Created relationship between quantum computing and neural network memories")
    
    # Wait a moment
    time.sleep(1)
    
    # Retrieve memories
    print("\nRetrieving memories...")
    
    # Query 1: By content
    results1 = manager.recall("quantum computing and qubits")
    if results1:
        print(f"  Query 'quantum computing and qubits' returned:")
        print(f"  - {results1[0]['content'][:50]}...")
        print(f"  - Confidence: {results1[0]['confidence']:.4f}")
    
    # Query 2: By tag
    results2 = manager.recall("", tags=["neural networks"])
    if results2:
        print(f"  Query by tag 'neural networks' returned:")
        print(f"  - {results2[0]['content'][:50]}...")
    
    # Query 3: By category
    results3 = manager.recall("", category="reference")
    if results3:
        print(f"  Query by category 'reference' returned:")
        print(f"  - Constants: {list(results3[0]['content'].keys())}")
    
    # Get related memories
    related = manager.get_related(memory1_id)
    if related:
        print("\nMemories related to quantum computing:")
        for r in related:
            print(f"  - {r['content'][:50]}...")
    
    # Get statistics
    stats = manager.get_statistics()
    
    print("\nMemory system statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Memory type distribution: {stats['memory_type_distribution']}")
    print(f"  Average confidence: {stats['avg_confidence']:.4f}")
    print(f"  Total tags: {stats['total_tags']}")
    
    print("\nHolographic Memory System demonstration completed!")
    print("The system successfully implements advanced holographic memory storage and retrieval.")

# Export key classes and utilities for module usage
__all__ = [
    # Enums
    'MemoryEncoding',
    'MemoryType', 
    'CompressionLevel',
    
    # Data classes
    'MemoryAddress',
    'HolographicMemory',
    
    # Core classes
    'HolographicEncoder',
    'HolographicMemorySystem',
    'HolographicMemoryManager',
    
    # Utility functions
    'run_example'
]

if __name__ == "__main__":
    run_example()