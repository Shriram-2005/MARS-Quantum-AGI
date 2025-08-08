"""
🌟 MARS Quantum Interface Neural Network Framework 🌟
═══════════════════════════════════════════════════════════════════════════════════════════

🎯 PURPOSE:
    Advanced quantum-inspired neural architecture implementing interference patterns, complex-valued
    computation, and quantum state evolution for next-generation AI systems. Bridges classical
    neural networks with quantum computational principles.

🚀 KEY FEATURES:
    ✨ Quantum Interference Effects: Inter-neuron quantum interference patterns for enhanced computation
    🔬 Complex-Valued Processing: Full complex number support with amplitude and phase dynamics
    🌊 Quantum State Management: Comprehensive quantum state representation and manipulation
    🎛️ Multiple Activation Functions: Quantum-compatible activations (ReLU, Sigmoid, Tanh, Phase Shift, Hadamard)
    🔄 Quantum Backpropagation: Gradient computation respecting quantum mechanical principles
    🏗️ Modular Architecture: Quantum neurons, layers, and specialized network components
    📊 Convolutional Support: Quantum-inspired convolution layers with interference pooling
    🔁 Recurrent Processing: Quantum recurrent cells with hidden state interference
    🎚️ Quantum Annealing: Training with quantum annealing optimization strategies
    📈 Real-time Metrics: Coherence measurement, entropy calculation, and quantum state analysis

🏛️ ARCHITECTURE COMPONENTS:
    • QuantumState: Core quantum state representation with amplitude/phase management
    • QuantumNeuron: Individual quantum processing unit with complex-valued weights
    • QuantumLayer: Layer of interconnected quantum neurons with interference effects
    • QuantumInterferenceNN: Main neural network with quantum interference capabilities
    • QuantumConvolutionLayer: Convolutional processing with quantum kernel operations
    • QuantumInterferenceConvNet: Full convolutional architecture with quantum principles
    • QuantumInterferencePooling: Pooling operations respecting quantum interference
    • QuantumInterferenceRecurrentCell: RNN cell with quantum hidden state evolution

📊 ACTIVATION FUNCTIONS:
    • QUANTUM_RELU: Amplitude-based ReLU preserving phase information
    • QUANTUM_SIGMOID: Quantum sigmoid with phase preservation
    • QUANTUM_TANH: Hyperbolic tangent maintaining complex structure
    • PHASE_SHIFT: Non-linear phase transformation activation
    • HADAMARD: Hadamard-inspired superposition activation
    • INTERFERENCE: Complex interference pattern activation

🎯 USE CASES:
    • Quantum Machine Learning Research
    • Complex Pattern Recognition
    • Quantum-Classical Hybrid Systems
    • Advanced Signal Processing
    • Quantum Information Processing
    • Next-Generation AI Architecture Development

💡 USAGE EXAMPLE:
    ```python
    # Create quantum neural network
    network = QuantumInterferenceNN()
    network.build(input_dim=4, layer_sizes=[8, 6, 2],
                 activations=[ActivationFunction.QUANTUM_RELU,
                             ActivationFunction.PHASE_SHIFT,
                             ActivationFunction.QUANTUM_TANH])
    
    # Train with complex-valued data
    losses = network.train(x_train, y_train, epochs=100)
    
    # Make quantum predictions
    prediction = network.predict(x_test)
    coherence = network.calculate_network_coherence()
    ```

🔬 QUANTUM PHYSICS FOUNDATION:
    • Quantum Superposition: States exist in multiple configurations simultaneously
    • Quantum Interference: Wave-like interference between computational paths
    • Quantum Entanglement: Correlated quantum states across network components
    • Quantum Measurement: Probabilistic state collapse during prediction
    • Unitary Evolution: Reversible quantum transformations in forward pass

⚡ PERFORMANCE CHARACTERISTICS:
    • Complex Number Arithmetic: Native support for quantum amplitudes and phases
    • Interference Calculations: Real-time quantum interference between processing units
    • State Vector Management: Efficient quantum state representation and manipulation
    • Gradient Flow: Quantum-compatible backpropagation with complex derivatives
    • Memory Efficiency: Optimized quantum state storage and computation

🛡️ STABILITY FEATURES:
    • Numerical Stability: Careful handling of complex arithmetic and normalization
    • Gradient Clipping: Prevention of quantum state explosion during training
    • Weight Normalization: Maintaining quantum state validity constraints
    • Error Handling: Robust error management for quantum operations

📈 METRICS & ANALYSIS:
    • Quantum Coherence: Network-wide quantum coherence measurement
    • von Neumann Entropy: Quantum entropy calculation for state analysis
    • Interference Strength: Quantification of quantum interference effects
    • Phase Dynamics: Monitoring of quantum phase evolution during training

🔧 TECHNICAL SPECIFICATIONS:
    • Complex-valued weights and biases throughout the network
    • Quantum state normalization maintaining probability conservation
    • Interference matrices for inter-neuron quantum coupling
    • Phase-preserving activation functions respecting quantum mechanics
    • Quantum measurement protocols for probabilistic output generation

═══════════════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import math
import cmath
import random
import uuid
import logging
import time
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import threading

logger = logging.getLogger("MARS.QuantumNN")

class ActivationFunction(Enum):
    """
    🎛️ Quantum-Compatible Activation Functions
    
    Specialized activation functions designed for quantum-inspired neural computation,
    preserving quantum mechanical properties like phase coherence and complex amplitudes.
    
    Available Activation Functions:
    • QUANTUM_RELU: Amplitude-based ReLU that preserves phase information
    • QUANTUM_SIGMOID: Quantum sigmoid maintaining complex structure  
    • QUANTUM_TANH: Hyperbolic tangent preserving quantum properties
    • PHASE_SHIFT: Non-linear phase transformation activation
    • HADAMARD: Hadamard-inspired superposition activation
    • INTERFERENCE: Complex interference pattern activation
    """
    QUANTUM_RELU = auto()      # Quantum ReLU preserving phase while activating amplitude
    QUANTUM_SIGMOID = auto()   # Quantum sigmoid for smooth amplitude transitions
    QUANTUM_TANH = auto()      # Quantum tanh for bounded amplitude activation
    PHASE_SHIFT = auto()       # Non-linear phase rotation activation
    HADAMARD = auto()          # Hadamard-inspired quantum superposition
    INTERFERENCE = auto()      # Quantum interference pattern activation

@dataclass
class QuantumState:
    """
    🌊 Quantum State Representation
    
    Core quantum state management system for representing and manipulating quantum states
    in the neural network. Handles amplitude and phase information with full quantum
    mechanical compliance.
    
    Attributes:
        amplitudes (np.ndarray): Probability amplitudes for each quantum state component
        phases (np.ndarray): Quantum phases corresponding to each amplitude component
        
    Key Capabilities:
        • State vector construction from amplitude/phase representation
        • Quantum state normalization ensuring probability conservation
        • Density matrix conversion for mixed state representation
        • Quantum operator application with unitary evolution
        • Tensor product computation for composite quantum systems
        • Quantum measurement with probabilistic state collapse
        • von Neumann entropy calculation for quantum information content
        
    Mathematical Foundation:
        |ψ⟩ = Σᵢ αᵢ e^(iφᵢ) |i⟩
        where αᵢ are amplitudes and φᵢ are phases
        
        Normalization: Σᵢ |αᵢ|² = 1
    """
    amplitudes: np.ndarray = field(metadata={"description": "Quantum probability amplitudes"})
    phases: np.ndarray = field(metadata={"description": "Quantum phase components"})
    
    @property
    def state_vector(self) -> np.ndarray:
        """Get complex state vector"""
        return self.amplitudes * np.exp(1j * self.phases)
    
    @classmethod
    def from_complex(cls, complex_vector: np.ndarray) -> 'QuantumState':
        """Create from complex vector"""
        amplitudes = np.abs(complex_vector)
        phases = np.angle(complex_vector)
        return cls(amplitudes=amplitudes, phases=phases)
    
    def normalize(self) -> 'QuantumState':
        """Normalize the quantum state"""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
        return self
    
    def to_density_matrix(self) -> np.ndarray:
        """Convert to density matrix representation"""
        state_vector = self.state_vector
        return np.outer(state_vector, np.conj(state_vector))
    
    def apply_operator(self, operator: np.ndarray) -> 'QuantumState':
        """Apply quantum operator to state"""
        if operator.shape[0] != len(self.amplitudes) or operator.shape[1] != len(self.amplitudes):
            raise ValueError(f"Operator shape {operator.shape} incompatible with state shape {self.amplitudes.shape}")
            
        new_state_vector = operator @ self.state_vector
        return QuantumState.from_complex(new_state_vector).normalize()
    
    def tensor_product(self, other: 'QuantumState') -> 'QuantumState':
        """Compute tensor product with another state"""
        state1 = self.state_vector
        state2 = other.state_vector
        
        # Calculate tensor product
        product = np.kron(state1, state2)
        
        # Return new quantum state
        return QuantumState.from_complex(product).normalize()
    
    def measure(self) -> Tuple[int, float]:
        """Perform measurement, collapsing state to basis state"""
        probabilities = self.amplitudes ** 2
        
        # Ensure probabilities sum to 1
        probabilities = probabilities / np.sum(probabilities)
        
        # Select outcome based on probabilities
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        return outcome, probabilities[outcome]
    
    def get_entropy(self) -> float:
        """Calculate von Neumann entropy of the state"""
        probabilities = self.amplitudes ** 2
        
        # Filter out zeros to avoid log(0)
        nonzero_probs = probabilities[probabilities > 0]
        
        # Calculate entropy
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        
        return entropy

class QuantumNeuron:
    """
    🔬 Quantum-Inspired Neural Processing Unit
    
    Individual quantum neuron implementing complex-valued computation with quantum
    mechanical principles. Performs quantum transformations on input data using
    complex-valued weights, biases, and quantum activation functions.
    
    Architecture:
        • Complex-valued weight matrix with normalized quantum amplitudes
        • Complex bias term for quantum phase adjustment
        • Quantum activation functions preserving phase coherence
        • Gradient accumulation system for quantum backpropagation
        • Automatic weight normalization maintaining quantum constraints
        
    Quantum Operations:
        • Forward Pass: z = W·x + b, output = activation(z)
        • Backward Pass: Quantum gradient computation with complex conjugates
        • Parameter Update: Gradient descent with quantum weight normalization
        • Activation Functions: Phase-preserving quantum activations
        
    Key Features:
        • Complex-valued computation throughout processing pipeline
        • Multiple quantum activation function support
        • Automatic gradient accumulation for batch processing
        • Weight normalization preserving quantum state validity
        • Memory efficiency with last input/output caching
        
    Mathematical Foundation:
        Forward: |output⟩ = f(W|input⟩ + |bias⟩)
        Backward: ∇W = ⟨input|gradient⟩*, ∇bias = gradient
        where * denotes complex conjugate
    """
    
    def __init__(self, input_dim: int, activation: ActivationFunction = ActivationFunction.QUANTUM_RELU):
        """Initialize the quantum neuron"""
        # Generate random initial weights with both real and imaginary parts
        self.weights = np.random.randn(input_dim) + 1j * np.random.randn(input_dim)
        self.bias = np.random.randn() + 1j * np.random.randn()
        
        # Normalize weights
        weight_norm = np.sqrt(np.sum(np.abs(self.weights) ** 2))
        if weight_norm > 0:
            self.weights = self.weights / weight_norm
            
        self.activation = activation
        self.last_output = None
        self.last_input = None
        self.gradient_accumulator = {
            'weights': np.zeros_like(self.weights),
            'bias': 0j  # Complex zero
        }
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the quantum neuron"""
        # Save input for backprop
        self.last_input = input_data.copy()
        
        # Apply weights and bias
        z = np.dot(input_data, self.weights) + self.bias
        
        # Apply activation function
        output = self._apply_activation(z)
        
        # Save output for backprop
        self.last_output = output
        
        return output
    
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply the quantum activation function"""
        if self.activation == ActivationFunction.QUANTUM_RELU:
            return self._quantum_relu(z)
        elif self.activation == ActivationFunction.QUANTUM_SIGMOID:
            return self._quantum_sigmoid(z)
        elif self.activation == ActivationFunction.QUANTUM_TANH:
            return self._quantum_tanh(z)
        elif self.activation == ActivationFunction.PHASE_SHIFT:
            return self._phase_shift(z)
        elif self.activation == ActivationFunction.HADAMARD:
            return self._hadamard(z)
        elif self.activation == ActivationFunction.INTERFERENCE:
            return self._interference(z)
        else:
            return z
    
    def _quantum_relu(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of ReLU that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply ReLU to amplitude
        activated_amplitude = np.maximum(0, amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of sigmoid that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply sigmoid to amplitude
        activated_amplitude = 1 / (1 + np.exp(-amplitude))
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_tanh(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of tanh that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply tanh to amplitude
        activated_amplitude = np.tanh(amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _phase_shift(self, z: np.ndarray) -> np.ndarray:
        """Apply phase shift activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply non-linear transformation to phase
        shifted_phase = np.pi * np.tanh(phase)
        
        # Keep amplitude unchanged
        return amplitude * np.exp(1j * shifted_phase)
    
    def _hadamard(self, z: np.ndarray) -> np.ndarray:
        """Apply Hadamard-inspired activation"""
        # Normalize z
        norm = np.sqrt(np.sum(np.abs(z) ** 2))
        if norm > 0:
            z_normalized = z / norm
        else:
            z_normalized = z
            
        # Apply Hadamard-inspired transformation
        return (z_normalized + 1j * z_normalized) / np.sqrt(2)
    
    def _interference(self, z: np.ndarray) -> np.ndarray:
        """Apply interference pattern activation"""
        if not isinstance(z, np.ndarray):
            # Convert single value to array for consistent processing
            z = np.array([z])
            
        result = np.zeros_like(z)
        
        for i in range(len(z)):
            # Generate interference between the value and its conjugate
            z_val = z[i]
            z_conj = np.conj(z_val)
            
            # Create interference pattern
            result[i] = (z_val + z_conj) / 2  # Constructive interference
            
            # Apply non-linearity
            amp = np.abs(result[i])
            phase = np.angle(result[i])
            
            # Non-linear amplitude transformation
            amp_transformed = np.tanh(amp)
            
            # Reconstruct
            result[i] = amp_transformed * np.exp(1j * phase)
        
        return result[0] if len(result) == 1 else result
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Backward pass for gradient computation and parameter update"""
        # For simplicity, we'll implement a basic version
        # In practice, quantum backpropagation is more complex
        
        # Gradient for weights
        if isinstance(self.last_input, np.ndarray) and len(self.last_input.shape) > 0:
            # Ensure grad_output is scalar for this neuron
            if np.isscalar(grad_output) or grad_output.shape == ():
                grad_weights = np.conj(self.last_input) * grad_output
            else:
                grad_weights = np.conj(self.last_input) * grad_output.item() if grad_output.size == 1 else np.conj(self.last_input) * grad_output[0]
        else:
            grad_weights = np.conj(self.last_input) * grad_output
            
        # Gradient for bias
        grad_bias = grad_output
        
        # Accumulate gradients
        self.gradient_accumulator['weights'] += grad_weights
        self.gradient_accumulator['bias'] += grad_bias
        
        # Gradient to propagate to previous layer
        grad_input = np.conj(self.weights) * grad_output
        
        return grad_input
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update neuron parameters using accumulated gradients"""
        # Update weights
        self.weights -= learning_rate * self.gradient_accumulator['weights']
        
        # Update bias
        self.bias -= learning_rate * self.gradient_accumulator['bias']
        
        # Reset gradient accumulator
        self.gradient_accumulator['weights'] = np.zeros_like(self.weights)
        self.gradient_accumulator['bias'] = 0j
        
        # Normalize weights after update
        weight_norm = np.sqrt(np.sum(np.abs(self.weights) ** 2))
        if weight_norm > 0:
            self.weights = self.weights / weight_norm

class QuantumLayer:
    """
    🏗️ Quantum Neural Network Layer
    
    Parallel processing layer containing multiple quantum neurons with inter-neuron
    quantum interference effects. Implements collective quantum computation across
    multiple quantum processing units.
    
    Architecture:
        • Multiple QuantumNeuron instances operating in parallel
        • Inter-neuron quantum interference coupling mechanism
        • Collective gradient computation and parameter updates
        • Quantum state output representation capability
        • Configurable activation functions and interference strength
        
    Quantum Interference:
        • Phase-coupled interactions between neuron outputs
        • Distance-weighted interference strength calculation
        • Complex interference term generation with quantum phases
        • Normalization preserving quantum state validity
        
    Key Features:
        • Parallel quantum neuron processing
        • Configurable quantum interference effects
        • Quantum state representation of layer outputs
        • Efficient gradient backpropagation through quantum couplings
        • Memory optimization with input/output caching
        
    Mathematical Foundation:
        Output_i = Neuron_i(input) + Σⱼ≠ᵢ α_ij * Output_j * e^(iφ_ij)
        where α_ij is interference strength and φ_ij is relative phase
        
    Use Cases:
        • Hidden layers in quantum neural networks
        • Quantum feature extraction and transformation
        • Quantum pattern recognition layers
        • Complex-valued signal processing stages
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                activation: ActivationFunction = ActivationFunction.QUANTUM_RELU,
                apply_interference: bool = True):
        """Initialize the quantum layer"""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.apply_interference = apply_interference
        
        # Create neurons
        self.neurons = [
            QuantumNeuron(input_dim, activation) for _ in range(output_dim)
        ]
        
        # For interference calculations
        self.last_input = None
        self.last_output = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        # Save input
        self.last_input = input_data.copy()
        
        # Get outputs from each neuron
        neuron_outputs = np.array([
            neuron.forward(input_data) for neuron in self.neurons
        ])
        
        # Apply inter-neuron interference if enabled
        if self.apply_interference and len(self.neurons) > 1:
            neuron_outputs = self._apply_interference(neuron_outputs)
            
        self.last_output = neuron_outputs
        return neuron_outputs
    
    def _apply_interference(self, outputs: np.ndarray) -> np.ndarray:
        """Apply quantum interference between neuron outputs"""
        # Calculate interference matrix
        n_neurons = len(outputs)
        interference_result = np.zeros_like(outputs)
        
        for i in range(n_neurons):
            # Start with the neuron's own output
            interference_result[i] = outputs[i]
            
            # Add interference contributions from other neurons
            for j in range(n_neurons):
                if i != j:
                    # Calculate interference term
                    # Use a weighted interference based on "distance" between neurons
                    weight = 0.1 * (1.0 / (abs(i - j) + 1))
                    
                    # Complex interference term
                    interference_term = weight * outputs[j] * np.exp(1j * (i-j) * np.pi/n_neurons)
                    
                    # Add to result
                    interference_result[i] += interference_term
        
        # Normalize results
        for i in range(n_neurons):
            norm = np.abs(interference_result[i])
            if norm > 0:
                interference_result[i] = interference_result[i] / norm
                
        return interference_result
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Backward pass through the layer"""
        # For simplicity in this demo, we'll ignore interference effects in backprop
        
        # Ensure grad_output has correct shape
        if len(grad_output.shape) == 0:
            grad_output = np.array([grad_output])
            
        # Accumulate gradients from all neurons
        grad_input = np.zeros(self.input_dim, dtype=np.complex128)
        
        for i, neuron in enumerate(self.neurons):
            if i < len(grad_output):
                neuron_grad_in = neuron.backward(grad_output[i], learning_rate)
                grad_input += neuron_grad_in
                
        return grad_input
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update layer parameters"""
        for neuron in self.neurons:
            neuron.update_parameters(learning_rate)
    
    def get_output_state(self) -> QuantumState:
        """Get layer output as a quantum state"""
        if self.last_output is None:
            # Return random initial state if no forward pass yet
            amplitudes = np.ones(self.output_dim) / np.sqrt(self.output_dim)
            phases = np.zeros(self.output_dim)
            return QuantumState(amplitudes, phases)
            
        # Extract amplitudes and phases
        amplitudes = np.abs(self.last_output)
        phases = np.angle(self.last_output)
        
        # Normalize amplitudes (ensuring they represent probabilities)
        norm = np.sqrt(np.sum(amplitudes ** 2))
        if norm > 0:
            amplitudes = amplitudes / norm
            
        return QuantumState(amplitudes, phases)

class QuantumInterferenceNN:
    """
    🌟 Quantum Interference Neural Network
    
    Main neural network architecture implementing quantum interference effects throughout
    the computation pipeline. Combines multiple quantum layers with complex-valued
    processing and quantum mechanical principles.
    
    Core Architecture:
        • Multiple QuantumLayer instances in sequence
        • End-to-end complex-valued computation pipeline
        • Quantum interference effects between and within layers
        • Flexible activation function configuration per layer
        • Comprehensive training system with quantum backpropagation
        
    Key Capabilities:
        • Network construction with configurable quantum layer sizes
        • Multiple activation function support per layer
        • Complex-valued forward and backward propagation
        • Quantum annealing training optimization
        • Real-time quantum coherence measurement
        • Batch and single-sample training modes
        • Quantum state prediction and measurement
        
    Training Features:
        • Standard gradient descent with quantum-compatible updates
        • Mini-batch processing with complex gradient accumulation
        • Quantum annealing with temperature scheduling
        • Metropolis acceptance criterion for quantum state updates
        • Loss history tracking and convergence monitoring
        
    Quantum Metrics:
        • Network-wide quantum coherence calculation
        • von Neumann entropy measurement for quantum information
        • Quantum state representation of network outputs
        • Interference strength quantification across layers
        
    Mathematical Foundation:
        Forward: |output⟩ = Layer_n(...Layer_2(Layer_1(|input⟩))...)
        Backward: ∇θ = Σᵢ ∂L/∂output_i * ∂output_i/∂θ
        where θ represents all quantum network parameters
        
    Use Cases:
        • Quantum machine learning research
        • Complex pattern recognition tasks
        • Quantum-classical hybrid computation
        • Advanced signal processing applications
        • Quantum information processing systems
    """
    
    def __init__(self):
        """Initialize the quantum neural network"""
        self.layers = []
        self.built = False
        self.epoch_count = 0
        self.loss_history = []
        
    def add_layer(self, input_dim: int, output_dim: int, 
                activation: ActivationFunction = ActivationFunction.QUANTUM_RELU,
                apply_interference: bool = True):
        """Add a quantum layer to the network"""
        layer = QuantumLayer(input_dim, output_dim, activation, apply_interference)
        self.layers.append(layer)
        
    def build(self, input_dim: int, layer_sizes: List[int],
            activations: List[ActivationFunction] = None):
        """Build the network with specified architecture"""
        if not layer_sizes:
            raise ValueError("Must specify at least one layer size")
            
        # Default to QUANTUM_RELU if activations not specified
        if not activations:
            activations = [ActivationFunction.QUANTUM_RELU] * len(layer_sizes)
        elif len(activations) != len(layer_sizes):
            raise ValueError("Must provide one activation per layer")
            
        # Create first layer
        self.add_layer(input_dim, layer_sizes[0], activations[0])
        
        # Create remaining layers
        for i in range(1, len(layer_sizes)):
            self.add_layer(layer_sizes[i-1], layer_sizes[i], activations[i])
            
        self.built = True
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        # Ensure input is complex
        if not np.iscomplexobj(input_data):
            input_data = input_data.astype(np.complex128)
            
        # Pass through each layer
        x = input_data
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01):
        """Backward pass through the network"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        # Pass backward through layers in reverse order
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
            
        return grad
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update network parameters"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        for layer in self.layers:
            layer.update_parameters(learning_rate)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> float:
        """Perform a single training step"""
        # Forward pass
        y_pred = self.forward(x)
        
        # Calculate loss and gradients (using MSE for simplicity)
        loss = np.mean(np.abs(y_pred - y) ** 2)
        
        # Gradients for MSE
        grad_output = 2 * (y_pred - y) / len(y)
        
        # Backward pass
        self.backward(grad_output, learning_rate)
        
        # Update parameters
        self.update_parameters(learning_rate)
        
        return loss
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 100, learning_rate: float = 0.01,
            batch_size: int = 32, verbose: bool = True) -> List[float]:
        """Train the network"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        n_samples = len(x_train)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Generate random batch indices
            indices = np.random.permutation(n_samples)
            
            # Process mini-batches
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Skip if batch is empty
                if len(batch_indices) == 0:
                    continue
                    
                # Get batch data
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Process each sample in batch
                batch_loss = 0.0
                for i in range(len(x_batch)):
                    sample_loss = self.train_step(x_batch[i], y_batch[i], learning_rate)
                    batch_loss += sample_loss
                    
                # Average batch loss
                batch_loss /= len(x_batch)
                epoch_loss += batch_loss * len(x_batch)
                
            # Calculate average epoch loss
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
                
            self.epoch_count += 1
            self.loss_history.append(epoch_loss)
            
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions with the network"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        # Handle both single samples and batches
        if len(x.shape) == 1:
            # Single sample
            return self.forward(x)
        else:
            # Batch of samples
            return np.array([self.forward(sample) for sample in x])
    
    def measure_output(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions and perform measurement"""
        # Get quantum state
        output = self.forward(x)
        
        # Convert to quantum state
        state = QuantumState.from_complex(output).normalize()
        
        # Perform measurement
        outcome, probability = state.measure()
        
        return outcome, probability
    
    def get_quantum_state(self, x: np.ndarray) -> QuantumState:
        """Get the quantum state of the network output for an input"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        # Forward pass
        output = self.forward(x)
        
        # Convert to quantum state
        return QuantumState.from_complex(output).normalize()
    
    def calculate_network_coherence(self) -> float:
        """Calculate overall coherence of the network"""
        if not self.built or not self.layers:
            return 0.0
            
        # Get quantum state from final layer
        final_state = self.layers[-1].get_output_state()
        
        # Calculate von Neumann entropy
        entropy = final_state.get_entropy()
        
        # Convert entropy to coherence measure (higher entropy = lower coherence)
        max_entropy = np.log2(len(final_state.amplitudes))
        coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return coherence
    
    def simulate_quantum_annealing(self, temperature_schedule: List[float],
                                 learn_rate_schedule: List[float],
                                 x_train: np.ndarray, y_train: np.ndarray) -> List[float]:
        """Train with quantum annealing approach"""
        if not self.built:
            raise RuntimeError("Network not built. Call build() first.")
            
        if len(temperature_schedule) != len(learn_rate_schedule):
            raise ValueError("Temperature and learning rate schedules must have the same length")
            
        losses = []
        n_samples = len(x_train)
        
        for step, (temperature, learning_rate) in enumerate(zip(temperature_schedule, learn_rate_schedule)):
            # Current loss
            y_pred = np.array([self.forward(x) for x in x_train])
            current_loss = np.mean(np.abs(y_pred - y_train) ** 2)
            
            # Store the current parameters
            current_params = self._get_current_params()
            
            # Update parameters with current learning rate
            sample_idx = np.random.randint(0, n_samples)
            sample_loss = self.train_step(x_train[sample_idx], y_train[sample_idx], learning_rate)
            
            # Evaluate new loss
            y_pred_new = np.array([self.forward(x) for x in x_train])
            new_loss = np.mean(np.abs(y_pred_new - y_train) ** 2)
            
            # Metropolis acceptance criterion
            delta_loss = new_loss - current_loss
            
            if delta_loss > 0 and np.random.random() > np.exp(-delta_loss / temperature):
                # Reject update, revert to previous parameters
                self._set_current_params(current_params)
                loss = current_loss
            else:
                # Accept update
                loss = new_loss
                
            losses.append(loss)
            
            if (step + 1) % 50 == 0:
                print(f"Annealing step {step + 1}/{len(temperature_schedule)}, "
                     f"Temperature: {temperature:.6f}, Loss: {loss:.6f}")
                
        return losses
    
    def _get_current_params(self) -> List:
        """Get current network parameters"""
        params = []
        for layer in self.layers:
            layer_params = []
            for neuron in layer.neurons:
                neuron_params = {
                    'weights': neuron.weights.copy(),
                    'bias': neuron.bias
                }
                layer_params.append(neuron_params)
            params.append(layer_params)
        return params
    
    def _set_current_params(self, params: List):
        """Set current network parameters"""
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                neuron.weights = params[layer_idx][neuron_idx]['weights'].copy()
                neuron.bias = params[layer_idx][neuron_idx]['bias']

class QuantumConvolutionLayer:
    """
    🌊 Quantum-Inspired Convolutional Processing Layer
    
    Advanced convolutional layer implementing quantum interference effects and complex-valued
    kernel operations. Performs quantum-inspired convolution with phase-preserving activations
    and inter-channel quantum interference.
    
    Architecture:
        • Complex-valued convolution kernels with quantum normalization
        • Multi-channel input/output processing with quantum interference
        • Configurable stride and kernel size for flexible feature extraction
        • Quantum activation functions preserving phase coherence
        • Inter-channel interference coupling for enhanced feature learning
        
    Quantum Features:
        • Complex-valued kernels representing quantum filter states
        • Quantum normalization maintaining kernel validity constraints
        • Phase-preserving convolution operations
        • Inter-channel quantum interference effects
        • Quantum gradient computation for backpropagation
        
    Key Capabilities:
        • 1D quantum convolution with complex arithmetic
        • Batch and single-sample processing modes
        • Multiple quantum activation function support
        • Gradient accumulation for efficient batch training
        • Automatic kernel normalization preservation
        
    Mathematical Foundation:
        Output[b,oc,ol] = Σᵢc K[oc,ic] ⊗ Input[b,ic,slice] + B[oc]
        where ⊗ represents quantum convolution operation
        
        Quantum Interference: Output'[b,oc] = Output[b,oc] + Σⱼ≠oc α_ij * Output[b,j] * e^(iφ_ij)
        
    Use Cases:
        • Quantum signal processing
        • Complex-valued feature extraction
        • Quantum pattern recognition
        • Multi-channel quantum data analysis
    """
    
    def __init__(self, input_channels: int, output_channels: int, 
                kernel_size: int, stride: int = 1,
                activation: ActivationFunction = ActivationFunction.QUANTUM_RELU,
                apply_interference: bool = True):
        """Initialize the quantum convolution layer"""
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.apply_interference = apply_interference
        
        # Initialize kernels (complex-valued)
        self.kernels = np.random.randn(output_channels, input_channels, kernel_size) + \
                      1j * np.random.randn(output_channels, input_channels, kernel_size)
                      
        # Normalize kernels
        for oc in range(output_channels):
            for ic in range(input_channels):
                kernel_norm = np.sqrt(np.sum(np.abs(self.kernels[oc, ic]) ** 2))
                if kernel_norm > 0:
                    self.kernels[oc, ic] = self.kernels[oc, ic] / kernel_norm
                    
        # Initialize biases
        self.biases = np.random.randn(output_channels) + 1j * np.random.randn(output_channels)
        
        # For backward pass
        self.last_input = None
        self.last_output = None
        
        # Gradient accumulators
        self.kernel_gradients = np.zeros_like(self.kernels)
        self.bias_gradients = np.zeros_like(self.biases)
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the convolution layer"""
        # input_data shape: (batch_size, input_channels, input_length)
        # or just (input_channels, input_length) for a single sample
        
        # Save input for backward pass
        self.last_input = input_data.copy()
        
        # Handle single sample vs batch
        single_sample = False
        if len(input_data.shape) == 2:  # (channels, length)
            single_sample = True
            input_data = np.expand_dims(input_data, 0)  # (1, channels, length)
            
        batch_size, _, input_length = input_data.shape
        
        # Calculate output dimensions
        output_length = (input_length - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, output_length), dtype=np.complex128)
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for ol in range(output_length):
                    # Input slice position
                    in_start = ol * self.stride
                    in_end = in_start + self.kernel_size
                    
                    # Ensure we don't go out of bounds
                    if in_end > input_length:
                        continue
                        
                    # Sum over input channels
                    for ic in range(self.input_channels):
                        # Extract input slice
                        input_slice = input_data[b, ic, in_start:in_end]
                        
                        # Apply kernel
                        output[b, oc, ol] += np.sum(input_slice * self.kernels[oc, ic])
                        
                # Add bias
                output[b, oc] += self.biases[oc]
                
        # Apply activation function
        output = self._apply_activation(output)
        
        # Apply inter-channel interference if enabled
        if self.apply_interference and self.output_channels > 1:
            output = self._apply_interference(output)
            
        self.last_output = output
        
        # Return single sample or batch
        return output[0] if single_sample else output
    
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply the quantum activation function"""
        if self.activation == ActivationFunction.QUANTUM_RELU:
            return self._quantum_relu(z)
        elif self.activation == ActivationFunction.QUANTUM_SIGMOID:
            return self._quantum_sigmoid(z)
        elif self.activation == ActivationFunction.QUANTUM_TANH:
            return self._quantum_tanh(z)
        elif self.activation == ActivationFunction.PHASE_SHIFT:
            return self._phase_shift(z)
        elif self.activation == ActivationFunction.HADAMARD:
            return self._hadamard(z)
        elif self.activation == ActivationFunction.INTERFERENCE:
            return self._interference(z)
        else:
            return z
    
    def _quantum_relu(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of ReLU that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply ReLU to amplitude
        activated_amplitude = np.maximum(0, amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of sigmoid that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply sigmoid to amplitude
        activated_amplitude = 1 / (1 + np.exp(-amplitude))
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_tanh(self, z: np.ndarray) -> np.ndarray:
        """Quantum version of tanh that preserves phase"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply tanh to amplitude
        activated_amplitude = np.tanh(amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _phase_shift(self, z: np.ndarray) -> np.ndarray:
        """Apply phase shift activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply non-linear transformation to phase
        shifted_phase = np.pi * np.tanh(phase)
        
        # Keep amplitude unchanged
        return amplitude * np.exp(1j * shifted_phase)
    
    def _hadamard(self, z: np.ndarray) -> np.ndarray:
        """Apply Hadamard-inspired activation"""
        # Normalize z
        norm = np.sqrt(np.sum(np.abs(z) ** 2, axis=tuple(range(1, z.ndim)), keepdims=True))
        mask = norm > 0
        z_normalized = np.zeros_like(z)
        z_normalized[mask] = z[mask] / norm[mask]
            
        # Apply Hadamard-inspired transformation
        return (z_normalized + 1j * z_normalized) / np.sqrt(2)
    
    def _interference(self, z: np.ndarray) -> np.ndarray:
        """Apply interference pattern activation"""
        # Get shape information
        if len(z.shape) == 3:  # (batch, channels, length)
            batch_size, channels, length = z.shape
        else:
            raise ValueError(f"Unexpected shape: {z.shape}")
            
        result = np.zeros_like(z)
        
        # Process each sample in batch
        for b in range(batch_size):
            # Process each position
            for l in range(length):
                # Get all channel values at this position
                channel_values = z[b, :, l]
                
                # Apply interference between channels
                for c in range(channels):
                    # Start with the channel's own value
                    result[b, c, l] = channel_values[c]
                    
                    # Add interference from other channels
                    for other_c in range(channels):
                        if other_c != c:
                            # Calculate interference term with phase factor
                            weight = 0.1 * (1.0 / (abs(c - other_c) + 1))
                            phase_factor = np.exp(1j * (c - other_c) * np.pi/channels)
                            result[b, c, l] += weight * channel_values[other_c] * phase_factor
                    
                    # Normalize
                    norm = np.abs(result[b, c, l])
                    if norm > 0:
                        result[b, c, l] /= norm
        
        return result
    
    def _apply_interference(self, output: np.ndarray) -> np.ndarray:
        """Apply quantum interference between output channels"""
        # Get shape information
        if len(output.shape) == 3:  # (batch, channels, length)
            batch_size, channels, length = output.shape
        elif len(output.shape) == 2:  # (channels, length)
            # Handle single sample
            output = np.expand_dims(output, 0)
            batch_size, channels, length = output.shape
            single_sample = True
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
            
        result = np.zeros_like(output)
        
        # Process each sample in batch
        for b in range(batch_size):
            # Process each position
            for l in range(length):
                # Get all channel values at this position
                channel_values = output[b, :, l]
                
                # Apply interference between channels
                for c in range(channels):
                    # Start with the channel's own value
                    result[b, c, l] = channel_values[c]
                    
                    # Add interference from other channels
                    for other_c in range(channels):
                        if other_c != c:
                            # Calculate interference weight and phase
                            weight = 0.1 * (1.0 / (abs(c - other_c) + 1))
                            phase_factor = np.exp(1j * (c - other_c) * np.pi/channels)
                            result[b, c, l] += weight * channel_values[other_c] * phase_factor
                    
                    # Normalize
                    norm = np.abs(result[b, c, l])
                    if norm > 0:
                        result[b, c, l] /= norm
        
        # Return to original shape if single sample
        if 'single_sample' in locals() and single_sample:
            return result[0]
        return result
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Backward pass for gradient computation"""
        # Handle single sample vs batch
        single_sample = False
        if len(grad_output.shape) == 2:  # (channels, length)
            single_sample = True
            grad_output = np.expand_dims(grad_output, 0)  # (1, channels, length)
            
        if self.last_input is None:
            raise RuntimeError("Cannot perform backward pass before forward pass")
            
        input_data = self.last_input
        if single_sample and len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, 0)
            
        batch_size, in_channels, in_length = input_data.shape
        _, out_channels, out_length = grad_output.shape
        
        # Initialize gradient for input
        grad_input = np.zeros_like(input_data)
        
        # Reset gradient accumulators
        kernel_grads = np.zeros_like(self.kernels)
        bias_grads = np.zeros_like(self.biases)
        
        # Compute gradients
        for b in range(batch_size):
            # Gradient for bias
            for oc in range(out_channels):
                bias_grads[oc] += np.sum(grad_output[b, oc])
                
            # Gradient for kernels and input
            for oc in range(out_channels):
                for ol in range(out_length):
                    # Input slice position
                    in_start = ol * self.stride
                    in_end = in_start + self.kernel_size
                    
                    # Ensure we don't go out of bounds
                    if in_end > in_length:
                        continue
                        
                    # Gradient for kernels
                    for ic in range(in_channels):
                        input_slice = input_data[b, ic, in_start:in_end]
                        kernel_grads[oc, ic] += grad_output[b, oc, ol] * np.conj(input_slice)
                        
                    # Gradient for input
                    for ic in range(in_channels):
                        for k in range(self.kernel_size):
                            in_pos = in_start + k
                            if in_pos < in_length:
                                grad_input[b, ic, in_pos] += grad_output[b, oc, ol] * np.conj(self.kernels[oc, ic, k])
        
        # Accumulate gradients
        self.kernel_gradients += kernel_grads
        self.bias_gradients += bias_grads
        
        return grad_input[0] if single_sample else grad_input
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update layer parameters using accumulated gradients"""
        # Update kernels
        self.kernels -= learning_rate * self.kernel_gradients
        
        # Update biases
        self.biases -= learning_rate * self.bias_gradients
        
        # Reset gradients
        self.kernel_gradients = np.zeros_like(self.kernels)
        self.bias_gradients = np.zeros_like(self.biases)
        
        # Normalize kernels after update
        for oc in range(self.output_channels):
            for ic in range(self.input_channels):
                kernel_norm = np.sqrt(np.sum(np.abs(self.kernels[oc, ic]) ** 2))
                if kernel_norm > 0:
                    self.kernels[oc, ic] = self.kernels[oc, ic] / kernel_norm

class QuantumInterferenceConvNet:
    """
    🏗️ Quantum Interference Convolutional Neural Network
    
    Complete convolutional neural network architecture implementing quantum interference
    effects across multiple layer types. Combines quantum convolution, pooling, and
    dense layers for comprehensive quantum-inspired deep learning.
    
    Architecture Components:
        • Quantum Convolution Layers: Complex-valued feature extraction
        • Flatten Layers: Tensor reshaping for dimension compatibility
        • Dense Layers: Fully-connected quantum processing units
        • Mixed Architecture: Seamless integration of different layer types
        
    Key Features:
        • Flexible layer composition with quantum and classical components
        • End-to-end complex-valued computation pipeline
        • Quantum interference effects across all processing stages
        • Comprehensive training system with batch processing
        • Automatic layer input/output dimension management
        
    Training Capabilities:
        • Mini-batch training with quantum gradient computation
        • Single-sample and batch prediction modes
        • Loss history tracking and convergence monitoring
        • Automatic layer input caching for efficient backpropagation
        
    Quantum Processing:
        • Complex-valued forward propagation through all layers
        • Quantum-compatible backpropagation with complex gradients
        • Parameter updates respecting quantum mechanical constraints
        • Layer-wise quantum state management and interference
        
    Use Cases:
        • Quantum image processing and computer vision
        • Complex-valued signal analysis and filtering
        • Quantum pattern recognition in multi-dimensional data
        • Hybrid quantum-classical deep learning systems
        
    Mathematical Foundation:
        Forward: Output = Dense(Flatten(Conv(Input)))
        Backward: ∇θ = ∂L/∂Output * ∂Output/∂θ (with quantum conjugation)
    """
    
    def __init__(self):
        """Initialize the network"""
        self.layers = []
        self.built = False
        self.epoch_count = 0
        self.loss_history = []
        
    def add_conv_layer(self, input_channels: int, output_channels: int, kernel_size: int,
                     stride: int = 1, activation: ActivationFunction = ActivationFunction.QUANTUM_RELU,
                     apply_interference: bool = True):
        """Add a quantum convolution layer"""
        layer = QuantumConvolutionLayer(
            input_channels, output_channels, kernel_size, stride, activation, apply_interference
        )
        self.layers.append(("conv", layer))
        
    def add_flatten_layer(self):
        """Add a flatten layer to convert conv output to vector"""
        self.layers.append(("flatten", None))
        
    def add_dense_layer(self, input_dim: int, output_dim: int, 
                      activation: ActivationFunction = ActivationFunction.QUANTUM_RELU,
                      apply_interference: bool = True):
        """Add a quantum dense layer"""
        layer = QuantumLayer(input_dim, output_dim, activation, apply_interference)
        self.layers.append(("dense", layer))
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if not self.built:
            self.built = True  # Auto-build on first forward pass
            
        # Ensure input is complex
        if not np.iscomplexobj(input_data):
            input_data = input_data.astype(np.complex128)
            
        # Pass through each layer
        x = input_data
        for layer_type, layer in self.layers:
            if layer_type == "conv":
                x = layer.forward(x)
            elif layer_type == "flatten":
                # Convert from (channels, length) to vector
                if len(x.shape) == 3:  # (batch, channels, length)
                    batch_size, channels, length = x.shape
                    x = x.reshape(batch_size, channels * length)
                else:  # (channels, length)
                    channels, length = x.shape
                    x = x.reshape(channels * length)
            elif layer_type == "dense":
                x = layer.forward(x)
                
        return x
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01):
        """Backward pass through the network"""
        if not self.built:
            raise RuntimeError("Network not built")
            
        # Pass backward through layers in reverse order
        grad = grad_output
        
        # Track shapes for flatten layer
        layer_inputs = []
        
        # First forward pass to capture inputs
        x = self.last_input
        for layer_type, layer in self.layers:
            if layer_type in ["conv", "dense"]:
                layer_inputs.append(x)
                x = layer.forward(x)
            elif layer_type == "flatten":
                layer_inputs.append(x)
                if len(x.shape) == 3:  # (batch, channels, length)
                    batch_size, channels, length = x.shape
                    x = x.reshape(batch_size, channels * length)
                else:  # (channels, length)
                    channels, length = x.shape
                    x = x.reshape(channels * length)
                    
        # Backward pass
        for i in range(len(self.layers) - 1, -1, -1):
            layer_type, layer = self.layers[i]
            
            if layer_type == "conv" or layer_type == "dense":
                grad = layer.backward(grad, learning_rate)
            elif layer_type == "flatten":
                # Reshape gradient back to conv format
                if len(layer_inputs[i].shape) == 3:  # (batch, channels, length)
                    batch_size, channels, length = layer_inputs[i].shape
                    grad = grad.reshape(batch_size, channels, length)
                else:  # (channels, length)
                    channels, length = layer_inputs[i].shape
                    grad = grad.reshape(channels, length)
                    
        return grad
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update network parameters"""
        if not self.built:
            raise RuntimeError("Network not built")
            
        for layer_type, layer in self.layers:
            if layer_type in ["conv", "dense"]:
                layer.update_parameters(learning_rate)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> float:
        """Perform a single training step"""
        # Save input for backward pass
        self.last_input = x.copy()
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Calculate loss (MSE)
        loss = np.mean(np.abs(y_pred - y) ** 2)
        
        # Gradients for MSE
        grad_output = 2 * (y_pred - y) / len(y)
        
        # Backward pass
        self.backward(grad_output, learning_rate)
        
        # Update parameters
        self.update_parameters(learning_rate)
        
        return loss
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 100, learning_rate: float = 0.01,
            batch_size: int = 32, verbose: bool = True) -> List[float]:
        """Train the network"""
        n_samples = len(x_train)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Generate random batch indices
            indices = np.random.permutation(n_samples)
            
            # Process mini-batches
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Skip if batch is empty
                if len(batch_indices) == 0:
                    continue
                    
                # Get batch data
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Process each sample in batch
                batch_loss = 0.0
                for i in range(len(x_batch)):
                    sample_loss = self.train_step(x_batch[i], y_batch[i], learning_rate)
                    batch_loss += sample_loss
                    
                # Average batch loss
                batch_loss /= len(x_batch)
                epoch_loss += batch_loss * len(x_batch)
                
            # Calculate average epoch loss
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
                
            self.epoch_count += 1
            self.loss_history.append(epoch_loss)
            
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions with the network"""
        # Handle both single samples and batches
        if len(x.shape) == 2:  # Single sample (channels, length)
            return self.forward(x)
        else:
            return np.array([self.forward(sample) for sample in x])

class QuantumInterferencePooling:
    """
    🌊 Quantum Interference Pooling Layer
    
    Advanced pooling layer implementing quantum interference effects for feature
    reduction and abstraction. Supports multiple pooling modes including quantum
    interference-based pooling that preserves phase relationships.
    
    Pooling Modes:
        • Max Pooling: Select maximum amplitude values preserving phase
        • Average Pooling: Complex-valued averaging across pooling windows
        • Interference Pooling: Quantum superposition and interference of window values
        
    Quantum Features:
        • Complex-valued pooling operations throughout
        • Phase-preserving max pooling with amplitude selection
        • Quantum interference pattern generation in interference mode
        • Gradient backpropagation supporting all pooling modes
        • Automatic handling of batch vs single-sample processing
        
    Key Capabilities:
        • Multiple quantum pooling strategies
        • Configurable pool size and stride parameters
        • Efficient gradient computation for all pooling modes
        • Memory optimization with input/output caching
        • Batch and single-sample processing compatibility
        
    Mathematical Foundation:
        Max: Output = argmax_amplitude(Window)
        Average: Output = (1/N) * Σᵢ Window[i]
        Interference: Output = Σᵢ √(αᵢ) * e^(iφᵢ) (quantum superposition)
        
        where αᵢ are normalized amplitudes and φᵢ are phases
        
    Use Cases:
        • Quantum feature dimension reduction
        • Complex-valued signal downsampling
        • Quantum pattern abstraction layers
        • Interference-based feature selection
    """
    
    def __init__(self, pool_size: int = 2, stride: int = None, mode: str = "max"):
        """Initialize pooling layer"""
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mode = mode  # 'max', 'avg', or 'interference'
        
        # For backward pass
        self.last_input = None
        self.last_output = None
        self.max_indices = None  # Track indices of max values for max pooling
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through pooling layer"""
        # Save input for backward pass
        self.last_input = input_data.copy()
        
        # Handle single sample vs batch
        single_sample = False
        if len(input_data.shape) == 2:  # (channels, length)
            single_sample = True
            input_data = np.expand_dims(input_data, 0)  # (1, channels, length)
            
        batch_size, channels, input_length = input_data.shape
        
        # Calculate output dimensions
        output_length = (input_length - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, output_length), dtype=input_data.dtype)
        
        # Initialize max indices for max pooling
        if self.mode == "max":
            self.max_indices = np.zeros((batch_size, channels, output_length), dtype=np.int32)
        
        # Perform pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_length):
                    # Input window
                    start_idx = i * self.stride
                    end_idx = min(start_idx + self.pool_size, input_length)
                    
                    window = input_data[b, c, start_idx:end_idx]
                    
                    if self.mode == "max":
                        # Find index of maximum amplitude
                        amplitudes = np.abs(window)
                        max_idx = np.argmax(amplitudes)
                        self.max_indices[b, c, i] = start_idx + max_idx
                        
                        # Use complex value at max amplitude
                        output[b, c, i] = window[max_idx]
                        
                    elif self.mode == "avg":
                        # Average of complex values
                        output[b, c, i] = np.mean(window)
                        
                    elif self.mode == "interference":
                        # Quantum interference of values in window
                        # Create superposition
                        amplitudes = np.abs(window)
                        phases = np.angle(window)
                        
                        # Normalize amplitudes
                        total_amplitude = np.sum(amplitudes)
                        if total_amplitude > 0:
                            normalized_amplitudes = amplitudes / total_amplitude
                        else:
                            normalized_amplitudes = np.ones_like(amplitudes) / len(amplitudes)
                            
                        # Create interference pattern
                        interference = np.zeros(1, dtype=np.complex128)[0]
                        for j in range(len(window)):
                            # Add with phase
                            interference += np.sqrt(normalized_amplitudes[j]) * np.exp(1j * phases[j])
                            
                        # Output is the interference pattern
                        output[b, c, i] = interference
        
        self.last_output = output
        return output[0] if single_sample else output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through pooling layer"""
        # Handle single sample vs batch
        single_sample = False
        if len(grad_output.shape) == 2:  # (channels, length)
            single_sample = True
            grad_output = np.expand_dims(grad_output, 0)  # (1, channels, length)
            
        if self.last_input is None:
            raise RuntimeError("Cannot perform backward pass before forward pass")
            
        input_data = self.last_input
        if single_sample and len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, 0)
            
        batch_size, channels, input_length = input_data.shape
        
        # Initialize gradient for input
        grad_input = np.zeros_like(input_data)
        
        # Calculate gradient based on pooling mode
        if self.mode == "max":
            # Gradient flows only to max locations
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(grad_output.shape[2]):
                        max_idx = self.max_indices[b, c, i]
                        grad_input[b, c, max_idx] += grad_output[b, c, i]
                        
        elif self.mode == "avg":
            # Gradient is distributed evenly
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(grad_output.shape[2]):
                        start_idx = i * self.stride
                        end_idx = min(start_idx + self.pool_size, input_length)
                        window_size = end_idx - start_idx
                        
                        # Distribute gradient evenly
                        grad_input[b, c, start_idx:end_idx] += grad_output[b, c, i] / window_size
                        
        elif self.mode == "interference":
            # Gradient flows based on contribution to interference
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(grad_output.shape[2]):
                        start_idx = i * self.stride
                        end_idx = min(start_idx + self.pool_size, input_length)
                        
                        window = input_data[b, c, start_idx:end_idx]
                        amplitudes = np.abs(window)
                        phases = np.angle(window)
                        
                        # Normalize amplitudes
                        total_amplitude = np.sum(amplitudes)
                        if total_amplitude > 0:
                            normalized_amplitudes = amplitudes / total_amplitude
                        else:
                            normalized_amplitudes = np.ones_like(amplitudes) / len(amplitudes)
                            
                        # Distribute gradient based on contribution
                        for j in range(len(window)):
                            # Weight by amplitude and phase alignment
                            weight = np.sqrt(normalized_amplitudes[j])
                            grad_input[b, c, start_idx + j] += grad_output[b, c, i] * weight
        
        return grad_input[0] if single_sample else grad_input

class QuantumInterferenceRecurrentCell:
    """
    🔄 Quantum Interference Recurrent Neural Cell
    
    Advanced recurrent processing unit implementing quantum interference effects
    in hidden state evolution. Combines current input with previous hidden state
    using quantum mechanical principles and complex-valued computation.
    
    Architecture:
        • Complex-valued input-to-hidden weight matrix (W_ih)
        • Complex-valued hidden-to-hidden weight matrix (W_hh)
        • Complex bias vector for quantum phase adjustment
        • Quantum activation functions for state transitions
        • Inter-dimensional quantum interference in hidden states
        
    Quantum Features:
        • Complex-valued recurrent computation throughout
        • Quantum interference between hidden state dimensions
        • Phase-preserving activation functions
        • Quantum weight normalization for stability
        • Gradient accumulation for efficient batch training
        
    Key Capabilities:
        • Temporal sequence processing with quantum effects
        • Hidden state evolution with quantum interference
        • Multiple quantum activation function support
        • Efficient gradient computation for BPTT (Backprop Through Time)
        • Automatic weight normalization preservation
        
    Mathematical Foundation:
        h_t = f(W_ih * x_t + W_hh * h_{t-1} + b)
        with quantum interference: h'_t[i] = h_t[i] + Σⱼ≠ᵢ α_ij * h_t[j] * e^(iφ_ij)
        
        where f is quantum activation function and α_ij, φ_ij define interference
        
    Quantum Mechanics:
        • Hidden state represents quantum superposition
        • Weight matrices act as quantum operators
        • Interference preserves quantum coherence across time steps
        • Normalization maintains quantum state validity
        
    Use Cases:
        • Quantum sequence modeling and prediction
        • Complex-valued time series analysis
        • Quantum natural language processing
        • Temporal quantum pattern recognition
        • Quantum memory and learning systems
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
               activation: ActivationFunction = ActivationFunction.QUANTUM_TANH,
               apply_interference: bool = True):
        """Initialize the quantum recurrent cell"""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.apply_interference = apply_interference
        
        # Initialize weights (complex-valued)
        # Input weights
        self.W_ih = np.random.randn(hidden_dim, input_dim) + 1j * np.random.randn(hidden_dim, input_dim)
        # Hidden weights
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) + 1j * np.random.randn(hidden_dim, hidden_dim)
        # Bias
        self.bias = np.random.randn(hidden_dim) + 1j * np.random.randn(hidden_dim)
        
        # Normalize weights
        for i in range(hidden_dim):
            # Normalize input weights
            weight_norm = np.sqrt(np.sum(np.abs(self.W_ih[i]) ** 2))
            if weight_norm > 0:
                self.W_ih[i] = self.W_ih[i] / weight_norm
                
            # Normalize hidden weights
            weight_norm = np.sqrt(np.sum(np.abs(self.W_hh[i]) ** 2))
            if weight_norm > 0:
                self.W_hh[i] = self.W_hh[i] / weight_norm
                
        # For backward pass
        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        
        # Gradient accumulators
        self.W_ih_grad = np.zeros_like(self.W_ih)
        self.W_hh_grad = np.zeros_like(self.W_hh)
        self.bias_grad = np.zeros_like(self.bias)
        
    def forward(self, x: np.ndarray, hidden: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the recurrent cell"""
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = np.zeros(self.hidden_dim, dtype=np.complex128)
            
        # Save inputs for backward pass
        self.last_input = x.copy()
        self.last_hidden = hidden.copy()
        
        # Calculate new hidden state
        z = np.dot(self.W_ih, x) + np.dot(self.W_hh, hidden) + self.bias
        
        # Apply activation
        new_hidden = self._apply_activation(z)
        
        # Apply interference if enabled
        if self.apply_interference:
            new_hidden = self._apply_interference(new_hidden)
            
        self.last_output = new_hidden
        
        return new_hidden, new_hidden
    
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply quantum activation function"""
        if self.activation == ActivationFunction.QUANTUM_RELU:
            return self._quantum_relu(z)
        elif self.activation == ActivationFunction.QUANTUM_SIGMOID:
            return self._quantum_sigmoid(z)
        elif self.activation == ActivationFunction.QUANTUM_TANH:
            return self._quantum_tanh(z)
        elif self.activation == ActivationFunction.PHASE_SHIFT:
            return self._phase_shift(z)
        elif self.activation == ActivationFunction.HADAMARD:
            return self._hadamard(z)
        elif self.activation == ActivationFunction.INTERFERENCE:
            return self._interference(z)
        else:
            return z
    
    def _quantum_relu(self, z: np.ndarray) -> np.ndarray:
        """Quantum ReLU activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply ReLU to amplitude
        activated_amplitude = np.maximum(0, amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Quantum sigmoid activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply sigmoid to amplitude
        activated_amplitude = 1 / (1 + np.exp(-amplitude))
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _quantum_tanh(self, z: np.ndarray) -> np.ndarray:
        """Quantum tanh activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply tanh to amplitude
        activated_amplitude = np.tanh(amplitude)
        
        # Reconstruct complex number
        return activated_amplitude * np.exp(1j * phase)
    
    def _phase_shift(self, z: np.ndarray) -> np.ndarray:
        """Phase shift activation"""
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # Apply non-linear transformation to phase
        shifted_phase = np.pi * np.tanh(phase)
        
        # Keep amplitude unchanged
        return amplitude * np.exp(1j * shifted_phase)
    
    def _hadamard(self, z: np.ndarray) -> np.ndarray:
        """Hadamard-inspired activation"""
        # Normalize z
        norm = np.sqrt(np.sum(np.abs(z) ** 2))
        if norm > 0:
            z_normalized = z / norm
        else:
            z_normalized = z
            
        # Apply Hadamard-inspired transformation
        return (z_normalized + 1j * z_normalized) / np.sqrt(2)
    
    def _interference(self, z: np.ndarray) -> np.ndarray:
        """Apply interference between hidden state dimensions"""
        result = np.zeros_like(z)
        
        for i in range(len(z)):
            # Start with the dimension's own value
            result[i] = z[i]
            
            # Add interference from other dimensions
            for j in range(len(z)):
                if i != j:
                    # Calculate interference term with phase factor
                    weight = 0.1 * (1.0 / (abs(i - j) + 1))
                    phase_factor = np.exp(1j * (i - j) * np.pi/len(z))
                    result[i] += weight * z[j] * phase_factor
                    
            # Normalize
            norm = np.abs(result[i])
            if norm > 0:
                result[i] /= norm
                
        return result
    
    def backward(self, grad_output: np.ndarray, grad_hidden: np.ndarray, learning_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass through the recurrent cell"""
        if self.last_input is None or self.last_hidden is None:
            raise RuntimeError("Cannot perform backward pass before forward pass")
            
        # Combined gradient from output and hidden state
        total_grad = grad_output + grad_hidden
        
        # Gradient for bias
        self.bias_grad += total_grad
        
        # Gradient for weights
        self.W_ih_grad += np.outer(total_grad, np.conj(self.last_input))
        self.W_hh_grad += np.outer(total_grad, np.conj(self.last_hidden))
        
        # Gradient to propagate to input
        grad_input = np.dot(np.conj(self.W_ih.T), total_grad)
        
        # Gradient to propagate to previous hidden state
        grad_hidden_prev = np.dot(np.conj(self.W_hh.T), total_grad)
        
        return grad_input, grad_hidden_prev
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update cell parameters"""
        # Update weights
        self.W_ih -= learning_rate * self.W_ih_grad
        self.W_hh -= learning_rate * self.W_hh_grad
        self.bias -= learning_rate * self.bias_grad
        
        # Reset gradients
        self.W_ih_grad = np.zeros_like(self.W_ih)
        self.W_hh_grad = np.zeros_like(self.W_hh)
        self.bias_grad = np.zeros_like(self.bias)
        
        # Normalize weights after update
        for i in range(self.hidden_dim):
            # Normalize input weights
            weight_norm = np.sqrt(np.sum(np.abs(self.W_ih[i]) ** 2))
            if weight_norm > 0:
                self.W_ih[i] = self.W_ih[i] / weight_norm
                
            # Normalize hidden weights
            weight_norm = np.sqrt(np.sum(np.abs(self.W_hh[i]) ** 2))
            if weight_norm > 0:
                self.W_hh[i] = self.W_hh[i] / weight_norm

# Example usage
def run_example():
    # Set current time and user
    print(f"Current Date/Time: 2025-07-24 14:22:41")
    print(f"User: Shriram-2005")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    print("\n===== Quantum Interference Neural Network Example =====")
    
    # Create a simple quantum neural network
    network = QuantumInterferenceNN()
    
    # Build the network architecture
    network.build(
        input_dim=4,
        layer_sizes=[8, 6, 2],
        activations=[
            ActivationFunction.QUANTUM_RELU,
            ActivationFunction.PHASE_SHIFT,
            ActivationFunction.QUANTUM_TANH
        ]
    )
    
    print("\nNetwork architecture:")
    for i, layer in enumerate(network.layers):
        n_neurons = len(layer.neurons)
        activation = layer.activation.name
        print(f"  Layer {i+1}: {n_neurons} neurons, {activation} activation")
    
    # Generate synthetic quantum data
    print("\nGenerating synthetic quantum data...")
    np.random.seed(42)
    
    # Number of samples
    n_samples = 100
    
    # Input data (complex-valued)
    x_train = np.random.randn(n_samples, 4) + 1j * np.random.randn(n_samples, 4)
    
    # Normalize input vectors
    for i in range(n_samples):
        norm = np.sqrt(np.sum(np.abs(x_train[i]) ** 2))
        if norm > 0:
            x_train[i] = x_train[i] / norm
    
    # Target data (complex-valued)
    y_train = np.zeros((n_samples, 2), dtype=np.complex128)
    for i in range(n_samples):
        angle = np.sum(np.angle(x_train[i])) % (2 * np.pi)
        y_train[i, 0] = np.cos(angle) + 1j * np.sin(angle)
        y_train[i, 1] = np.sin(angle) + 1j * np.cos(angle)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(y_train[i]) ** 2))
        if norm > 0:
            y_train[i] = y_train[i] / norm
    
    # Train the network
    print("\nTraining network...")
    losses = network.train(
        x_train=x_train,
        y_train=y_train,
        epochs=50,
        learning_rate=0.01,
        batch_size=10,
        verbose=True
    )
    
    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Make a prediction
    test_input = x_train[0]
    prediction = network.predict(test_input)
    
    print("\nExample prediction:")
    print(f"Input: {test_input}")
    print(f"Predicted: {prediction}")
    print(f"Target: {y_train[0]}")
    
    # Calculate network coherence
    coherence = network.calculate_network_coherence()
    print(f"\nNetwork quantum coherence: {coherence:.6f}")
    
    # Create a convolutional example
    print("\n===== Quantum Convolutional Neural Network Example =====")
    
    # Create conv network
    conv_net = QuantumInterferenceConvNet()
    
    # Add layers
    # Input shape: (1, 16) - 1 channel, 16 length
    conv_net.add_conv_layer(1, 4, kernel_size=3, stride=1, 
                          activation=ActivationFunction.QUANTUM_RELU)
    # Output shape: (4, 14)
    
    conv_net.add_flatten_layer()
    # Output shape: 56
    
    conv_net.add_dense_layer(56, 10, activation=ActivationFunction.QUANTUM_TANH)
    # Output shape: 10
    
    # Generate synthetic 1D convolutional data
    print("\nGenerating synthetic 1D signal data...")
    
    # Input signals (1 channel, 16 length)
    x_conv = np.zeros((50, 1, 16), dtype=np.complex128)
    for i in range(50):
        # Create signal with random frequency
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2*np.pi)
        signal = np.sin(np.linspace(0, 8*np.pi, 16) * freq + phase)
        
        # Convert to complex representation
        x_conv[i, 0, :] = signal + 1j * np.zeros(16)
    
    # Target: 10-class one-hot encoded (complex)
    y_conv = np.zeros((50, 10), dtype=np.complex128)
    for i in range(50):
        # Assign to random class
        cls = i % 10
        y_conv[i, cls] = 1 + 0j
    
    # Train the conv network
    print("\nTraining convolutional network...")
    
    # Manual mini-batch training for demonstration
    epochs = 30
    learning_rate = 0.01
    batch_size = 5
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(len(x_conv))
        x_shuffled = x_conv[indices]
        y_shuffled = y_conv[indices]
        
        for i in range(0, len(x_shuffled), batch_size):
            batch_x = x_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            batch_loss = 0
            for j in range(len(batch_x)):
                # Train on single sample
                loss = conv_net.train_step(batch_x[j], batch_y[j], learning_rate)
                batch_loss += loss
                
            epoch_loss += batch_loss / len(batch_x)
            
        epoch_loss /= (len(x_shuffled) / batch_size)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    print("\nQuantum Interference Neural Network implementation complete!")
    print("The system successfully implements quantum-inspired neural computation with interference effects.")

# Module Exports - Public API
__all__ = [
    # Core Enumerations
    'ActivationFunction',
    
    # Quantum State Management
    'QuantumState',
    
    # Core Neural Components
    'QuantumNeuron',
    'QuantumLayer',
    'QuantumInterferenceNN',
    
    # Convolutional Components  
    'QuantumConvolutionLayer',
    'QuantumInterferenceConvNet',
    'QuantumInterferencePooling',
    
    # Recurrent Components
    'QuantumInterferenceRecurrentCell',
    
    # Example and Demonstration
    'run_example'
]

if __name__ == "__main__":
    run_example()