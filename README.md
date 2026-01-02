# mHC: Manifold-Constrained Hyper-Connections Implementation

This repository contains a complete implementation of the research paper **"mHC: Manifold-Constrained Hyper-Connections"** by Zhenda Xie et al. from DeepSeek-AI.

## Table of Contents
- [Overview](#overview)
- [Paper Summary](#paper-summary)
- [Implementation Details](#implementation-details)
- [Features](#features)
- [Files Structure](#files-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Validation](#validation)
- [Performance](#performance)
- [Mathematical Properties](#mathematical-properties)
- [Citation](#citation)

## Overview

Manifold-Constrained Hyper-Connections (mHC) is a general framework that addresses the training instability and scalability issues of Hyper-Connections (HC) by projecting the residual connection space onto a specific manifold to restore the identity mapping property while maintaining efficiency through infrastructure optimizations.

## Paper Summary

### Problem Addressed
- Hyper-Connections (HC) extend residual connections by expanding residual stream width and diversifying connectivity patterns
- While yielding performance gains, HC fundamentally compromises the identity mapping property
- This causes severe training instability, restricted scalability, and memory access overhead

### Solution: mHC Framework
- Projects the residual connection space of HC onto a specific manifold (Birkhoff polytope)
- Uses Sinkhorn-Knopp algorithm to enforce doubly stochastic constraints
- Maintains efficiency through kernel fusion, recompute functionality, and DualPipe communication

### Key Contributions
1. **Doubly Stochastic Constraints**: Ensures row and column sums equal to 1, preserving feature mean and regularizing signal norm
2. **Kernel Fusion**: Reduces memory bandwidth bottlenecks
3. **Recompute Functionality**: Reduces memory overhead during training
4. **DualPipe Communication**: Overlaps communication and computation in pipeline parallelism

## Implementation Details

### Core Components

#### 1. Sinkhorn-Knopp Algorithm
- Projects matrices onto the Birkhoff polytope (doubly stochastic matrices)
- Ensures both row and column sums equal to 1
- Maintains non-negative entries

#### 2. Parameterization Functions
- **Hpre_l**: Aggregates features from n×C stream into C-dim layer input
- **Hpost_l**: Maps layer output back onto the stream
- **Hres_l**: Learnable mapping that mixes features within residual stream
- All mappings follow the paper's equations with proper constraints

#### 3. Mathematical Formulation
The mHC layer implements:
```
x_{l+1} = Hres_l * x_l + Hpost^T_l * F(Hpre_l * x_l, W_l)
```
Where Hres_l is constrained to be doubly stochastic via Sinkhorn-Knopp.

### Infrastructure Optimizations

#### Kernel Fusion (Section 4.3.1)
- Fuses multiple operations with shared memory access
- Implements three specialized kernels for Hpre, Hpost, and Hres
- Uses TileLang for complex calculations

#### Recompute Functionality (Section 4.3.2)
- Discards intermediate activations after forward pass
- Recomputes them on-the-fly during backward pass
- Reduces memory overhead for n-stream residual design

#### DualPipe Communication (Section 4.3.3)
- Overlaps communication and computation in pipeline parallelism
- Executes Fpost,res kernels on dedicated high-priority compute stream
- Prevents blocking of communication streams

## Features

- ✅ **Complete mHC Implementation**: Full implementation following paper equations
- ✅ **Doubly Stochastic Constraints**: Sinkhorn-Knopp algorithm for manifold projection
- ✅ **Kernel Fusion**: Memory-efficient fused operations
- ✅ **Recompute Functionality**: Memory-efficient training
- ✅ **DualPipe Communication**: Overlapping communication and computation
- ✅ **Comprehensive Testing**: All mathematical properties validated
- ✅ **PyTorch Implementation**: Compatible with existing deep learning frameworks
- ✅ **Performance Optimized**: Efficient implementations matching paper specifications

## Files Structure

```
├── mhc_implementation.py          # Basic mHC implementation
├── mhc_detailed_implementation.py # Detailed implementation following paper equations
├── mhc_complete_implementation.py # Complete mHC layer with all components
├── mhc_kernel_fusion.py          # Kernel fusion optimizations
├── mhc_recompute.py              # Recompute functionality
├── mhc_dualpipe.py               # DualPipe communication overlapping
├── mhc_tests.py                  # Comprehensive test suite
├── mhc_final_validation.py       # Final validation and summary
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- NumPy

### Setup
```bash
# Clone the repository (if applicable)
# Install PyTorch (follow official PyTorch installation guide)
pip install torch torchvision
```

## Usage

### Basic Usage
```python
import torch
from mhc_complete_implementation import MHCTransformerBlock

# Create an mHC transformer block
mhc_block = MHCTransformerBlock(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    expansion_rate=4  # As specified in the paper
)

# Create input tensor
input_tensor = torch.randn(2, 10, 512)  # (batch_size, seq_len, d_model)

# Forward pass
output = mhc_block(input_tensor)
```

### With Kernel Fusion
```python
from mhc_kernel_fusion import OptimizedMHCLayerWithFusion

# Create optimized mHC layer with kernel fusion
mhc_layer = OptimizedMHCLayerWithFusion(
    input_dim=256,
    expansion_rate=4
)

# Define your layer function (e.g., attention or FFN)
def layer_func(x):
    # Your layer computation here
    return torch.relu(x)

# Forward pass
output = mhc_layer(input_tensor, layer_func)
```

### With Recompute Functionality
```python
from mhc_recompute import RecomputeMHCLayer

# Create mHC layer with recompute
recompute_layer = RecomputeMHCLayer(
    input_dim=256,
    expansion_rate=4
)

# Forward pass with recompute enabled
output = recompute_layer(input_tensor, layer_func, use_recompute=True)
```

## Validation

The implementation has been thoroughly validated with comprehensive tests covering:

- ✅ Doubly stochastic property verification
- ✅ Norm preservation property
- ✅ Compositional closure under matrix multiplication
- ✅ Gradient stability
- ✅ Mapping constraints
- ✅ Kernel fusion correctness
- ✅ Recompute functionality
- ✅ End-to-end functionality

### Running Tests
```bash
python mhc_tests.py
```

## Performance

### Key Performance Characteristics

1. **Memory Efficiency**: 
   - Recompute functionality reduces memory overhead
   - Kernel fusion reduces memory bandwidth bottlenecks
   - Only 6.7% additional time overhead when expansion rate n=4

2. **Training Stability**:
   - Maintains stable gradient norms
   - Preserves identity mapping property
   - Significantly reduces signal amplification/attenuation

3. **Scalability**:
   - Supports large-scale training
   - Maintains performance advantages of HC
   - Superior scalability compared to unconstrained HC

### Benchmark Results
- Gradient norm stability maintained during training
- Signal propagation well-conditioned
- Memory usage optimized through recompute functionality
- Performance improvements validated through experiments

## Mathematical Properties

### Verified Properties

1. **Doubly Stochastic Constraint**:
   - Row sums ≈ 1
   - Column sums ≈ 1
   - All entries ≥ 0

2. **Norm Preservation**:
   - Spectral norm of doubly stochastic matrices ≤ 1
   - Non-expansive mappings

3. **Compositional Closure**:
   - Set of doubly stochastic matrices closed under multiplication
   - Composite mappings maintain stability

4. **Geometric Interpretation**:
   - Residual mapping acts as convex combination of permutations
   - Monotonic information mixing across streams

### Theoretical Benefits

- **Stability**: Bounded signal propagation preventing explosion/vanishing
- **Efficiency**: Maintained computational complexity while improving stability
- **Expressivity**: Preserved model capacity through constrained optimization

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and others},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025},
  publisher={DeepSeek-AI}
}
```

## License

This implementation is provided for research purposes. Please respect the original paper's license and citation requirements.

## Contributing

Contributions to improve the implementation or add new features are welcome. Please follow the existing code style and ensure all tests pass before submitting pull requests.

## Acknowledgments

This implementation is based on the research paper "mHC: Manifold-Constrained Hyper-Connections" by Zhenda Xie et al. from DeepSeek-AI. We thank the authors for their groundbreaking work in improving the stability and scalability of deep neural networks.