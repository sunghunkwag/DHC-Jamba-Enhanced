# DHC-Jamba-Enhanced

## Hybrid Transformer-Mamba Architecture for Spatial-Temporal Modeling

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

DHC-Jamba Enhanced combines the spatial processing capabilities of DHC-SSM with the advanced Jamba architecture, featuring hybrid Transformer-Mamba layers and Mixture-of-Experts for efficient and powerful sequence modeling.

## Overview

This project replaces the simplified State Space Model (SSM) in DHC-SSM-Enhanced with the latest Jamba architecture from AI21 Labs. The Jamba architecture interleaves Transformer attention layers with Mamba state-space layers, providing both the quality of Transformers and the efficiency of linear-complexity SSMs.

### Key Features

- **Hybrid Architecture**: Combines Transformer attention and Mamba SSM layers
- **Mixture of Experts (MoE)**: Sparse expert routing for increased model capacity
- **O(n) Complexity**: Linear computational complexity for sequence processing
- **Flexible Configuration**: Customizable layer ratios and MoE frequency
- **RL Support**: Specialized adapters for reinforcement learning tasks

## Architecture Components

### 1. Jamba Blocks

The core building block interleaves:
- **Mamba Layers**: Selective state-space models with linear complexity
- **Transformer Layers**: Multi-head attention for capturing dependencies
- **MoE Layers**: Mixture-of-Experts for parameter-efficient scaling

### 2. DHC-Jamba Model

Three-stage architecture:
1. **Spatial Encoder**: CNN-based feature extraction
2. **Jamba Model**: Hybrid Transformer-Mamba temporal processing
3. **Classification Head**: Task-specific output projection

### 3. RL Adapters

Specialized policy and value networks using Jamba architecture for reinforcement learning tasks.

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.0.0+
- CUDA (optional, for GPU acceleration)

### Install

```bash
git clone https://github.com/sunghunkwag/DHC-Jamba-Enhanced.git
cd DHC-Jamba-Enhanced
pip install -e .
```

### With RL Support

```bash
pip install -e ".[rl]"
```

## Usage

### Computer Vision

```python
from dhc_jamba import DHCJambaModel, DHCJambaConfig

config = DHCJambaConfig(
    input_channels=3,
    hidden_dim=64,
    output_dim=10,
    num_layers=8,
    mamba_ratio=7,  # 7 Mamba layers per 1 Transformer layer
    moe_frequency=2,  # MoE every 2 layers
    num_experts=8,
)

model = DHCJambaModel(config)
output = model(images)
```

### Reinforcement Learning

```python
from dhc_jamba import JambaRLPolicy, JambaRLValue
import gymnasium as gym

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = JambaRLPolicy(obs_dim, action_dim, hidden_dim=128, num_layers=4)
value_fn = JambaRLValue(obs_dim, hidden_dim=128, num_layers=4)
```

## Testing

Run the comprehensive test suite:

```bash
python tests/test_jamba_model.py
```

## Benchmarks

Comprehensive benchmark suite for performance evaluation:

### General Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

Tests model size, forward pass speed, memory usage, and training performance.

### MuJoCo RL Benchmarks

```bash
python benchmarks/run_mujoco_benchmark.py
```

Evaluates reinforcement learning performance on MuJoCo environments.

**Expected Improvements over Simple SSM**:
- Performance: +20-36% higher rewards
- Sample Efficiency: +20-29% faster convergence
- Stability: +37-57% reduced variance

See [BENCHMARKS.md](BENCHMARKS.md) and [MUJOCO_BENCHMARKS.md](MUJOCO_BENCHMARKS.md) for detailed analysis.

## Project Structure

```
DHC-Jamba-Enhanced/
├── dhc_jamba/
│   ├── core/
│   │   ├── model.py          # Main DHC-Jamba model
│   │   └── jamba.py          # Jamba architecture implementation
│   ├── adapters/
│   │   └── rl_policy_jamba.py  # RL policy/value networks
│   ├── layers/
│   │   ├── mamba.py          # Mamba state-space layer
│   │   ├── attention.py      # Multi-head attention
│   │   ├── moe.py            # Mixture-of-Experts
│   │   └── normalization.py  # RMSNorm
│   └── utils/
├── tests/
│   └── test_jamba_model.py   # Comprehensive tests
└── examples/
```

## Architecture Details

### Jamba Layer Configuration

The default configuration follows the Jamba paper:
- **Mamba-to-Attention Ratio**: 7:1 (7 Mamba layers for every 1 Transformer layer)
- **MoE Frequency**: Every 2 layers
- **Number of Experts**: 8
- **Experts per Token**: 2 (top-2 routing)

### Complexity Analysis

- **Spatial Encoder**: O(HW) for H×W images
- **Jamba Model**: O(n) for sequence length n
- **Total**: O(n) linear complexity

## Differences from DHC-SSM

| Feature | DHC-SSM | DHC-Jamba |
|---------|---------|-----------|
| Temporal Processing | Simple SSM | Hybrid Transformer-Mamba |
| Attention | None | Multi-head attention |
| MoE | None | Sparse expert routing |
| Complexity | O(n) | O(n) |
| Parameters | ~500K | ~2M (configurable) |
| Quality | Good | Better (hybrid approach) |

## References

This implementation is based on:

1. Lieber, O., et al. (2024). "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887.
2. Jamba Team, et al. (2024). "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale." arXiv:2408.12570.

## Citation

```bibtex
@software{dhc_jamba_2025,
  title={DHC-Jamba Enhanced: Hybrid Transformer-Mamba Architecture},
  author={Kwag, Sunghun},
  year={2025},
  version={1.0.0},
  url={https://github.com/sunghunkwag/DHC-Jamba-Enhanced}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original DHC-SSM architecture
- AI21 Labs for the Jamba architecture
- Mamba state-space model research
