# Benchmark Suite

This directory contains comprehensive benchmark scripts for evaluating DHC-Jamba Enhanced performance.

## Available Benchmarks

### 1. General Model Benchmarks (`run_benchmarks.py`)

Tests core model performance metrics:
- Model size and parameter counts
- Forward pass speed
- Memory usage
- Training iteration speed
- RL adapter performance
- Layer-by-layer comparison

**Usage**:
```bash
python benchmarks/run_benchmarks.py
```

**Requirements**:
- PyTorch 2.0+
- CUDA (optional, for GPU tests)

**Output**: JSON results in `benchmark_results/`

### 2. MuJoCo RL Benchmarks (`run_mujoco_benchmark.py`)

Tests reinforcement learning performance on MuJoCo environments:
- Pendulum-v1
- HalfCheetah-v4
- Hopper-v4 (optional)
- Walker2d-v4 (optional)

Compares Simple SSM baseline vs Jamba architecture using PPO algorithm.

**Usage**:
```bash
# Full benchmark suite
python benchmarks/run_mujoco_benchmark.py

# Specific environment
python benchmarks/run_mujoco_benchmark.py --env Pendulum-v1

# Custom configuration
python benchmarks/run_mujoco_benchmark.py \
    --env HalfCheetah-v4 \
    --episodes 500 \
    --hidden-dim 256
```

**Requirements**:
- PyTorch 2.0+
- Gymnasium 0.29+
- MuJoCo 3.0+

**Output**: JSON results and learning curves in `benchmark_results/`

## Installation

### Minimal (Structure tests only)
```bash
pip install -e .
```

### Full (All benchmarks)
```bash
pip install torch gymnasium mujoco
pip install -e .
```

### With GPU Support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium mujoco
pip install -e .
```

## Benchmark Results

Results are automatically saved to `benchmark_results/` with timestamps:
- `benchmarks_YYYYMMDD_HHMMSS.json` - General benchmarks
- `mujoco_benchmarks_YYYYMMDD_HHMMSS.json` - RL benchmarks

## Expected Runtime

| Benchmark | Environment | Estimated Time |
|-----------|-------------|----------------|
| General | CPU | 5-10 minutes |
| General | GPU | 2-5 minutes |
| MuJoCo (Pendulum) | CPU | 10 minutes |
| MuJoCo (HalfCheetah) | CPU | 2 hours |
| MuJoCo (Full Suite) | CPU | 7 hours |
| MuJoCo (Full Suite) | GPU | 3 hours |

## Interpreting Results

### Model Size
- **Small**: <1M parameters - Fast, suitable for simple tasks
- **Medium**: 1-5M parameters - Balanced, recommended for most tasks
- **Large**: >5M parameters - High capacity, complex tasks only

### Forward Pass Speed
- **Good**: >100 samples/second
- **Acceptable**: 50-100 samples/second
- **Slow**: <50 samples/second

### RL Performance
- **Sample Efficiency**: Fewer samples = better
- **Final Reward**: Higher = better
- **Stability**: Lower variance = more reliable
- **Convergence Speed**: Fewer episodes = faster learning

## Troubleshooting

### PyTorch Not Available
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### MuJoCo Installation Issues
```bash
# On Ubuntu/Debian
sudo apt-get install libglew-dev patchelf

# Then install MuJoCo
pip install mujoco
```

### CUDA Out of Memory
- Reduce batch size in benchmark scripts
- Use CPU instead: `--device cpu`
- Test smaller model configurations

### Slow Performance
- Enable GPU if available
- Reduce number of episodes: `--episodes 100`
- Use smaller models: `--hidden-dim 64`

## Contributing

To add new benchmarks:

1. Create new script in `benchmarks/` directory
2. Follow naming convention: `run_<benchmark_name>.py`
3. Save results to `benchmark_results/` with timestamps
4. Update this README with usage instructions

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{dhc_jamba_benchmarks_2025,
  title={DHC-Jamba Enhanced Benchmark Suite},
  author={Kwag, Sunghun},
  year={2025},
  url={https://github.com/sunghunkwag/DHC-Jamba-Enhanced}
}
```
