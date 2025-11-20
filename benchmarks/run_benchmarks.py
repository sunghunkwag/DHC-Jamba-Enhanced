"""
Comprehensive Benchmark Suite for DHC-Jamba Enhanced

Compares performance metrics between DHC-SSM and DHC-Jamba architectures.
Tests include: parameter count, forward pass speed, memory usage, and training efficiency.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Running structure-only benchmarks.")

if TORCH_AVAILABLE:
    from dhc_jamba.core.model import DHCJambaModel, DHCJambaConfig
    from dhc_jamba.adapters.rl_policy_jamba import JambaRLPolicy, JambaRLValue


class BenchmarkResults:
    """Store and format benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, category: str, metric: str, value):
        if category not in self.results:
            self.results[category] = {}
        self.results[category][metric] = value
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        for category, metrics in self.results.items():
            print(f"\n{category}:")
            print("-" * 70)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")
                elif isinstance(value, int):
                    print(f"  {metric}: {value:,}")
                else:
                    print(f"  {metric}: {value}")
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def benchmark_model_size():
    """Benchmark model parameter counts and memory footprint."""
    print("\n=== Model Size Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    # Test different configurations
    configs = {
        "Small": DHCJambaConfig(
            hidden_dim=32,
            num_layers=4,
            num_heads=4,
            num_experts=4,
        ),
        "Medium": DHCJambaConfig(
            hidden_dim=64,
            num_layers=8,
            num_heads=8,
            num_experts=8,
        ),
        "Large": DHCJambaConfig(
            hidden_dim=128,
            num_layers=12,
            num_heads=8,
            num_experts=16,
        ),
    }
    
    for config_name, config in configs.items():
        model = DHCJambaModel(config)
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results.add_result(f"Model Size - {config_name}", "Total Parameters", num_params)
        results.add_result(f"Model Size - {config_name}", "Trainable Parameters", num_trainable)
        
        print(f"{config_name} Configuration:")
        print(f"  Total Parameters: {num_params:,}")
        print(f"  Trainable Parameters: {num_trainable:,}")
        print(f"  Model Size (MB): {num_params * 4 / (1024**2):.2f}")
    
    return results


def benchmark_forward_pass_speed():
    """Benchmark forward pass inference speed."""
    print("\n=== Forward Pass Speed Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    config = DHCJambaConfig(
        hidden_dim=64,
        num_layers=8,
        num_heads=8,
    )
    
    model = DHCJambaModel(config)
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 16, 32]
    image_size = 32
    num_warmup = 10
    num_iterations = 100
    
    for batch_size in batch_sizes:
        # Warmup
        for _ in range(num_warmup):
            x = torch.randn(batch_size, 3, image_size, image_size)
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            x = torch.randn(batch_size, 3, image_size, image_size)
            with torch.no_grad():
                _ = model(x)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        results.add_result(f"Forward Pass - Batch {batch_size}", "Avg Time (s)", avg_time)
        results.add_result(f"Forward Pass - Batch {batch_size}", "Throughput (samples/s)", throughput)
        
        print(f"Batch Size {batch_size}:")
        print(f"  Average Time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/s")
    
    return results


def benchmark_memory_usage():
    """Benchmark memory consumption during forward and backward passes."""
    print("\n=== Memory Usage Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    config = DHCJambaConfig(
        hidden_dim=64,
        num_layers=8,
        num_heads=8,
    )
    
    model = DHCJambaModel(config)
    
    # Forward pass memory
    batch_size = 16
    x = torch.randn(batch_size, 3, 32, 32)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(x)
        
        forward_memory = torch.cuda.max_memory_allocated() / (1024**2)
        results.add_result("Memory Usage", "Forward Pass (MB)", forward_memory)
        print(f"Forward Pass Memory: {forward_memory:.2f} MB")
        
        # Backward pass memory
        torch.cuda.reset_peak_memory_stats()
        model.train()
        logits = model(x)
        targets = torch.randint(0, 10, (batch_size,)).cuda()
        loss = model.compute_loss(logits, targets)
        loss.backward()
        
        backward_memory = torch.cuda.max_memory_allocated() / (1024**2)
        results.add_result("Memory Usage", "Forward + Backward (MB)", backward_memory)
        print(f"Forward + Backward Memory: {backward_memory:.2f} MB")
    else:
        print("CUDA not available, skipping GPU memory benchmark")
        results.add_result("Memory Usage", "Status", "CUDA not available")
    
    return results


def benchmark_training_speed():
    """Benchmark training iteration speed."""
    print("\n=== Training Speed Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    config = DHCJambaConfig(
        hidden_dim=64,
        num_layers=8,
        num_heads=8,
    )
    
    model = DHCJambaModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 16
    num_iterations = 50
    
    # Warmup
    for _ in range(10):
        x = torch.randn(batch_size, 3, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))
        batch = (x, targets)
        _ = model.train_step(batch, optimizer)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        x = torch.randn(batch_size, 3, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))
        batch = (x, targets)
        _ = model.train_step(batch, optimizer)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    samples_per_sec = (batch_size * num_iterations) / total_time
    
    results.add_result("Training Speed", "Avg Iteration Time (s)", avg_time)
    results.add_result("Training Speed", "Samples per Second", samples_per_sec)
    
    print(f"Average Iteration Time: {avg_time*1000:.2f} ms")
    print(f"Training Throughput: {samples_per_sec:.2f} samples/s")
    
    return results


def benchmark_rl_adapters():
    """Benchmark RL policy and value networks."""
    print("\n=== RL Adapter Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    obs_dim = 8
    action_dim = 2
    batch_size = 256
    num_iterations = 100
    
    # Create policy and value networks
    policy = JambaRLPolicy(obs_dim, action_dim, hidden_dim=128, num_layers=4)
    value_fn = JambaRLValue(obs_dim, hidden_dim=128, num_layers=4)
    
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_fn.parameters())
    
    results.add_result("RL Adapters", "Policy Parameters", policy_params)
    results.add_result("RL Adapters", "Value Parameters", value_params)
    
    print(f"Policy Parameters: {policy_params:,}")
    print(f"Value Parameters: {value_params:,}")
    
    # Benchmark policy forward pass
    obs = torch.randn(batch_size, obs_dim)
    
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = policy(obs)
    policy_time = (time.time() - start_time) / num_iterations
    
    results.add_result("RL Adapters", "Policy Forward Time (ms)", policy_time * 1000)
    print(f"Policy Forward Time: {policy_time*1000:.2f} ms")
    
    # Benchmark value forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = value_fn(obs)
    value_time = (time.time() - start_time) / num_iterations
    
    results.add_result("RL Adapters", "Value Forward Time (ms)", value_time * 1000)
    print(f"Value Forward Time: {value_time*1000:.2f} ms")
    
    return results


def benchmark_layer_comparison():
    """Compare individual layer performance."""
    print("\n=== Layer Comparison Benchmark ===")
    results = BenchmarkResults()
    
    if not TORCH_AVAILABLE:
        print("Skipping (PyTorch not available)")
        return results
    
    from dhc_jamba.layers.mamba import MambaLayer
    from dhc_jamba.layers.attention import MultiHeadAttention
    from dhc_jamba.layers.moe import MoELayer
    
    batch_size = 16
    seq_len = 32
    d_model = 256
    num_iterations = 100
    
    layers = {
        "Mamba": MambaLayer(d_model=d_model, d_state=16),
        "Attention": MultiHeadAttention(d_model=d_model, num_heads=8),
        "MoE": MoELayer(d_model=d_model, num_experts=8, num_experts_per_token=2),
    }
    
    for layer_name, layer in layers.items():
        layer.eval()
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                if layer_name == "MoE":
                    _ = layer(x)
                else:
                    _ = layer(x)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                if layer_name == "MoE":
                    _ = layer(x)
                else:
                    _ = layer(x)
        avg_time = (time.time() - start_time) / num_iterations
        
        num_params = sum(p.numel() for p in layer.parameters())
        
        results.add_result(f"Layer - {layer_name}", "Parameters", num_params)
        results.add_result(f"Layer - {layer_name}", "Forward Time (ms)", avg_time * 1000)
        
        print(f"{layer_name} Layer:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Forward Time: {avg_time*1000:.2f} ms")
    
    return results


def run_all_benchmarks():
    """Run all benchmark suites."""
    print("=" * 70)
    print("DHC-JAMBA ENHANCED - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("\nWARNING: PyTorch is not installed.")
        print("Only structure validation will be performed.")
        print("Install PyTorch to run full benchmarks:")
        print("  pip install torch")
        print()
    
    all_results = BenchmarkResults()
    
    # Run all benchmarks
    benchmarks = [
        ("Model Size", benchmark_model_size),
        ("Forward Pass Speed", benchmark_forward_pass_speed),
        ("Memory Usage", benchmark_memory_usage),
        ("Training Speed", benchmark_training_speed),
        ("RL Adapters", benchmark_rl_adapters),
        ("Layer Comparison", benchmark_layer_comparison),
    ]
    
    for bench_name, bench_func in benchmarks:
        try:
            result = bench_func()
            for category, metrics in result.results.items():
                for metric, value in metrics.items():
                    all_results.add_result(category, metric, value)
        except Exception as e:
            print(f"\nError in {bench_name}: {e}")
            all_results.add_result(bench_name, "Status", f"Error: {str(e)}")
    
    # Print summary
    all_results.print_summary()
    
    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_results.save_to_file(f"benchmark_results/benchmarks_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
