"""
Comprehensive tests for DHC-Jamba model.

Tests the core Jamba architecture components and integration.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dhc_jamba.core.model import DHCJambaModel, DHCJambaConfig
from dhc_jamba.core.jamba import JambaModel, JambaConfig, JambaBlock
from dhc_jamba.layers.mamba import MambaLayer
from dhc_jamba.layers.attention import MultiHeadAttention
from dhc_jamba.layers.moe import MoELayer


def test_mamba_layer():
    """Test Mamba layer forward pass."""
    print("\n=== Testing Mamba Layer ===")
    
    batch_size = 4
    seq_len = 10
    d_model = 64
    
    layer = MambaLayer(d_model=d_model, d_state=16, d_conv=4)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print(f"✓ Mamba layer output shape: {output.shape}")
    print(f"✓ Mamba layer parameters: {sum(p.numel() for p in layer.parameters()):,}")


def test_attention_layer():
    """Test multi-head attention layer."""
    print("\n=== Testing Attention Layer ===")
    
    batch_size = 4
    seq_len = 10
    d_model = 64
    num_heads = 8
    
    layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print(f"✓ Attention layer output shape: {output.shape}")
    print(f"✓ Attention layer parameters: {sum(p.numel() for p in layer.parameters()):,}")


def test_moe_layer():
    """Test Mixture-of-Experts layer."""
    print("\n=== Testing MoE Layer ===")
    
    batch_size = 4
    seq_len = 10
    d_model = 64
    num_experts = 8
    
    layer = MoELayer(d_model=d_model, num_experts=num_experts, num_experts_per_token=2)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, router_logits = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert router_logits.shape == (batch_size, seq_len, num_experts), \
        f"Expected router logits shape {(batch_size, seq_len, num_experts)}, got {router_logits.shape}"
    
    print(f"✓ MoE layer output shape: {output.shape}")
    print(f"✓ MoE router logits shape: {router_logits.shape}")
    print(f"✓ MoE layer parameters: {sum(p.numel() for p in layer.parameters()):,}")


def test_jamba_block():
    """Test Jamba block with Mamba and Attention variants."""
    print("\n=== Testing Jamba Block ===")
    
    batch_size = 4
    seq_len = 10
    d_model = 64
    
    # Test Mamba block
    mamba_block = JambaBlock(
        d_model=d_model,
        d_state=16,
        num_heads=8,
        layer_type='mamba',
        use_moe=False
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, router_logits = mamba_block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert router_logits is None
    print(f"✓ Mamba block output shape: {output.shape}")
    
    # Test Attention block with MoE
    attn_block = JambaBlock(
        d_model=d_model,
        num_heads=8,
        layer_type='attention',
        use_moe=True,
        num_experts=8
    )
    
    output, router_logits = attn_block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert router_logits is not None
    print(f"✓ Attention+MoE block output shape: {output.shape}")
    print(f"✓ Router logits shape: {router_logits.shape}")


def test_jamba_model():
    """Test full Jamba model."""
    print("\n=== Testing Jamba Model ===")
    
    batch_size = 4
    seq_len = 10
    d_model = 64
    num_layers = 8
    
    config = JambaConfig(
        d_model=d_model,
        num_layers=num_layers,
        d_state=16,
        d_conv=4,
        num_heads=8,
        mamba_ratio=7,
        moe_frequency=2,
        num_experts=8,
        num_experts_per_token=2,
    )
    
    model = JambaModel(
        d_model=config.d_model,
        num_layers=config.num_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        num_heads=config.num_heads,
        mamba_ratio=config.mamba_ratio,
        moe_frequency=config.moe_frequency,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, router_logits_list = model(x, return_router_logits=True)
    
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Jamba model output shape: {output.shape}")
    print(f"✓ Number of layers: {num_layers}")
    print(f"✓ Number of MoE layers: {len(router_logits_list)}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")


def test_dhc_jamba_model():
    """Test full DHC-Jamba model."""
    print("\n=== Testing DHC-Jamba Model ===")
    
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    num_classes = 10
    
    config = DHCJambaConfig(
        input_channels=channels,
        hidden_dim=32,  # Smaller for testing
        output_dim=num_classes,
        num_layers=4,
        d_state=8,
        num_heads=4,
        mamba_ratio=3,
        moe_frequency=2,
        num_experts=4,
    )
    
    model = DHCJambaModel(config)
    x = torch.randn(batch_size, channels, height, width)
    
    # Test forward pass
    logits = model(x)
    assert logits.shape == (batch_size, num_classes)
    print(f"✓ DHC-Jamba output shape: {logits.shape}")
    
    # Test with features
    logits, features = model(x, return_features=True)
    assert 'spatial' in features
    assert 'temporal' in features
    assert 'logits' in features
    print(f"✓ Spatial features shape: {features['spatial'].shape}")
    print(f"✓ Temporal features shape: {features['temporal'].shape}")
    
    # Test diagnostics
    diagnostics = model.get_diagnostics()
    print(f"✓ Architecture: {diagnostics['architecture']}")
    print(f"✓ Complexity: {diagnostics['complexity']}")
    print(f"✓ Total parameters: {diagnostics['num_parameters']:,}")


def test_training_step():
    """Test training step."""
    print("\n=== Testing Training Step ===")
    
    batch_size = 8
    channels = 3
    height = 32
    width = 32
    num_classes = 10
    
    config = DHCJambaConfig(
        input_channels=channels,
        hidden_dim=32,
        output_dim=num_classes,
        num_layers=2,  # Small for testing
        num_heads=4,
        mamba_ratio=1,
    )
    
    model = DHCJambaModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy batch
    x = torch.randn(batch_size, channels, height, width)
    targets = torch.randint(0, num_classes, (batch_size,))
    batch = (x, targets)
    
    # Training step
    metrics = model.train_step(batch, optimizer)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert metrics['loss'] > 0
    assert 0 <= metrics['accuracy'] <= 1
    
    print(f"✓ Loss: {metrics['loss']:.4f}")
    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("\n=== Testing Gradient Flow ===")
    
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    num_classes = 10
    
    config = DHCJambaConfig(
        input_channels=channels,
        hidden_dim=32,
        output_dim=num_classes,
        num_layers=4,
        num_heads=4,
    )
    
    model = DHCJambaModel(config)
    x = torch.randn(batch_size, channels, height, width)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    logits = model(x)
    loss = model.compute_loss(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
    
    assert has_gradients, "No gradients found!"
    print("✓ Gradients are flowing correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("DHC-Jamba Enhanced - Comprehensive Test Suite")
    print("=" * 60)
    
    test_mamba_layer()
    test_attention_layer()
    test_moe_layer()
    test_jamba_block()
    test_jamba_model()
    test_dhc_jamba_model()
    test_training_step()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
