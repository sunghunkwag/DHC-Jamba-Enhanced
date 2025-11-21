"""
DHC-Jamba Model

Combines spatial encoding with Jamba temporal processing for
efficient spatial-temporal modeling with O(n) complexity.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .jamba import JambaModel, JambaConfig

logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """CNN-based spatial feature extraction with adaptive pooling for small inputs."""
    
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # First conv layer (always applied)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Second conv layer (always applied)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Third conv layer (always applied)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling (works for any input size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Get input dimensions
        _, _, h, w = x.shape
        
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        # Only pool if dimensions are large enough (>= 4)
        if h >= 4 and w >= 4:
            x = nn.functional.max_pool2d(x, 2)
            h, w = h // 2, w // 2
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        # Only pool if dimensions are large enough (>= 4)
        if h >= 4 and w >= 4:
            x = nn.functional.max_pool2d(x, 2)
            h, w = h // 2, w // 2
        
        # Third conv block
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Always use adaptive pooling to get fixed output size
        x = self.adaptive_pool(x)
        
        return x.squeeze(-1).squeeze(-1)


class DHCJambaModel(nn.Module):
    """
    DHC-Jamba: Hybrid Transformer-Mamba architecture for spatial-temporal modeling.
    
    Architecture:
    1. Spatial Encoder (CNN) - O(1) per position
    2. Jamba Model (Hybrid Transformer-Mamba) - O(n) complexity
    3. Classification Head - O(1)
    
    Total complexity: O(n)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use config attributes or set defaults
        hidden_dim = getattr(config, 'hidden_dim', 64)
        input_channels = getattr(config, 'input_channels', 3)
        output_dim = getattr(config, 'output_dim', 10)
        
        # Jamba configuration
        jamba_config = getattr(config, 'jamba_config', None)
        if jamba_config is None:
            jamba_config = JambaConfig(
                d_model=hidden_dim * 4,
                num_layers=getattr(config, 'num_layers', 8),
                d_state=getattr(config, 'd_state', 16),
                d_conv=getattr(config, 'd_conv', 4),
                num_heads=getattr(config, 'num_heads', 8),
                mamba_ratio=getattr(config, 'mamba_ratio', 7),
                moe_frequency=getattr(config, 'moe_frequency', 2),
                num_experts=getattr(config, 'num_experts', 8),
                num_experts_per_token=getattr(config, 'num_experts_per_token', 2),
                dropout=getattr(config, 'dropout', 0.1),
            )
        
        self.spatial_encoder = SpatialEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim
        )
        
        self.jamba_model = JambaModel(
            d_model=jamba_config.d_model,
            num_layers=jamba_config.num_layers,
            d_state=jamba_config.d_state,
            d_conv=jamba_config.d_conv,
            num_heads=jamba_config.num_heads,
            mamba_ratio=jamba_config.mamba_ratio,
            moe_frequency=jamba_config.moe_frequency,
            num_experts=jamba_config.num_experts,
            num_experts_per_token=jamba_config.num_experts_per_token,
            dropout=jamba_config.dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        """
        Forward pass through DHC-Jamba model.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            return_features: Whether to return intermediate features
        
        Returns:
            logits: Classification logits
            features: (optional) Dictionary of intermediate features
        """
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)
        
        # Add sequence dimension for Jamba (treat as single timestep)
        jamba_input = spatial_features.unsqueeze(1)  # (batch, 1, d_model)
        
        # Jamba processing
        jamba_output, _, _ = self.jamba_model(jamba_input)
        
        # Remove sequence dimension
        temporal_features = jamba_output.squeeze(1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        if return_features:
            features = {
                'spatial': spatial_features,
                'temporal': temporal_features,
                'logits': logits
            }
            return logits, features
        
        return logits
    
    def compute_loss(self, logits, targets):
        """Standard cross-entropy loss."""
        return nn.functional.cross_entropy(logits, targets)
    
    def train_step(self, batch, optimizer):
        """Single training step."""
        x, targets = batch
        
        optimizer.zero_grad()
        logits = self(x)
        loss = self.compute_loss(logits, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def evaluate_step(self, batch):
        """Single evaluation step."""
        x, targets = batch
        
        with torch.no_grad():
            logits = self(x)
            loss = self.compute_loss(logits, targets)
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    @property
    def num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device
    
    def get_diagnostics(self):
        """Get model diagnostics."""
        return {
            'architecture': 'DHC-Jamba',
            'complexity': 'O(n)',
            'num_parameters': self.num_parameters,
            'device': str(self.device),
            'layers': {
                'spatial_encoder': {
                    'type': 'CNN',
                    'output_dim': self.config.hidden_dim * 4 if hasattr(self.config, 'hidden_dim') else 256,
                    'complexity': 'O(n)'
                },
                'jamba_model': {
                    'type': 'Hybrid Transformer-Mamba',
                    'num_layers': self.jamba_model.num_layers,
                    'complexity': 'O(n)'
                },
                'classifier': {
                    'type': 'MLP',
                    'output_dim': self.config.output_dim if hasattr(self.config, 'output_dim') else 10,
                    'complexity': 'O(1)'
                }
            }
        }


class DHCJambaConfig:
    """Configuration for DHC-Jamba model."""
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 10,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 8,
        mamba_ratio: int = 7,
        moe_frequency: int = 2,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
    ):
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_heads = num_heads
        self.mamba_ratio = mamba_ratio
        self.moe_frequency = moe_frequency
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dropout = dropout
        
        # Create Jamba config
        self.jamba_config = JambaConfig(
            d_model=hidden_dim * 4,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            mamba_ratio=mamba_ratio,
            moe_frequency=moe_frequency,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dropout=dropout,
        )
