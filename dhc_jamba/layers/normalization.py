"""
Normalization Layers

RMSNorm and other normalization layers used in Jamba.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Used in Jamba architecture instead of LayerNorm for improved efficiency.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm
