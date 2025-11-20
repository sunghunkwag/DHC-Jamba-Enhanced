"""
Jamba Architecture Implementation

Hybrid Transformer-Mamba model with Mixture-of-Experts.
Based on the Jamba and Jamba-1.5 papers.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from ..layers.mamba import MambaLayer
from ..layers.attention import MultiHeadAttention, TransformerFFN
from ..layers.moe import MoELayer
from ..layers.normalization import RMSNorm


class JambaBlock(nn.Module):
    """
    Single Jamba block that interleaves Mamba and Transformer layers.
    
    The block structure follows the Jamba paper:
    - Interleaves Mamba and Transformer layers
    - Applies MoE at specified intervals
    - Uses RMSNorm for normalization
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 8,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
        layer_type: str = 'mamba',  # 'mamba' or 'attention'
    ):
        """
        Initialize Jamba block.
        
        Args:
            d_model: Model dimension
            d_state: Mamba state dimension
            d_conv: Mamba convolution size
            num_heads: Number of attention heads
            use_moe: Whether to use MoE
            num_experts: Number of experts in MoE
            num_experts_per_token: Number of experts per token
            dropout: Dropout probability
            layer_type: Type of layer ('mamba' or 'attention')
        """
        super().__init__()
        self.d_model = d_model
        self.layer_type = layer_type
        self.use_moe = use_moe
        
        # Pre-normalization
        self.norm1 = RMSNorm(d_model)
        
        # Main layer (Mamba or Attention)
        if layer_type == 'mamba':
            self.main_layer = MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
            )
        elif layer_type == 'attention':
            self.main_layer = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalid layer_type: {layer_type}")
        
        # Feed-forward or MoE
        self.norm2 = RMSNorm(d_model)
        
        if use_moe:
            self.ffn = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
            )
        else:
            self.ffn = TransformerFFN(
                d_model=d_model,
                dropout=dropout,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Jamba block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            router_logits: Router logits if using MoE, else None
        """
        router_logits = None
        
        # Main layer with residual
        residual = x
        x = self.norm1(x)
        
        if self.layer_type == 'attention':
            x = self.main_layer(x, mask=mask)
        else:
            x = self.main_layer(x)
        
        x = residual + x
        
        # FFN/MoE with residual
        residual = x
        x = self.norm2(x)
        
        if self.use_moe:
            x, router_logits = self.ffn(x)
        else:
            x = self.ffn(x)
        
        x = residual + x
        
        return x, router_logits


class JambaModel(nn.Module):
    """
    Full Jamba model with interleaved Mamba and Transformer blocks.
    
    This implementation follows the Jamba architecture pattern:
    - Ratio of Mamba to Transformer layers (e.g., 7:1)
    - MoE applied every N layers
    - RMSNorm for normalization
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 8,
        mamba_ratio: int = 7,  # 7 Mamba layers for every 1 Transformer layer
        moe_frequency: int = 2,  # Apply MoE every 2 layers
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize Jamba model.
        
        Args:
            d_model: Model dimension
            num_layers: Total number of layers
            d_state: Mamba state dimension
            d_conv: Mamba convolution size
            num_heads: Number of attention heads
            mamba_ratio: Ratio of Mamba to Transformer layers
            moe_frequency: Apply MoE every N layers
            num_experts: Number of experts in MoE
            num_experts_per_token: Number of experts per token
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Determine layer type (Mamba or Attention)
            # Every (mamba_ratio + 1)th layer is Attention, rest are Mamba
            if (i + 1) % (mamba_ratio + 1) == 0:
                layer_type = 'attention'
            else:
                layer_type = 'mamba'
            
            # Determine if MoE should be used
            use_moe = (i % moe_frequency == 0)
            
            layer = JambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                use_moe=use_moe,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                dropout=dropout,
                layer_type=layer_type,
            )
            
            self.layers.append(layer)
        
        # Final normalization
        self.norm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through Jamba model.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_router_logits: Whether to return router logits
        
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            router_logits: List of router logits if requested
        """
        router_logits_list = [] if return_router_logits else None
        
        # Pass through all layers
        for layer in self.layers:
            x, router_logits = layer(x, mask=mask)
            
            if return_router_logits and router_logits is not None:
                router_logits_list.append(router_logits)
        
        # Final normalization
        x = self.norm(x)
        
        return x, router_logits_list


class JambaConfig:
    """Configuration for Jamba model."""
    
    def __init__(
        self,
        d_model: int = 256,
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
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_heads = num_heads
        self.mamba_ratio = mamba_ratio
        self.moe_frequency = moe_frequency
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dropout = dropout
