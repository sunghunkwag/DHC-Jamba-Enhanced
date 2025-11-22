"""
Jamba Architecture Implementation

Production-grade hybrid Transformer-Mamba model with Mixture-of-Experts.
Based on Jamba and Jamba-1.5 papers from AI21 Labs.

Features:
- Flexible layer interleaving (Mamba/Attention)
- Sparse Mixture-of-Experts integration
- Advanced normalization and residual connections
- Gradient checkpointing for memory efficiency
- Inference optimization with KV caching
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import math

from ..layers.mamba import MambaLayer
from ..layers.attention import MultiHeadAttention, TransformerFFN
from ..layers.moe import MoELayer
from ..layers.normalization import RMSNorm


@dataclass
class JambaConfig:
    """
    Configuration for Jamba model.
    
    Provides default values following the Jamba paper recommendations.
    """
    d_model: int = 256
    num_layers: int = 8
    d_state: int = 16
    d_conv: int = 4
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # For GQA, defaults to num_heads
    mamba_ratio: int = 7  # Mamba:Attention ratio (7:1 default)
    moe_frequency: int = 2  # Apply MoE every N layers (0 = no MoE)
    num_experts: int = 8
    num_experts_per_token: int = 2
    dropout: float = 0.1
    use_rope: bool = True
    use_swiglu: bool = True
    use_gradient_checkpointing: bool = False
    max_seq_len: int = 2048
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.mamba_ratio >= 0, "mamba_ratio must be non-negative"
        if self.num_kv_heads is not None:
            assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"


class JambaBlock(nn.Module):
    """
    Single Jamba block with flexible architecture.
    
    Supports:
    - Mamba or Transformer attention layers
    - Optional Mixture-of-Experts
    - Pre-normalization with RMSNorm
    - Residual connections
    - Gradient checkpointing
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
        layer_type: str = 'mamba',
        use_rope: bool = True,
        use_swiglu: bool = True,
        max_seq_len: int = 2048,
    ):
        """
        Initialize Jamba block.
        
        Args:
            d_model: Model dimension
            d_state: Mamba state dimension
            d_conv: Mamba convolution size
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value heads (for GQA)
            use_moe: Whether to use MoE in FFN
            num_experts: Number of experts in MoE
            num_experts_per_token: Experts per token in MoE
            dropout: Dropout probability
            layer_type: 'mamba' or 'attention'
            use_rope: Whether to use RoPE in attention
            use_swiglu: Whether to use SwiGLU activation
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.layer_type = layer_type
        self.use_moe = use_moe
        
        # Pre-normalization for main layer
        self.norm1 = RMSNorm(d_model)
        
        # Main layer (Mamba or Attention)
        if layer_type == 'mamba':
            self.main_layer = MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                use_fast_path=True,
            )
        elif layer_type == 'attention':
            self.main_layer = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
        else:
            raise ValueError(f"Invalid layer_type: {layer_type}. Must be 'mamba' or 'attention'")
        
        # Pre-normalization for FFN
        self.norm2 = RMSNorm(d_model)
        
        # Feed-forward network or MoE
        if use_moe:
            self.ffn = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                use_swiglu=use_swiglu,
            )
        else:
            self.ffn = TransformerFFN(
                d_model=d_model,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through Jamba block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional KV cache for attention layers
        
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            router_logits: Router logits if MoE is used, else None
            new_kv_cache: Updated KV cache if attention layer, else None
        """
        # Main layer with residual connection
        residual = x
        x = self.norm1(x)
        
        if self.layer_type == 'attention':
            x, new_kv_cache = self.main_layer(x, mask=mask, kv_cache=kv_cache)
        else:
            x = self.main_layer(x)
            new_kv_cache = None
        
        x = residual + x
        
        # FFN with residual connection
        residual = x
        x = self.norm2(x)
        
        if self.use_moe:
            x, router_logits = self.ffn(x)
        else:
            x = self.ffn(x)
            router_logits = None
        
        x = residual + x
        
        return x, router_logits, new_kv_cache


class JambaModel(nn.Module):
    """
    Full Jamba model with hybrid Transformer-Mamba architecture.
    
    Interleaves Mamba and Transformer layers according to specified ratio.
    Applies MoE at regular intervals for increased capacity.
    
    Architecture:
    - Flexible layer interleaving (e.g., 7 Mamba : 1 Attention)
    - Sparse MoE for parameter-efficient scaling
    - RMSNorm for stable training
    - Optional gradient checkpointing
    - Inference optimization with caching
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        mamba_ratio: int = 7,
        moe_frequency: int = 2,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_swiglu: bool = True,
        use_gradient_checkpointing: bool = False,
        max_seq_len: int = 2048,
    ):
        """
        Initialize Jamba model.
        
        Args:
            d_model: Model dimension
            num_layers: Total number of layers
            d_state: Mamba state dimension
            d_conv: Mamba convolution size
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads for GQA
            mamba_ratio: Ratio of Mamba to Attention layers
            moe_frequency: Apply MoE every N layers (0 = no MoE)
            num_experts: Number of experts in MoE
            num_experts_per_token: Experts per token
            dropout: Dropout probability
            use_rope: Use rotary position embeddings
            use_swiglu: Use SwiGLU activation
            use_gradient_checkpointing: Use gradient checkpointing
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.mamba_ratio = mamba_ratio
        self.moe_frequency = moe_frequency
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Build layers with interleaving pattern
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # Determine layer type based on ratio
            # E.g., with ratio=7: layers 0-6 are Mamba, layer 7 is Attention, repeat
            if mamba_ratio == 0:
                layer_type = 'attention'
            elif (layer_idx % (mamba_ratio + 1)) == mamba_ratio:
                layer_type = 'attention'
            else:
                layer_type = 'mamba'
            
            # Determine if this layer should use MoE
            use_moe = (moe_frequency > 0) and ((layer_idx + 1) % moe_frequency == 0)
            
            layer = JambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_moe=use_moe,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                dropout=dropout,
                layer_type=layer_type,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
                max_seq_len=max_seq_len,
            )
            
            self.layers.append(layer)
        
        # Final normalization
        self.norm_f = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through Jamba model.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_router_logits: Whether to return MoE router logits
            kv_caches: Optional list of KV caches for each layer
        
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            router_logits_list: List of router logits if return_router_logits=True
            new_kv_caches: Updated KV caches if provided
        """
        router_logits_list = [] if return_router_logits else None
        new_kv_caches = [] if kv_caches is not None else None
        
        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            # Get KV cache for this layer if available
            kv_cache = kv_caches[layer_idx] if kv_caches is not None else None
            
            # Apply gradient checkpointing if enabled
            if self.use_gradient_checkpointing and self.training:
                x, router_logits, new_kv_cache = checkpoint.checkpoint(
                    layer, x, mask, kv_cache,
                    use_reentrant=False
                )
            else:
                x, router_logits, new_kv_cache = layer(x, mask, kv_cache)
            
            # Collect router logits if requested
            if return_router_logits and router_logits is not None:
                router_logits_list.append(router_logits)
            
            # Collect new KV cache
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv_cache)
        
        # Final normalization
        x = self.norm_f(x)
        
        return x, router_logits_list, new_kv_caches
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters in the model.
        
        Args:
            non_embedding: Whether to exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model FLOPs utilization (MFU).
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
        
        Returns:
            MFU as a fraction of peak FLOPS
        """
        # Rough estimate of FLOPs per token
        N = self.get_num_params()
        L = self.num_layers
        H = self.d_model
        
        # Approximate FLOPs (simplified)
        flops_per_token = 6 * N + 12 * L * H * H
        flops_per_iter = flops_per_token * fwdbwd_per_iter
        flops_per_sec = flops_per_iter / dt
        
        # A100 peak FLOPS (bfloat16)
        flops_promised = 312e12
        
        mfu = flops_per_sec / flops_promised
        return mfu
    
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configure optimizer with weight decay for different parameter groups.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas
            device_type: Device type ('cuda' or 'cpu')
        
        Returns:
            Configured optimizer
        """
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Conv1d)):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (RMSNorm, nn.LayerNorm)):
                    no_decay.add(fpn)
                elif hasattr(p, '_no_weight_decay'):
                    no_decay.add(fpn)
                else:
                    # Default to decay
                    decay.add(fpn)
        
        # Create parameter groups
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available
        use_fused = device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused
        )
        
        return optimizer
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Input token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _, _ = self(idx)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
