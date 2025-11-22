"""
Multi-Head Attention Layer

Advanced Transformer attention mechanism with optimizations for the Jamba architecture.

Features:
- Flash Attention compatible implementation
- Rotary Position Embeddings (RoPE) support
- Grouped Query Attention (GQA) option
- Efficient memory usage
- Numerical stability improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Provides relative position information through rotation in complex space.
    More efficient than absolute position embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Initialize rotary embeddings.
        
        Args:
            dim: Dimension of embeddings (should be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute rotation matrices for common sequence lengths
        self._precompute_freqs_cis(max_seq_len)
    
    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute complex exponentials for rotation."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary embeddings to input.
        
        Args:
            x: Input tensor (batch, seq_len, num_heads, head_dim)
            seq_len: Sequence length
        
        Returns:
            Rotated tensor
        """
        # Extend freqs_cis if needed
        if seq_len > self.freqs_cis.shape[0]:
            self._precompute_freqs_cis(seq_len)
        
        # Get frequencies for current sequence
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Reshape x to complex numbers
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
        x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
        
        return x_rotated.type_as(x)


class MultiHeadAttention(nn.Module):
    """
    Advanced multi-head self-attention mechanism.
    
    Supports:
    - Standard multi-head attention
    - Grouped Query Attention (GQA) for efficiency
    - Rotary Position Embeddings (RoPE)
    - Flash Attention compatible implementation
    - Causal and bidirectional attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        causal: bool = False,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA), defaults to num_heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            use_rope: Whether to use rotary position embeddings
            max_seq_len: Maximum sequence length for RoPE
            causal: Whether to use causal (autoregressive) attention
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.causal = causal
        self.use_rope = use_rope
        
        # Grouped Query Attention: num_kv_heads < num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_groups = num_heads // self.num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        if use_rope:
            self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        # Query, key, value projections
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        
        # Output projection - smaller initialization
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask (batch, 1, seq_len, seq_len)
            kv_cache: Optional (key, value) cache for inference
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            new_kv_cache: Updated (key, value) cache if provided
        """
        batch, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)  # (batch, seq_len, num_kv_heads * d_k)
        V = self.v_proj(x)  # (batch, seq_len, num_kv_heads * d_k)
        
        # Reshape for multi-head attention
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k)
        K = K.view(batch, seq_len, self.num_kv_heads, self.d_k)
        V = V.view(batch, seq_len, self.num_kv_heads, self.d_k)
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            Q = self.rope(Q, seq_len)
            K = self.rope(K, seq_len)
        
        # Handle KV cache for inference
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)
            new_kv_cache = (K, V)
        else:
            new_kv_cache = None
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)  # (batch, num_kv_heads, kv_seq_len, d_k)
        V = V.transpose(1, 2)  # (batch, num_kv_heads, kv_seq_len, d_k)
        
        # Grouped Query Attention: repeat K and V if needed
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention
        output = self._compute_attention(Q, K, V, mask)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, d_k)
        output = output.view(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, new_kv_cache
    
    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query (batch, num_heads, seq_len, d_k)
            K: Key (batch, num_heads, kv_seq_len, d_k)
            V: Value (batch, num_heads, kv_seq_len, d_k)
            mask: Optional mask
        
        Returns:
            Attention output (batch, num_heads, seq_len, d_k)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            seq_len = Q.size(2)
            kv_seq_len = K.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=Q.device, dtype=torch.bool),
                diagonal=kv_seq_len - seq_len + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output


class TransformerFFN(nn.Module):
    """
    Transformer Feed-Forward Network.
    
    Two-layer MLP with GELU activation and dropout.
    Uses SwiGLU variant for better performance.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        use_swiglu: bool = True,
    ):
        """
        Initialize FFN.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension, defaults to 4 * d_model
            dropout: Dropout probability
            bias: Whether to use bias
            use_swiglu: Whether to use SwiGLU activation
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU requires gating, so we need 2x hidden dim
            self.fc1 = nn.Linear(d_model, self.d_ff * 2, bias=bias)
            self.fc2 = nn.Linear(self.d_ff, d_model, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)
            self.fc2 = nn.Linear(self.d_ff, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.use_swiglu:
            # SwiGLU: x * sigmoid(x) * W
            x_proj = self.fc1(x)
            x_gate, x_linear = x_proj.chunk(2, dim=-1)
            x = F.silu(x_gate) * x_linear
        else:
            # Standard GELU
            x = self.fc1(x)
            x = F.gelu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
