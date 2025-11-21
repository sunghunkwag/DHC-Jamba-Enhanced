"""
Mamba State Space Model Layer

Based on the Mamba architecture used in Jamba.
Implements selective state-space models with linear complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MambaLayer(nn.Module):
    """
    Mamba layer implementing selective state-space model.
    
    This is a simplified but functional implementation based on the Jamba paper.
    The Mamba layer provides linear complexity O(n) sequence processing.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        """
        Initialize Mamba layer.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for hidden dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.normal_(self.dt_proj.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Mamba layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            state: Optional initial state (batch, d_inner, d_state)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            next_state: Final hidden state (batch, d_inner, d_state)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_proj, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Convolution for local context
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim to original length
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, 2 * d_state)
        B, C = ssm_params.split(self.d_state, dim=-1)
        
        # Discretization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Selective scan (simplified version)
        y, next_state = self._selective_scan(x_conv, A, B, C, state)
        
        # Residual connection
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output, next_state
    
    # Optimize for compilation if available
    @torch.compile
    def _selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified selective scan operation.
        
        This is a simplified implementation that maintains the core functionality
        while being easier to understand and implement.
        
        Args:
            x: Input (batch, seq_len, d_inner)
            A: State transition matrix (d_inner, d_state)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)
            h_init: Optional initial state (batch, d_inner, d_state)
        
        Returns:
            Output (batch, seq_len, d_inner)
            Final state (batch, d_inner, d_state)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        if h_init is not None:
            h = h_init
        else:
            h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current input
            x_t = x[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            
            # State update: h = A * h + B * x
            h = h * torch.exp(A.unsqueeze(0))  # (batch, d_inner, d_state)
            h = h + x_t.unsqueeze(-1) * B_t  # (batch, d_inner, d_state)
            
            # Output: y = C * h + D * x
            y_t = (h * C_t).sum(dim=-1)  # (batch, d_inner)
            y_t = y_t + self.D * x_t  # Skip connection
            
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        return y, h
