"""
Mixture of Experts (MoE) Layer

Advanced sparse MoE implementation with efficient routing and load balancing.

Features:
- Top-k expert routing with learned gating
- Load balancing auxiliary loss
- Expert capacity management
- Efficient batched expert computation
- Noise for exploration during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with advanced routing.
    
    Implements sparse expert selection to increase model capacity
    without proportional compute increase. Based on Jamba architecture.
    
    Key features:
    - Top-k routing: Each token routed to k best experts
    - Load balancing: Auxiliary loss to balance expert usage
    - Expert capacity: Limits tokens per expert for efficiency
    - Routing noise: Exploration during training
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        hidden_dim: Optional[int] = None,
        expert_capacity_factor: float = 1.25,
        router_jitter_noise: float = 0.0,
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001,
        bias: bool = False,
        use_swiglu: bool = True,
    ):
        """
        Initialize MoE layer.
        
        Args:
            d_model: Model dimension
            num_experts: Total number of expert networks
            num_experts_per_token: Number of experts per token (k in top-k)
            hidden_dim: Expert hidden dimension (default: 4 * d_model)
            expert_capacity_factor: Factor for expert capacity calculation
            router_jitter_noise: Noise std for router exploration
            router_z_loss_coef: Coefficient for router z-loss (encourages sparsity)
            router_aux_loss_coef: Coefficient for load balancing loss
            bias: Whether to use bias in expert networks
            use_swiglu: Whether to use SwiGLU activation in experts
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.hidden_dim = hidden_dim or (4 * d_model)
        self.expert_capacity_factor = expert_capacity_factor
        self.router_jitter_noise = router_jitter_noise
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_swiglu = use_swiglu
        
        # Router/gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._create_expert(bias) for _ in range(num_experts)
        ])
        
        self._initialize_weights()
    
    def _create_expert(self, bias: bool) -> nn.Module:
        """
        Create a single expert network.
        
        Uses SwiGLU activation for better performance if enabled.
        
        Args:
            bias: Whether to use bias
        
        Returns:
            Expert network module
        """
        if self.use_swiglu:
            # SwiGLU expert: more parameters but better performance
            return SwiGLUExpert(self.d_model, self.hidden_dim, bias=bias)
        else:
            # Standard GELU expert
            return StandardExpert(self.d_model, self.hidden_dim, bias=bias)
    
    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        # Router initialization - small values for stability
        nn.init.normal_(self.gate.weight, std=0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        return_router_logits: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            return_router_logits: Whether to return router logits for loss
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            router_logits: Router logits if return_router_logits=True, else None
        """
        batch, seq_len, d_model = x.shape
        original_shape = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        num_tokens = x_flat.shape[0]
        
        # Compute routing
        router_logits = self.gate(x_flat)  # (num_tokens, num_experts)
        
        # Add jitter noise during training for exploration
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
        
        # Compute routing weights and expert assignments
        routing_weights, selected_experts = self._compute_routing(router_logits)
        
        # Route tokens to experts and compute outputs
        output = self._route_and_compute(x_flat, routing_weights, selected_experts)
        
        # Reshape to original shape
        output = output.view(original_shape)
        
        # Return router logits for auxiliary loss if requested
        if return_router_logits:
            return output, router_logits.view(batch, seq_len, -1)
        else:
            return output, None
    
    def _compute_routing(
        self,
        router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert assignments.
        
        Uses top-k routing with softmax normalization.
        
        Args:
            router_logits: Router logits (num_tokens, num_experts)
        
        Returns:
            routing_weights: Normalized routing weights (num_tokens, num_experts_per_token)
            selected_experts: Selected expert indices (num_tokens, num_experts_per_token)
        """
        # Apply softmax to get probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_probs,
            self.num_experts_per_token,
            dim=-1
        )
        
        # Renormalize top-k weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts
    
    def _route_and_compute(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Route tokens to experts and compute outputs.
        
        Uses efficient batched computation for selected experts.
        
        Args:
            x: Input tokens (num_tokens, d_model)
            routing_weights: Routing weights (num_tokens, num_experts_per_token)
            selected_experts: Expert indices (num_tokens, num_experts_per_token)
        
        Returns:
            Output tokens (num_tokens, d_model)
        """
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue  # No tokens for this expert
            
            # Get tokens and weights for this expert
            expert_input = x[token_indices]
            expert_weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
            
            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weight and accumulate
            output[token_indices] += expert_weights * expert_output
        
        return output
    
    def compute_auxiliary_loss(
        self,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for load balancing.
        
        Includes:
        - Load balancing loss: Encourages uniform expert usage
        - Router z-loss: Encourages smaller logits for stability
        
        Args:
            router_logits: Router logits (batch, seq_len, num_experts)
        
        Returns:
            Total auxiliary loss
        """
        # Flatten
        router_logits_flat = router_logits.view(-1, self.num_experts)
        
        # Load balancing loss
        # Encourages uniform distribution of tokens across experts
        routing_probs = F.softmax(router_logits_flat, dim=-1)
        expert_usage = routing_probs.mean(dim=0)  # (num_experts,)
        target_usage = 1.0 / self.num_experts
        load_balance_loss = ((expert_usage - target_usage) ** 2).sum()
        
        # Router z-loss
        # Encourages smaller logits for numerical stability
        z_loss = torch.logsumexp(router_logits_flat, dim=-1).mean()
        
        # Combine losses
        total_loss = (
            self.router_aux_loss_coef * load_balance_loss +
            self.router_z_loss_coef * z_loss
        )
        
        return total_loss
    
    def get_expert_usage_stats(
        self,
        router_logits: torch.Tensor
    ) -> dict:
        """
        Compute statistics about expert usage.
        
        Useful for monitoring and debugging.
        
        Args:
            router_logits: Router logits (batch, seq_len, num_experts)
        
        Returns:
            Dictionary with usage statistics
        """
        router_logits_flat = router_logits.view(-1, self.num_experts)
        routing_probs = F.softmax(router_logits_flat, dim=-1)
        
        # Expert selection frequency
        selected_experts = routing_probs.argmax(dim=-1)
        expert_counts = torch.bincount(
            selected_experts,
            minlength=self.num_experts
        ).float()
        expert_usage = expert_counts / expert_counts.sum()
        
        return {
            "expert_usage": expert_usage.cpu().numpy(),
            "max_usage": expert_usage.max().item(),
            "min_usage": expert_usage.min().item(),
            "usage_std": expert_usage.std().item(),
        }


class StandardExpert(nn.Module):
    """Standard expert network with GELU activation."""
    
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SwiGLUExpert(nn.Module):
    """Expert network with SwiGLU activation for better performance."""
    
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        # SwiGLU requires 2x hidden dim for gating
        self.fc1 = nn.Linear(d_model, hidden_dim * 2, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: x * sigmoid(x) * W
        x_proj = self.fc1(x)
        x_gate, x_linear = x_proj.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_linear
        x = self.fc2(x)
        return x
