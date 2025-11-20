"""
Mixture of Experts (MoE) Layer

Implements sparse MoE routing as used in Jamba architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    
    Based on the MoE implementation in Jamba, which uses sparse expert selection
    to increase model capacity without proportional compute increase.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize MoE layer.
        
        Args:
            d_model: Model dimension
            num_experts: Total number of expert networks
            num_experts_per_token: Number of experts to use per token (k in top-k)
            hidden_dim: Hidden dimension for expert MLPs (default: 4 * d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.hidden_dim = hidden_dim or (4 * d_model)
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks (simple FFN)
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts)
        ])
        
        self._initialize_weights()
    
    def _create_expert(self) -> nn.Module:
        """Create a single expert network (2-layer MLP)."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model, bias=False),
        )
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.gate.weight, std=0.02)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            router_logits: Router logits for auxiliary loss
        """
        batch, seq_len, d_model = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        
        # Compute routing scores
        router_logits = self.gate(x_flat)  # (batch * seq_len, num_experts)
        
        # Top-k routing
        routing_weights, selected_experts = self._compute_routing_weights(router_logits)
        
        # Process through experts
        output = self._route_to_experts(x_flat, routing_weights, selected_experts)
        
        # Reshape back
        output = output.view(batch, seq_len, d_model)
        
        return output, router_logits.view(batch, seq_len, -1)
    
    def _compute_routing_weights(
        self,
        router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k routing weights.
        
        Args:
            router_logits: Router logits (batch * seq_len, num_experts)
        
        Returns:
            routing_weights: Normalized weights for selected experts
            selected_experts: Indices of selected experts
        """
        # Get top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits,
            self.num_experts_per_token,
            dim=-1
        )
        
        # Normalize weights with softmax
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights, selected_experts
    
    def _route_to_experts(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Route inputs to selected experts and combine outputs.
        
        Args:
            x: Input (batch * seq_len, d_model)
            routing_weights: Routing weights (batch * seq_len, num_experts_per_token)
            selected_experts: Expert indices (batch * seq_len, num_experts_per_token)
        
        Returns:
            Combined expert outputs
        """
        batch_seq_len = x.shape[0]
        output = torch.zeros_like(x)
        
        # Process each token
        for i in range(batch_seq_len):
            token_output = torch.zeros_like(x[i])
            
            # Combine outputs from selected experts
            for j in range(self.num_experts_per_token):
                expert_idx = selected_experts[i, j].item()
                weight = routing_weights[i, j]
                
                # Get expert output
                expert_output = self.experts[expert_idx](x[i].unsqueeze(0)).squeeze(0)
                
                # Weighted combination
                token_output += weight * expert_output
            
            output[i] = token_output
        
        return output


class MoEConfig:
    """Configuration for MoE layer."""
    
    def __init__(
        self,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.hidden_dim = hidden_dim
