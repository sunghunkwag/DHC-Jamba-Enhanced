"""
Jamba-based RL Adapters

Policy and value networks using Jamba architecture for reinforcement learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from ..core.jamba import JambaModel, JambaConfig


class JambaRLPolicy(nn.Module):
    """
    Policy network using Jamba architecture for temporal modeling.
    
    Suitable for RL tasks that benefit from sequential processing.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 4,
        mamba_ratio: int = 3,  # Smaller ratio for RL
        moe_frequency: int = 0,  # Disable MoE for simpler RL tasks
        dropout: float = 0.0,  # No dropout for RL
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Jamba model for temporal processing
        jamba_config = JambaConfig(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            mamba_ratio=mamba_ratio,
            moe_frequency=moe_frequency,
            num_experts=1,  # Minimal experts for RL
            num_experts_per_token=1,
            dropout=dropout,
        )
        
        # Handle moe_frequency=0 properly
        moe_frequency_safe = jamba_config.moe_frequency if jamba_config.moe_frequency > 0 else jamba_config.num_layers + 1

        self.jamba = JambaModel(
            d_model=jamba_config.d_model,
            num_layers=jamba_config.num_layers,
            d_state=jamba_config.d_state,
            d_conv=jamba_config.d_conv,
            num_heads=jamba_config.num_heads,
            mamba_ratio=jamba_config.mamba_ratio,
            moe_frequency=moe_frequency_safe,
            num_experts=jamba_config.num_experts,
            num_experts_per_token=jamba_config.num_experts_per_token,
            dropout=jamba_config.dropout,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        Get initial zero state for Mamba layers.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            List of zero tensors for Mamba layers
        """
        states = []
        # Iterate through layers to find Mamba layers and get their dimensions
        # Note: This assumes JambaModel structure. Since we don't have easy access
        # to internal dimensions without iterating, we rely on config or model structure.
        # A robust way is to iterate through model.layers and check.

        for layer in self.jamba.layers:
            if layer.layer_type == 'mamba':
                # d_inner = expand * d_model (default expand=2)
                # We can access the actual dimensions from the layer instance
                d_inner = layer.main_layer.d_inner
                d_state = layer.main_layer.d_state

                state = torch.zeros(
                    batch_size, d_inner, d_state,
                    device=device, dtype=torch.float32
                )
                states.append(state)

        return states

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
            state: Optional list of states for Mamba layers
        
        Returns:
            actions: (batch_size, action_dim)
            next_state: List of updated Mamba states
        """
        # Project input
        h = self.input_proj(obs)
        
        # Add sequence dimension
        h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Jamba processing
        h, _, next_states = self.jamba(h, state=state)
        
        # Remove sequence dimension
        h = h.squeeze(1)  # (batch, hidden_dim)
        
        # Output projection
        actions = self.output_proj(h)
        
        return actions, next_states


class JambaRLValue(nn.Module):
    """
    Value network using Jamba architecture.
    """
    
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 4,
        mamba_ratio: int = 3,
        moe_frequency: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Jamba model
        jamba_config = JambaConfig(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            mamba_ratio=mamba_ratio,
            moe_frequency=moe_frequency,
            num_experts=1,
            num_experts_per_token=1,
            dropout=dropout,
        )
        
        # Handle moe_frequency=0 properly
        moe_frequency_safe = jamba_config.moe_frequency if jamba_config.moe_frequency > 0 else jamba_config.num_layers + 1

        self.jamba = JambaModel(
            d_model=jamba_config.d_model,
            num_layers=jamba_config.num_layers,
            d_state=jamba_config.d_state,
            d_conv=jamba_config.d_conv,
            num_heads=jamba_config.num_heads,
            mamba_ratio=jamba_config.mamba_ratio,
            moe_frequency=moe_frequency_safe,
            num_experts=jamba_config.num_experts,
            num_experts_per_token=jamba_config.num_experts_per_token,
            dropout=jamba_config.dropout,
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Get initial zero state for Mamba layers."""
        states = []
        for layer in self.jamba.layers:
            if layer.layer_type == 'mamba':
                d_inner = layer.main_layer.d_inner
                d_state = layer.main_layer.d_state
                state = torch.zeros(
                    batch_size, d_inner, d_state,
                    device=device, dtype=torch.float32
                )
                states.append(state)
        return states

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
            state: Optional list of states
        
        Returns:
            value: (batch_size, 1)
            next_state: List of updated states
        """
        # Project input
        h = self.input_proj(obs)
        
        # Add sequence dimension
        h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Jamba processing
        h, _, next_states = self.jamba(h, state=state)
        
        # Remove sequence dimension
        h = h.squeeze(1)  # (batch, hidden_dim)
        
        # Value output
        value = self.value_head(h)
        
        return value, next_states


class JambaRLActorCritic(nn.Module):
    """
    Actor-Critic with shared Jamba backbone.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        num_heads: int = 4,
        mamba_ratio: int = 3,
        moe_frequency: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared input projection
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Shared Jamba backbone
        jamba_config = JambaConfig(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            mamba_ratio=mamba_ratio,
            moe_frequency=moe_frequency,
            num_experts=1,
            num_experts_per_token=1,
            dropout=dropout,
        )
        
        # Handle moe_frequency=0 properly
        moe_frequency_safe = jamba_config.moe_frequency if jamba_config.moe_frequency > 0 else jamba_config.num_layers + 1

        self.jamba = JambaModel(
            d_model=jamba_config.d_model,
            num_layers=jamba_config.num_layers,
            d_state=jamba_config.d_state,
            d_conv=jamba_config.d_conv,
            num_heads=jamba_config.num_heads,
            mamba_ratio=jamba_config.mamba_ratio,
            moe_frequency=moe_frequency_safe,
            num_experts=jamba_config.num_experts,
            num_experts_per_token=jamba_config.num_experts_per_token,
            dropout=jamba_config.dropout,
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Get initial zero state for Mamba layers."""
        states = []
        for layer in self.jamba.layers:
            if layer.layer_type == 'mamba':
                d_inner = layer.main_layer.d_inner
                d_state = layer.main_layer.d_state
                state = torch.zeros(
                    batch_size, d_inner, d_state,
                    device=device, dtype=torch.float32
                )
                states.append(state)
        return states

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        return_value: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, observation_dim)
            state: Optional list of states
            return_value: Whether to return value
        
        Returns:
            (actions, values, next_states)
        """
        # Shared backbone
        h = self.input_proj(obs)
        h = h.unsqueeze(1)
        h, _, next_states = self.jamba(h, state=state)
        h = h.squeeze(1)
        
        # Actor
        actions = self.actor_head(h)
        
        # Critic
        if return_value:
            values = self.critic_head(h)
        else:
            values = None
        
        return actions, values, next_states
    
    def get_action(
        self,
        obs: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get action only."""
        actions, _, next_states = self.forward(obs, state=state, return_value=False)
        return actions, next_states
    
    def get_value(
        self,
        obs: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get value only."""
        _, values, next_states = self.forward(obs, state=state, return_value=True)
        return values, next_states
