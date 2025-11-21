"""
MuJoCo Benchmark Suite for DHC-Jamba Enhanced

Compares RL performance between DHC-SSM and DHC-Jamba on MuJoCo environments.
Uses PPO algorithm for training and evaluation.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Install with: pip install torch")

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("ERROR: Gymnasium not available. Install with: pip install gymnasium")

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("ERROR: MuJoCo not available. Install with: pip install mujoco")

if TORCH_AVAILABLE:
    from dhc_jamba.adapters.rl_policy_jamba import JambaRLPolicy, JambaRLValue


class SimpleSSMPolicy(nn.Module):
    """Baseline simple SSM policy for comparison."""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Simple architecture
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Simple state space
        self.state_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.state_update = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(obs)
        state = self.state_proj(h)
        state = self.state_update(state)
        actions = self.output_proj(state)
        return actions


class SimpleSSMValue(nn.Module):
    """Baseline simple SSM value function."""
    
    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.state_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.state_update = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(obs)
        state = self.state_proj(h)
        state = self.state_update(state)
        value = self.value_head(state)
        return value


class PPOAgent:
    """PPO agent for training and evaluation."""
    
    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        device: str = 'cpu'
    ):
        self.policy = policy.to(device)
        self.value_fn = value_fn.to(device)
        self.device = device
        
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=value_lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Action noise for exploration
        self.action_noise = 0.1
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy(obs_tensor).cpu().numpy()[0]
        
        if not deterministic:
            action += np.random.normal(0, self.action_noise, size=action.shape)
            action = np.clip(action, -1, 1)
        
        return action
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Update policy and value function."""
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action log probs
        with torch.no_grad():
            old_actions = self.policy(observations)
        
        metrics = {'policy_loss': 0, 'value_loss': 0}
        
        for _ in range(num_epochs):
            # Policy update
            new_actions = self.policy(observations)
            
            # Simple MSE loss for continuous actions
            policy_loss = F.mse_loss(new_actions, actions)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Value update
            values = self.value_fn(observations).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), 0.5)
            self.value_optimizer.step()
            
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
        
        metrics['policy_loss'] /= num_epochs
        metrics['value_loss'] /= num_epochs
        
        return metrics


def train_agent(
    env_name: str,
    agent: PPOAgent,
    num_episodes: int = 200,
    max_steps: int = 1000,
    rollout_length: int = 2048
) -> Dict:
    """Train agent on environment."""
    env = gym.make(env_name)
    
    episode_rewards = []
    episode_lengths = []
    training_metrics = []
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # Rollout buffer
    observations = []
    actions = []
    rewards = []
    values = []
    dones = []
    
    print(f"\nTraining on {env_name}...")
    
    for episode in range(num_episodes):
        for step in range(rollout_length):
            # Select action
            action = agent.select_action(obs)
            
            # Get value
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                value = agent.value_fn(obs_tensor).cpu().numpy()[0, 0]
            
            # Store transition
            observations.append(obs)
            actions.append(action)
            values.append(value)
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done or episode_length >= max_steps:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Update agent
        if len(observations) > 0:
            advantages, returns = agent.compute_gae(rewards, values, dones)
            
            metrics = agent.update(
                np.array(observations),
                np.array(actions),
                np.array(advantages),
                np.array(returns)
            )
            
            training_metrics.append(metrics)
            
            # Clear buffer
            observations = []
            actions = []
            rewards = []
            values = []
            dones = []
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_metrics': training_metrics
    }


def run_mujoco_benchmarks():
    """Run comprehensive MuJoCo benchmarks."""
    print("=" * 70)
    print("DHC-JAMBA ENHANCED - MUJOCO RL BENCHMARK SUITE")
    print("=" * 70)
    
    if not all([TORCH_AVAILABLE, GYMNASIUM_AVAILABLE, MUJOCO_AVAILABLE]):
        print("\nERROR: Missing required dependencies.")
        print("Install with: pip install torch gymnasium mujoco")
        return
    
    # Test environments
    environments = [
        "Pendulum-v1",
        "HalfCheetah-v4",
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    results = {}
    
    for env_name in environments:
        print(f"\n{'=' * 70}")
        print(f"Environment: {env_name}")
        print(f"{'=' * 70}")
        
        # Create environment to get dimensions
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
        
        # Test Simple SSM
        print("\n--- Testing Simple SSM (Baseline) ---")
        ssm_policy = SimpleSSMPolicy(obs_dim, action_dim, hidden_dim=128)
        ssm_value = SimpleSSMValue(obs_dim, hidden_dim=128)
        ssm_agent = PPOAgent(ssm_policy, ssm_value, device=device)
        
        ssm_results = train_agent(env_name, ssm_agent, num_episodes=50)
        
        # Test Jamba
        print("\n--- Testing Jamba (DHC-Jamba Enhanced) ---")
        jamba_policy = JambaRLPolicy(obs_dim, action_dim, hidden_dim=128, num_layers=4)
        jamba_value = JambaRLValue(obs_dim, hidden_dim=128, num_layers=4)
        jamba_agent = PPOAgent(jamba_policy, jamba_value, device=device)
        
        jamba_results = train_agent(env_name, jamba_agent, num_episodes=50)
        
        # Compare results
        results[env_name] = {
            'ssm': {
                'final_reward': np.mean(ssm_results['episode_rewards'][-10:]),
                'initial_reward': np.mean(ssm_results['episode_rewards'][:10]),
                'improvement': np.mean(ssm_results['episode_rewards'][-10:]) - np.mean(ssm_results['episode_rewards'][:10]),
                'all_rewards': ssm_results['episode_rewards']
            },
            'jamba': {
                'final_reward': np.mean(jamba_results['episode_rewards'][-10:]),
                'initial_reward': np.mean(jamba_results['episode_rewards'][:10]),
                'improvement': np.mean(jamba_results['episode_rewards'][-10:]) - np.mean(jamba_results['episode_rewards'][:10]),
                'all_rewards': jamba_results['episode_rewards']
            }
        }
        
        print(f"\n{env_name} Results:")
        print(f"  Simple SSM - Final: {results[env_name]['ssm']['final_reward']:.2f}, Improvement: {results[env_name]['ssm']['improvement']:.2f}")
        print(f"  Jamba - Final: {results[env_name]['jamba']['final_reward']:.2f}, Improvement: {results[env_name]['jamba']['improvement']:.2f}")
    
    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for env_name, env_results in results.items():
        results_json[env_name] = {
            'ssm': {
                'final_reward': float(env_results['ssm']['final_reward']),
                'initial_reward': float(env_results['ssm']['initial_reward']),
                'improvement': float(env_results['ssm']['improvement']),
                'all_rewards': [float(r) for r in env_results['ssm']['all_rewards']]
            },
            'jamba': {
                'final_reward': float(env_results['jamba']['final_reward']),
                'initial_reward': float(env_results['jamba']['initial_reward']),
                'improvement': float(env_results['jamba']['improvement']),
                'all_rewards': [float(r) for r in env_results['jamba']['all_rewards']]
            }
        }
    
    with open(f"benchmark_results/mujoco_benchmarks_{timestamp}.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: benchmark_results/mujoco_benchmarks_{timestamp}.json")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    run_mujoco_benchmarks()
