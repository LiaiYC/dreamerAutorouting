# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 12:48:57 2025

@author: user
"""

"""
PPO Training for Multi-layer Circuit Routing Environment
"""

"""
PPO Training for Multi-layer Circuit Routing Environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import time
from datetime import datetime, timedelta
import os
import json

# 假設你的環境在這個路徑
import sys
sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


class CNNFeatureExtractor(nn.Module):
    """CNN for image feature extraction"""
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
    def forward(self, x):
        # x shape: (batch, height, width, channels)
        # Convert to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.conv(x)


class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    def __init__(self, image_shape, vector_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # Image encoder
        self.image_encoder = CNNFeatureExtractor(input_channels=image_shape[2])
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_img = torch.zeros(1, *image_shape)
            cnn_output_size = self.image_encoder(dummy_img).shape[1]
        
        # Vector encoder
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = cnn_output_size + 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, image, vector):
        # Encode image
        img_features = self.image_encoder(image)
        
        # Encode vector
        vec_features = self.vector_encoder(vector)
        
        # Fuse features
        fused = torch.cat([img_features, vec_features], dim=1)
        features = self.fusion(fused)
        
        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, image, vector, deterministic=False):
        """Sample action from policy"""
        action_logits, value = self.forward(image, vector)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, image, vector, actions):
        """Evaluate actions for PPO update"""
        action_logits, value = self.forward(image, vector)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value, entropy


class PPOMemory:
    """Rollout buffer for PPO"""
    def __init__(self):
        self.images = []
        self.vectors = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def add(self, image, vector, action, log_prob, value, reward, done):
        self.images.append(image)
        self.vectors.append(vector)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.images.clear()
        self.vectors.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        
    def get(self):
        return (
            torch.FloatTensor(np.array(self.images)),
            torch.FloatTensor(np.array(self.vectors)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones)
        )


class PPOAgent:
    """PPO Agent"""
    def __init__(self, image_shape, vector_dim, action_dim, device='cuda', 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create network
        self.policy = ActorCritic(image_shape, vector_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
    def select_action(self, obs, deterministic=False):
        """Select action from observation"""
        image = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
        vector = torch.FloatTensor(obs['vector']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action(image, vector, deterministic)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs, action, log_prob, value, reward, done):
        """Store transition in memory"""
        self.memory.add(obs['image'], obs['vector'], action, log_prob, value, reward, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # values is already a list, just append next_value
        values_extended = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self, next_obs, n_epochs=4, batch_size=64):
        """Update policy using PPO"""
        # Get data from memory
        images, vectors, actions, old_log_probs, values, rewards, dones = self.memory.get()
        
        # Move to device
        images = images.to(self.device)
        vectors = vectors.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Compute next value
        with torch.no_grad():
            next_image = torch.FloatTensor(next_obs['image']).unsqueeze(0).to(self.device)
            next_vector = torch.FloatTensor(next_obs['vector']).unsqueeze(0).to(self.device)
            _, next_value = self.policy(next_image, next_vector)
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards.tolist(), values.tolist(), 
                                               dones.tolist(), next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = len(images)
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_images = images[batch_indices]
                batch_vectors = vectors[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate_actions(
                    batch_images, batch_vectors, batch_actions
                )
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def train_ppo(
    task='circuit_routing_medium',
    total_timesteps=1000000,
    n_steps=2048,
    n_epochs=4,
    batch_size=64,
    lr=3e-4,
    save_freq=10000,
    log_freq=1000,
    eval_freq=5000,
    save_dir='./ppo_circuit_routing',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train PPO on circuit routing environment"""
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"🚀 PPO Training for Multi-layer Circuit Routing")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Device: {device}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target timesteps: {total_timesteps:,}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = CircuitRoutingEnv(task=task)
    
    # Get dimensions
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    
    print(f"Environment Info:")
    print(f"  Image shape: {image_shape}")
    print(f"  Vector dim: {vector_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Grid size: {env.grid_size}")
    print(f"  Num layers: {env.num_layers}")
    print(f"  Total pins: {env.total_pins}\n")
    
    # Create agent
    agent = PPOAgent(
        image_shape=image_shape,
        vector_dim=vector_dim,
        action_dim=action_dim,
        device=device,
        lr=lr
    )
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    timestep = 0
    episode = 0
    
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # Training loop
    while timestep < total_timesteps:
        # Collect rollout
        for _ in range(n_steps):
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, log_prob, value, reward, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            obs = next_obs
            
            # Episode finished
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode += 1
                
                # Reset
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
                
            # Logging
            if timestep % log_freq == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_length = np.mean(episode_lengths) if episode_lengths else 0
                elapsed = time.time() - start_time
                fps = timestep / elapsed
                
                print(f"Timestep: {timestep:,}/{total_timesteps:,} | "
                      f"Episode: {episode} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"FPS: {fps:.1f}")
            
            # Saving
            if timestep % save_freq == 0:
                save_path = os.path.join(save_dir, f'ppo_circuit_{timestep}.pt')
                agent.save(save_path)
            
            if timestep >= total_timesteps:
                break
        
        # Update policy
        if len(agent.memory.rewards) > 0:
            update_info = agent.update(obs, n_epochs=n_epochs, batch_size=batch_size)
            
            if timestep % log_freq == 0:
                print(f"  Policy Loss: {update_info['policy_loss']:.4f} | "
                      f"Value Loss: {update_info['value_loss']:.4f} | "
                      f"Entropy: {update_info['entropy']:.4f}")
    
    # Final save
    final_path = os.path.join(save_dir, 'ppo_circuit_final.pt')
    agent.save(final_path)
    
    # End timing
    end_time = time.time()
    end_datetime = datetime.now()
    elapsed_time = end_time - start_time
    
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save training statistics
    training_stats = {
        'task': task,
        'total_timesteps': timestep,
        'total_episodes': episode,
        'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': elapsed_time,
        'duration_formatted': f"{hours}h {minutes}m {seconds}s",
        'avg_reward_last_100': float(np.mean(episode_rewards)) if episode_rewards else 0,
        'avg_length_last_100': float(np.mean(episode_lengths)) if episode_lengths else 0,
        'fps': timestep / elapsed_time
    }
    
    stats_file = os.path.join(save_dir, 'training_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Training Completed!")
    print(f"{'='*70}")
    print(f"Start:    {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {hours}h {minutes}m {seconds}s")
    print(f"-" * 70)
    print(f"Total timesteps: {timestep:,}")
    print(f"Total episodes: {episode:,}")
    print(f"Average FPS: {timestep / elapsed_time:.2f}")
    print(f"Avg reward (last 100): {np.mean(episode_rewards):.2f}")
    print(f"Avg length (last 100): {np.mean(episode_lengths):.1f}")
    print(f"-" * 70)
    print(f"Model saved to: {final_path}")
    print(f"Stats saved to: {stats_file}")
    print(f"{'='*70}\n")
    
    return agent


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='circuit_routing_medium',
                        choices=['circuit_routing_easy', 'circuit_routing_medium', 
                                'circuit_routing_hard', 'circuit_routing_expert'])
    parser.add_argument('--timesteps', type=int, default=1000000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default='./ppo_circuit_routing')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_ppo(
        task=args.task,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device
    )