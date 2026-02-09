"""
Baseline Algorithms for Circuit Routing Comparison
Includes: DQN, Genetic Algorithm (GA), A*, Simulated Annealing (SA)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import time
from datetime import datetime, timedelta
import os
import json
import heapq

import sys
sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


# ======================= DQN =======================

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """DQN Network with CNN and MLP"""
    def __init__(self, image_shape, vector_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # CNN for image
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, image_shape[2], image_shape[0], image_shape[1])
            cnn_out = self.conv(dummy).shape[1]
        
        # MLP for vector
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, image, vector):
        # Image: (B, H, W, C) -> (B, C, H, W)
        img = image.permute(0, 3, 1, 2).float() / 255.0
        img_features = self.conv(img)
        vec_features = self.vector_mlp(vector)
        combined = torch.cat([img_features, vec_features], dim=1)
        return self.fusion(combined)


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent"""
    def __init__(self, image_shape, vector_dim, action_dim, device='cuda',
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, target_update=1000):
        
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_count = 0
        
        self.policy_net = DQNNetwork(image_shape, vector_dim, action_dim).to(device)
        self.target_net = DQNNetwork(image_shape, vector_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
    def select_action(self, obs, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            image = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
            vector = torch.FloatTensor(obs['vector']).unsqueeze(0).to(self.device)
            q_values = self.policy_net(image, vector)
            return q_values.argmax(1).item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, done)
    
    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare batch
        images = torch.FloatTensor(np.array([t['image'] for t in batch.state])).to(self.device)
        vectors = torch.FloatTensor(np.array([t['vector'] for t in batch.state])).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_images = torch.FloatTensor(np.array([t['image'] for t in batch.next_state])).to(self.device)
        next_vectors = torch.FloatTensor(np.array([t['vector'] for t in batch.next_state])).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q values
        q_values = self.policy_net(images, vectors).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_net(next_images, next_vectors).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}


# ======================= Genetic Algorithm =======================

class GeneticAlgorithm:
    """Genetic Algorithm for Circuit Routing"""
    def __init__(self, action_dim, pop_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.action_dim = action_dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = None
        
    def initialize_population(self, sequence_length):
        """Initialize random population"""
        self.population = [
            [random.randrange(self.action_dim) for _ in range(sequence_length)]
            for _ in range(self.pop_size)
        ]
        
    def evaluate_fitness(self, env, individual):
        """Evaluate individual fitness"""
        obs = env.reset()
        total_reward = 0
        
        for action in individual:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        return total_reward
    
    def select_parents(self, fitness_scores):
        """Tournament selection"""
        parents = []
        for _ in range(2):
            tournament = random.sample(list(zip(self.population, fitness_scores)), 5)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, individual):
        """Mutation"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randrange(self.action_dim)
        return individual
    
    def evolve(self, env, generations=10):
        """Evolve population"""
        best_fitness = float('-inf')
        best_individual = None
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(env, ind) for ind in self.population]
            
            # Track best
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = self.population[fitness_scores.index(max_fitness)]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best 2
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:2]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
        
        return best_individual, best_fitness


# ======================= A* Search =======================

class AStarAgent:
    """A* Search for Circuit Routing (Simplified Greedy Version)"""
    def __init__(self, action_dim, max_depth=20):
        self.action_dim = action_dim
        self.max_depth = max_depth
        
    def heuristic(self, vector):
        """Heuristic: negative sum of violations and unrouted nets"""
        # Based on environment definition
        unrouted = float(vector[3])  # unrouted_nets ratio
        violations = float(vector[5]) if len(vector) > 5 else 0.0  # total violations ratio
        wire_length = float(vector[0])  # wire_length ratio
        
        return -(unrouted * 100 + violations * 50 + wire_length * 10)
    
    def search(self, env, max_nodes=500):
        """Simplified greedy search (not true A*)"""
        obs = env.reset()
        actions = []
        
        for step in range(self.max_depth):
            # Try a few random actions and pick the best
            best_action = None
            best_heuristic = float('-inf')
            
            # Sample limited actions
            candidates = random.sample(range(self.action_dim), min(10, self.action_dim))
            
            for action in candidates:
                # Create temporary environment
                temp_env = CircuitRoutingEnv(env._task)
                temp_obs = temp_env.reset()
                
                # Replay all previous actions
                valid = True
                for prev_action in actions:
                    temp_obs, _, done, _ = temp_env.step(prev_action)
                    if done:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Try this action
                next_obs, reward, done, _ = temp_env.step(action)
                
                # Evaluate
                h_value = self.heuristic(next_obs['vector'])
                combined_score = reward + h_value * 0.1
                
                if combined_score > best_heuristic:
                    best_heuristic = combined_score
                    best_action = action
            
            # If no valid action found, pick random
            if best_action is None:
                best_action = random.randrange(self.action_dim)
            
            actions.append(best_action)
            
            # Actually execute in the environment
            obs, _, done, _ = env.step(best_action)
            if done:
                break
        
        return actions


# ======================= Simulated Annealing =======================

class SimulatedAnnealingAgent:
    """Simulated Annealing for Circuit Routing"""
    def __init__(self, action_dim, sequence_length=20, 
                 initial_temp=100.0, cooling_rate=0.95, min_temp=1.0):
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
    def evaluate(self, env, actions):
        """Evaluate action sequence"""
        obs = env.reset()
        total_reward = 0
        
        for action in actions:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        return total_reward
    
    def get_neighbor(self, actions):
        """Get neighbor solution by modifying one action"""
        neighbor = actions.copy()
        idx = random.randint(0, len(neighbor) - 1)
        neighbor[idx] = random.randrange(self.action_dim)
        return neighbor
    
    def optimize(self, env, max_iterations=1000):
        """Simulated annealing optimization"""
        # Initialize random solution
        current_solution = [random.randrange(self.action_dim) for _ in range(self.sequence_length)]
        current_energy = -self.evaluate(env, current_solution)  # Negative because we minimize
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        
        for iteration in range(max_iterations):
            # Get neighbor
            neighbor = self.get_neighbor(current_solution)
            neighbor_energy = -self.evaluate(env, neighbor)
            
            # Acceptance probability
            delta_e = neighbor_energy - current_energy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            # Cool down
            temperature = max(self.min_temp, temperature * self.cooling_rate)
        
        return best_solution, -best_energy


# ======================= Evaluation =======================

def evaluate_agent(agent, env, agent_type, n_episodes=10):
    """Evaluate agent performance"""
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        if agent_type == 'DQN':
            done = False
            while not done:
                action = agent.select_action(obs, eval_mode=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if info.get('is_last', False):
                    # Check success
                    if env._check_design_rules() and env._check_performance_threshold():
                        success_count += 1
                    break
                    
        elif agent_type == 'GA':
            agent.initialize_population(sequence_length=20)
            best_actions, best_reward = agent.evolve(env, generations=10)
            episode_reward = best_reward
            episode_length = len(best_actions)
            
        elif agent_type == 'A*':
            actions = agent.search(env, max_nodes=500)
            for action in actions:
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                if done:
                    break
                    
        elif agent_type == 'SA':
            actions, reward = agent.optimize(env, max_iterations=500)
            episode_reward = reward
            episode_length = len(actions)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes if agent_type == 'DQN' else 0.0
    }


def compare_algorithms(task='circuit_routing_medium', n_episodes=10):
    """Compare all algorithms"""
    print(f"\n{'='*70}")
    print(f"🔬 Baseline Algorithms Comparison")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Episodes per algorithm: {n_episodes}")
    print(f"{'='*70}\n")
    
    env = CircuitRoutingEnv(task=task)
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    # DQN (needs training first)
    print("Training DQN for 50000 steps...")
    dqn_agent = DQNAgent(image_shape, vector_dim, action_dim, device=device)
    
    obs = env.reset()
    for step in range(50000):
        action = dqn_agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        dqn_agent.store_transition(obs, action, reward, next_obs, done)
        
        if step > 1000:
            update_info = dqn_agent.update()
        
        obs = next_obs if not done else env.reset()
        
        if step % 10000 == 0:
            print(f"  Step {step}/50000")
    
    print("Evaluating DQN...")
    results['DQN'] = evaluate_agent(dqn_agent, env, 'DQN', n_episodes)
    
    # GA
    print("\nEvaluating Genetic Algorithm...")
    ga_agent = GeneticAlgorithm(action_dim)
    results['GA'] = evaluate_agent(ga_agent, env, 'GA', n_episodes)
    
    # A*
    print("Evaluating A* Search...")
    astar_agent = AStarAgent(action_dim)
    results['A*'] = evaluate_agent(astar_agent, env, 'A*', n_episodes)
    
    # SA
    print("Evaluating Simulated Annealing...")
    sa_agent = SimulatedAnnealingAgent(action_dim)
    results['SA'] = evaluate_agent(sa_agent, env, 'SA', n_episodes)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 Results Summary")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Length':<15}")
    print(f"{'-'*70}")
    
    for algo, metrics in results.items():
        print(f"{algo:<15} {metrics['mean_reward']:<15.2f} {metrics['std_reward']:<15.2f} {metrics['mean_length']:<15.1f}")
    
    print(f"{'='*70}\n")
    
    # Save results
    save_path = f'baseline_results_{task}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {save_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='circuit_routing_medium')
    parser.add_argument('--n_episodes', type=int, default=10)
    
    args = parser.parse_args()
    
    compare_algorithms(task=args.task, n_episodes=args.n_episodes)