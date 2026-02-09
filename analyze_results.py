"""
Analyze and Visualize PPO Training Results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import sys

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


def load_training_stats(stats_file='./ppo_circuit_routing/training_stats.json'):
    """Load training statistics"""
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    return stats


def print_training_summary(stats):
    """Print detailed training summary"""
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n⏱️  Time Information:")
    print(f"  Start Time:        {stats['start_time']}")
    print(f"  End Time:          {stats['end_time']}")
    print(f"  Total Duration:    {stats['duration_formatted']}")
    print(f"  Training Speed:    {stats['fps']:.2f} steps/sec")
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Total Timesteps:   {stats['total_timesteps']:,}")
    print(f"  Total Episodes:    {stats['total_episodes']:,}")
    print(f"  Avg Reward:        {stats['avg_reward_last_100']:.2f}")
    print(f"  Avg Length:        {stats['avg_length_last_100']:.1f}")
    
    print(f"\n🎮 Task Information:")
    print(f"  Task:              {stats['task']}")
    
    print("="*70 + "\n")


def evaluate_model(model_path, task, n_episodes=20, render=False):
    """Evaluate trained model"""
    import torch.nn as nn
    from ppo_train_circuit import ActorCritic, PPOAgent
    
    print(f"\n🧪 Evaluating Model on {n_episodes} Episodes")
    print("="*70)
    
    # Create environment
    env = CircuitRoutingEnv(task=task)
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    agent = PPOAgent(image_shape, vector_dim, action_dim, device=device)
    agent.load(model_path)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    drc_pass_count = 0
    performance_pass_count = 0
    
    detailed_results = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or episode_length >= env.max_iterations:
                break
        
        # Check success criteria
        drc_satisfied = env._check_design_rules()
        perf_satisfied = env._check_performance_threshold()
        success = drc_satisfied and perf_satisfied
        
        if success:
            success_count += 1
        if drc_satisfied:
            drc_pass_count += 1
        if perf_satisfied:
            performance_pass_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Store detailed results
        detailed_results.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'length': episode_length,
            'wire_length': env.wire_length,
            'si_pi_performance': env.si_pi_performance,
            'total_vias': env.total_vias,
            'unrouted_nets': env.unrouted_nets,
            'drc_violations': sum(env.drc_violations.values()),
            'success': success,
            'drc_pass': drc_satisfied,
            'perf_pass': perf_satisfied
        })
        
        print(f"Episode {ep+1:2d}: Reward={episode_reward:7.2f}, "
              f"Length={episode_length:3d}, Success={'✓' if success else '✗'}, "
              f"DRC={'✓' if drc_satisfied else '✗'}, Perf={'✓' if perf_satisfied else '✗'}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n🎯 Success Metrics:")
    print(f"  Overall Success:   {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"  DRC Pass Rate:     {drc_pass_count}/{n_episodes} ({drc_pass_count/n_episodes*100:.1f}%)")
    print(f"  Perf Pass Rate:    {performance_pass_count}/{n_episodes} ({performance_pass_count/n_episodes*100:.1f}%)")
    
    print(f"\n📈 Performance Statistics:")
    print(f"  Mean Reward:       {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Max Reward:        {np.max(episode_rewards):.2f}")
    print(f"  Min Reward:        {np.min(episode_rewards):.2f}")
    print(f"  Mean Length:       {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    print(f"\n🔧 Technical Metrics (Average):")
    avg_wire_length = np.mean([r['wire_length'] for r in detailed_results])
    avg_si_pi = np.mean([r['si_pi_performance'] for r in detailed_results])
    avg_vias = np.mean([r['total_vias'] for r in detailed_results])
    avg_unrouted = np.mean([r['unrouted_nets'] for r in detailed_results])
    avg_violations = np.mean([r['drc_violations'] for r in detailed_results])
    
    print(f"  Wire Length:       {avg_wire_length:.2f} (threshold: {env.wire_length_threshold})")
    print(f"  SI/PI Performance: {avg_si_pi:.3f} (threshold: {env.si_pi_threshold})")
    print(f"  Total Vias:        {avg_vias:.1f}")
    print(f"  Unrouted Nets:     {avg_unrouted:.1f}")
    print(f"  DRC Violations:    {avg_violations:.1f}")
    
    print("="*70 + "\n")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_count / n_episodes,
        'drc_pass_rate': drc_pass_count / n_episodes,
        'perf_pass_rate': performance_pass_count / n_episodes,
        'detailed_results': detailed_results
    }


def plot_evaluation_results(eval_results, save_dir='./ppo_circuit_routing'):
    """Plot evaluation results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO Model Evaluation Results', fontsize=16, fontweight='bold')
    
    detailed = eval_results['detailed_results']
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    rewards = eval_results['episode_rewards']
    ax.plot(rewards, marker='o', linewidth=2, markersize=6)
    ax.axhline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    lengths = eval_results['episode_lengths']
    ax.plot(lengths, marker='s', linewidth=2, markersize=6, color='orange')
    ax.axhline(np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Success Rate Breakdown
    ax = axes[0, 2]
    categories = ['Overall\nSuccess', 'DRC\nPass', 'Performance\nPass']
    rates = [
        eval_results['success_rate'] * 100,
        eval_results['drc_pass_rate'] * 100,
        eval_results['perf_pass_rate'] * 100
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Wire Length
    ax = axes[1, 0]
    wire_lengths = [r['wire_length'] for r in detailed]
    ax.plot(wire_lengths, marker='o', linewidth=2, markersize=6, color='green')
    ax.axhline(detailed[0]['wire_length'], color='r', linestyle='--', 
               label='Threshold', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Wire Length', fontsize=12)
    ax.set_title('Wire Length per Episode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 5. SI/PI Performance
    ax = axes[1, 1]
    si_pi = [r['si_pi_performance'] for r in detailed]
    ax.plot(si_pi, marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(np.mean(si_pi), color='r', linestyle='--', 
               label=f'Mean: {np.mean(si_pi):.3f}', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('SI/PI Performance', fontsize=12)
    ax.set_title('SI/PI Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. DRC Violations
    ax = axes[1, 2]
    violations = [r['drc_violations'] for r in detailed]
    colors_viol = ['green' if v == 0 else 'red' for v in violations]
    ax.bar(range(len(violations)), violations, color=colors_viol, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('DRC Violations', fontsize=12)
    ax.set_title('DRC Violations per Episode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def compare_with_random(task, n_episodes=10):
    """Compare trained model with random baseline"""
    print("\n🎲 Comparing with Random Baseline")
    print("="*70)
    
    env = CircuitRoutingEnv(task=task)
    
    random_rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = np.random.randint(0, env.act_space['action']['discrete'])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        random_rewards.append(episode_reward)
    
    print(f"Random Policy Mean Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print("="*70 + "\n")
    
    return np.mean(random_rewards)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PPO Training Results')
    parser.add_argument('--stats_file', type=str, 
                        default='./ppo_circuit_routing/training_stats.json',
                        help='Path to training stats JSON file')
    parser.add_argument('--model_path', type=str,
                        default='./ppo_circuit_routing/ppo_circuit_final.pt',
                        help='Path to trained model')
    parser.add_argument('--task', type=str, default='circuit_routing_easy',
                        help='Task name')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--compare_random', action='store_true',
                        help='Compare with random baseline')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    
    args = parser.parse_args()
    
    # Load and print training summary
    if os.path.exists(args.stats_file):
        stats = load_training_stats(args.stats_file)
        print_training_summary(stats)
        task = stats['task']
    else:
        print(f"⚠️  Stats file not found: {args.stats_file}")
        task = args.task
    
    # Evaluate model
    if os.path.exists(args.model_path):
        eval_results = evaluate_model(args.model_path, task, args.n_episodes)
        
        # Plot results
        if args.plot:
            plot_evaluation_results(eval_results)
        
        # Compare with random
        if args.compare_random:
            random_mean = compare_with_random(task, n_episodes=10)
            trained_mean = np.mean(eval_results['episode_rewards'])
            improvement = ((trained_mean - random_mean) / abs(random_mean)) * 100
            print(f"\n📈 Improvement over Random: {improvement:+.1f}%")
            print(f"   Random:  {random_mean:.2f}")
            print(f"   Trained: {trained_mean:.2f}\n")
    else:
        print(f"⚠️  Model file not found: {args.model_path}")


if __name__ == '__main__':
    main()