# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 12:47:51 2025

@author: user
"""

"""
DreamerV3 Results Visualization Tool
Matches the format of baseline algorithms (AStar, DQN, GA, SA, PPO)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys
import json
from pathlib import Path

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


def load_dreamerv3_model(checkpoint_path, env):
    """Load DreamerV3 model (placeholder - adjust to your actual implementation)"""
    # TODO: Implement actual DreamerV3 model loading
    # This is a placeholder that you'll need to adapt to your DreamerV3 implementation
    print("⚠️  Loading DreamerV3 model...")
    print("⚠️  Please implement actual model loading for your DreamerV3")
    return None


def run_dreamerv3_episode(model, env, deterministic=True):
    """Run episode with DreamerV3 model"""
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    metrics_history = {
        'wire_length': [],
        'si_pi_performance': [],
        'drc_violations': [],
        'unrouted_nets': [],
        'rewards': []
    }
    
    while not done and step < env.max_iterations:
        # TODO: Get action from DreamerV3 model
        # For now, using random action as placeholder
        if model is None:
            action = np.random.randint(0, env.act_space['action']['discrete'])
        else:
            # Your DreamerV3 action selection here
            action = model.select_action(obs, deterministic=deterministic)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Record metrics
        metrics_history['wire_length'].append(env.wire_length)
        metrics_history['si_pi_performance'].append(env.si_pi_performance)
        metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
        metrics_history['unrouted_nets'].append(env.unrouted_nets)
        metrics_history['rewards'].append(reward)
        
        step += 1
    
    return total_reward, metrics_history, env


def create_dreamerv3_visualization(env, metrics_history, total_reward, save_dir):
    """Create visualization matching baseline format"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'DreamerV3 Circuit Routing Results\nTotal Reward: {total_reward:.2f}',
                 fontsize=18, fontweight='bold')
    
    # ===== Layer Visualizations (Top 2 rows, columns 0-2) =====
    num_layers = min(env.num_layers, 6)
    for layer_idx in range(num_layers):
        row = layer_idx // 3
        col = layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        grid = env.routing_grid[:, :, layer_idx]
        im = ax.imshow(grid.T, cmap='viridis', origin='lower', interpolation='nearest', vmin=0, vmax=1)
        
        # Mark components (red circles)
        for comp_pos in env.component_positions:
            x, y = int(comp_pos[0]), int(comp_pos[1])
            circle = Circle((x, y), radius=2, color='red', alpha=0.7, zorder=5)
            ax.add_patch(circle)
        
        # Mark pins (red stars)
        pins_on_layer = [p for p in env.pins if p.layer == layer_idx]
        for pin in pins_on_layer:
            ax.plot(pin.x, pin.y, 'r*', markersize=6, zorder=10)
        
        # Mark vias (yellow circles)
        vias_on_layer = [v for v in env.vias if v.from_layer <= layer_idx <= v.to_layer]
        for via in vias_on_layer:
            circle = Circle((via.x, via.y), radius=1, color='yellow', 
                           edgecolor='black', linewidth=1, zorder=8)
            ax.add_patch(circle)
        
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # ===== Performance Metrics (Right side) =====
    steps = list(range(len(metrics_history['wire_length'])))
    
    # Wire Length Over Time
    ax1 = fig.add_subplot(gs[0, 3])
    ax1.plot(steps, metrics_history['wire_length'], 'b-', linewidth=2)
    ax1.axhline(env.wire_length_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.wire_length_threshold}')
    ax1.set_ylabel('Wire Length', fontsize=10)
    ax1.set_title('Wire Length Over Time', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # SI/PI Performance
    ax2 = fig.add_subplot(gs[1, 3])
    ax2.plot(steps, metrics_history['si_pi_performance'], 'g-', linewidth=2)
    ax2.axhline(env.si_pi_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.si_pi_threshold}')
    ax2.set_ylabel('SI/PI Performance', fontsize=10)
    ax2.set_title('SI/PI Performance', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # DRC Violations
    ax3 = fig.add_subplot(gs[2, 3])
    ax3.plot(steps, metrics_history['drc_violations'], 'r-', linewidth=2)
    ax3.fill_between(steps, metrics_history['drc_violations'], alpha=0.3, color='red')
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Violations', fontsize=10)
    ax3.set_title('DRC Violations', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== Statistics Text Box (Bottom left) =====
    ax_stats = fig.add_subplot(gs[2, 0:3])
    ax_stats.axis('off')
    
    stats_text = f"""
📊 Final Statistics:

🔧 Circuit Configuration:
  • Grid Size: {env.grid_size[0]} × {env.grid_size[1]}
  • Layers: {env.num_layers}
  • Components: {env.num_components}
  • Total Pins: {len(env.pins)}
  • Total Nets: {len(env.nets)}

📏 Routing Metrics:
  • Wire Length: {env.wire_length:.2f} / {env.wire_length_threshold:.2f}
  • SI/PI Performance: {env.si_pi_performance:.3f} / {env.si_pi_threshold:.3f}
  • Total Vias: {env.total_vias}
  • Unrouted Nets: {env.unrouted_nets}

⚠️ DRC Violations:
  • Trace Width: {env.drc_violations['trace_width']}
  • Trace Spacing: {env.drc_violations['trace_spacing']}
  • Via Spacing: {env.drc_violations['via_spacing']}
  • Via to Trace: {env.drc_violations['via_to_trace']}
  • Clearance: {env.drc_violations['clearance']}
  • Total: {sum(env.drc_violations.values())}

✅ Success Criteria:
  • DRC Rules: {'✅ PASS' if env._check_design_rules() else '❌ FAIL'}
  • Performance: {'✅ PASS' if env._check_performance_threshold() else '❌ FAIL'}
  • Overall: {'✅ SUCCESS' if (env._check_design_rules() and env._check_performance_threshold()) else '❌ FAILED'}
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    save_path = os.path.join(save_dir, 'DreamerV3_routing_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to: {save_path}")
    
    plt.show()


def create_evaluation_plot(episode_rewards, episode_lengths, success_metrics, 
                           wire_lengths, si_pi_perfs, drc_violations, save_dir):
    """Create evaluation plot like PPO format"""
    
    n_episodes = len(episode_rewards)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DreamerV3 Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(range(n_episodes), episode_rewards, marker='o', linewidth=2, markersize=6)
    ax.axhline(np.mean(episode_rewards), color='r', linestyle='--', 
               label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    ax.plot(range(n_episodes), episode_lengths, marker='s', linewidth=2, 
            markersize=6, color='orange')
    ax.axhline(np.mean(episode_lengths), color='r', linestyle='--', 
               label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Success Metrics
    ax = axes[0, 2]
    categories = ['Overall\nSuccess', 'DRC\nPass', 'Performance\nPass']
    rates = [
        success_metrics['overall'] * 100,
        success_metrics['drc'] * 100,
        success_metrics['performance'] * 100
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Wire Length per Episode
    ax = axes[1, 0]
    ax.plot(range(n_episodes), wire_lengths, marker='o', linewidth=2, 
            markersize=6, color='green')
    ax.axhline(wire_lengths[0], color='r', linestyle='--', label='Threshold', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Wire Length', fontsize=12)
    ax.set_title('Wire Length per Episode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 5. SI/PI Performance
    ax = axes[1, 1]
    ax.plot(range(n_episodes), si_pi_perfs, marker='o', linewidth=2, 
            markersize=6, color='purple')
    ax.axhline(np.mean(si_pi_perfs), color='r', linestyle='--', 
               label=f'Mean: {np.mean(si_pi_perfs):.3f}', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('SI/PI Performance', fontsize=12)
    ax.set_title('SI/PI Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. DRC Violations
    ax = axes[1, 2]
    colors_viol = ['green' if v == 0 else 'red' for v in drc_violations]
    ax.bar(range(n_episodes), drc_violations, color=colors_viol, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('DRC Violations', fontsize=12)
    ax.set_title('DRC Violations per Episode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'DreamerV3_evaluation_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Evaluation plot saved to: {save_path}")
    
    plt.show()


def evaluate_dreamerv3(model, task, n_episodes=20, save_dir='./dreamerv3_results'):
    """Evaluate DreamerV3 model"""
    
    print("\n" + "="*70)
    print("🤖 EVALUATING DREAMERV3 MODEL")
    print("="*70)
    print(f"Task: {task}")
    print(f"Episodes: {n_episodes}")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    env = CircuitRoutingEnv(task=task)
    
    episode_rewards = []
    episode_lengths = []
    wire_lengths = []
    si_pi_perfs = []
    drc_violations = []
    
    success_count = 0
    drc_pass_count = 0
    perf_pass_count = 0
    
    for ep in range(n_episodes):
        print(f"Running episode {ep + 1}/{n_episodes}...")
        
        total_reward, metrics_history, final_env = run_dreamerv3_episode(model, env)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(len(metrics_history['wire_length']))
        wire_lengths.append(final_env.wire_length)
        si_pi_perfs.append(final_env.si_pi_performance)
        drc_violations.append(sum(final_env.drc_violations.values()))
        
        # Check success
        drc_pass = final_env._check_design_rules()
        perf_pass = final_env._check_performance_threshold()
        success = drc_pass and perf_pass
        
        if success:
            success_count += 1
        if drc_pass:
            drc_pass_count += 1
        if perf_pass:
            perf_pass_count += 1
        
        status = "✅" if success else "❌"
        print(f"  Episode {ep+1}: Reward={total_reward:.2f}, Length={len(metrics_history['wire_length'])}, Success={status}")
        
        # Create detailed visualization for first episode
        if ep == 0:
            create_dreamerv3_visualization(final_env, metrics_history, total_reward, save_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"\n🎯 Success Metrics:")
    print(f"  Overall Success:   {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"  DRC Pass Rate:     {drc_pass_count}/{n_episodes} ({drc_pass_count/n_episodes*100:.1f}%)")
    print(f"  Perf Pass Rate:    {perf_pass_count}/{n_episodes} ({perf_pass_count/n_episodes*100:.1f}%)")
    
    print(f"\n📈 Performance Statistics:")
    print(f"  Mean Reward:       {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length:       {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean Wire Length:  {np.mean(wire_lengths):.2f}")
    print(f"  Mean SI/PI:        {np.mean(si_pi_perfs):.3f}")
    print(f"  Mean DRC Viols:    {np.mean(drc_violations):.1f}")
    print("="*70 + "\n")
    
    # Create evaluation plot
    success_metrics = {
        'overall': success_count / n_episodes,
        'drc': drc_pass_count / n_episodes,
        'performance': perf_pass_count / n_episodes
    }
    
    create_evaluation_plot(episode_rewards, episode_lengths, success_metrics,
                          wire_lengths, si_pi_perfs, drc_violations, save_dir)
    
    # Save summary JSON
    summary = {
        'task': task,
        'n_episodes': n_episodes,
        'success_metrics': success_metrics,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_wire_length': float(np.mean(wire_lengths)),
        'mean_si_pi': float(np.mean(si_pi_perfs)),
        'mean_drc_violations': float(np.mean(drc_violations))
    }
    
    json_path = os.path.join(save_dir, 'DreamerV3_evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved to: {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DreamerV3 Results')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to DreamerV3 checkpoint')
    parser.add_argument('--task', type=str, default='circuit_routing_easy',
                        help='Task name')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='./dreamerv3_results',
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Load environment
    env = CircuitRoutingEnv(task=args.task)
    
    # Load model (you'll need to implement this)
    if args.checkpoint:
        model = load_dreamerv3_model(args.checkpoint, env)
    else:
        print("⚠️  No checkpoint provided. Using random actions for demonstration.")
        model = None
    
    # Evaluate
    evaluate_dreamerv3(model, args.task, args.n_episodes, args.save_dir)


if __name__ == '__main__':
    main()