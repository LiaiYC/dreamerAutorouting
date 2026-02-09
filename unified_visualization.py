
"""
Unified Circuit Routing Visualization System
Combines analysis, routing visualization, and baseline comparison
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from pathlib import Path
import sys
from datetime import datetime

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


# ==================== Core Visualization Functions ====================

def load_agent(model_path, env, agent_type='PPO'):
    """Load trained agent"""
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if agent_type == 'PPO':
        from ppo_train_circuit import PPOAgent
        agent = PPOAgent(image_shape, vector_dim, action_dim, device=device)
        agent.load(model_path)
    elif agent_type == 'DQN':
        from baseline_algorithms import DQNAgent
        agent = DQNAgent(image_shape, vector_dim, action_dim, device=device)
        checkpoint = torch.load(model_path)
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent


def run_episode(agent, env, agent_type='PPO'):
    """Run single episode and collect metrics"""
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    metrics_history = {
        'wire_length': [],
        'si_pi_performance': [],
        'drc_violations': [],
        'unrouted_nets': [],
        'total_vias': [],
        'rewards': [],
        'steps': []
    }
    
    while not done and step < env.max_iterations:
        if agent_type == 'PPO':
            action, _, _ = agent.select_action(obs, deterministic=True)
        elif agent_type == 'DQN':
            action = agent.select_action(obs, eval_mode=True)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Record metrics
        metrics_history['wire_length'].append(env.wire_length)
        metrics_history['si_pi_performance'].append(env.si_pi_performance)
        metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
        metrics_history['unrouted_nets'].append(env.unrouted_nets)
        metrics_history['total_vias'].append(env.total_vias)
        metrics_history['rewards'].append(reward)
        metrics_history['steps'].append(step)
        
        step += 1
    
    return total_reward, metrics_history, env


def print_detailed_statistics(env, agent_name, total_reward):
    """Print comprehensive statistics"""
    print("\n" + "="*70)
    print(f"📊 {agent_name} - DETAILED STATISTICS")
    print("="*70)
    
    print(f"\n🎮 Episode Performance:")
    print(f"  Total Reward:      {total_reward:.2f}")
    print(f"  Total Steps:       {env.iterations}")
    
    print(f"\n🔧 Circuit Configuration:")
    print(f"  Grid Size:         {env.grid_size[0]} × {env.grid_size[1]}")
    print(f"  Number of Layers:  {env.num_layers}")
    print(f"  Components:        {env.num_components}")
    print(f"  Total Pins:        {len(env.pins)}")
    print(f"  Total Nets:        {len(env.nets)}")
    
    print(f"\n📏 Routing Metrics:")
    print(f"  Wire Length:       {env.wire_length:.2f} / {env.wire_length_threshold:.2f}")
    wire_status = "✅" if env.wire_length <= env.wire_length_threshold else "❌"
    print(f"                     {wire_status} {'PASS' if wire_status == '✅' else 'FAIL'}")
    
    print(f"  SI/PI Performance: {env.si_pi_performance:.3f} / {env.si_pi_threshold:.3f}")
    sipi_status = "✅" if env.si_pi_performance >= env.si_pi_threshold else "❌"
    print(f"                     {sipi_status} {'PASS' if sipi_status == '✅' else 'FAIL'}")
    
    print(f"  Total Vias:        {env.total_vias}")
    print(f"  Unrouted Nets:     {env.unrouted_nets}")
    
    print(f"\n⚠️  DRC Violations:")
    for violation_type, count in env.drc_violations.items():
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {violation_type:20s}: {count}")
    
    total_violations = sum(env.drc_violations.values())
    print(f"  {'─'*40}")
    print(f"  Total Violations:    {total_violations}")
    
    print(f"\n📍 Pin Distribution by Layer:")
    for layer in range(env.num_layers):
        pins_on_layer = [p for p in env.pins if p.layer == layer]
        print(f"  Layer {layer + 1}: {len(pins_on_layer):3d} pins")
    
    print(f"\n✅ Overall Success:")
    drc_pass = env._check_design_rules()
    perf_pass = env._check_performance_threshold()
    overall = drc_pass and perf_pass
    
    print(f"  DRC Rules:         {'✅ PASS' if drc_pass else '❌ FAIL'}")
    print(f"  Performance:       {'✅ PASS' if perf_pass else '❌ FAIL'}")
    print(f"  Overall:           {'✅ SUCCESS' if overall else '❌ FAILED'}")
    
    print("="*70 + "\n")
    
    return overall


def visualize_layer(env, layer_idx, ax, title=None):
    """Visualize single PCB layer"""
    grid = env.routing_grid[:, :, layer_idx]
    
    im = ax.imshow(grid.T, cmap='viridis', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    
    # Mark components
    for comp_pos in env.component_positions:
        x, y = int(comp_pos[0]), int(comp_pos[1])
        circle = Circle((x, y), radius=2, color='red', alpha=0.7, zorder=5)
        ax.add_patch(circle)
    
    # Mark pins on this layer
    pins_on_layer = [p for p in env.pins if p.layer == layer_idx]
    for pin in pins_on_layer:
        ax.plot(pin.x, pin.y, 'r*', markersize=8, zorder=10)
    
    # Mark vias
    vias_on_layer = [v for v in env.vias if v.from_layer <= layer_idx <= v.to_layer]
    for via in vias_on_layer:
        circle = Circle((via.x, via.y), radius=1, color='yellow', 
                       edgecolor='black', linewidth=1, zorder=8)
        ax.add_patch(circle)
    
    ax.set_xlim(-0.5, env.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, env.grid_size[1] - 0.5)
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    
    return im


def create_comprehensive_visualization(env, metrics_history, agent_name, total_reward, save_path):
    """Create comprehensive visualization with all information"""
    
    num_layers = min(env.num_layers, 6)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.4)
    
    fig.suptitle(f'{agent_name} - Multi-layer PCB Routing Analysis\n'
                 f'Total Reward: {total_reward:.2f} | '
                 f'Success: {"✅" if env._check_design_rules() and env._check_performance_threshold() else "❌"}',
                 fontsize=16, fontweight='bold')
    
    # ===== Layer Visualizations (Top 2 rows, columns 0-2) =====
    for layer_idx in range(num_layers):
        row = layer_idx // 3
        col = layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        im = visualize_layer(env, layer_idx, ax)
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # ===== 3D View (Top right) =====
    ax_3d = fig.add_subplot(gs[0:2, 3:5], projection='3d')
    layer_colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    for layer_idx in range(env.num_layers):
        grid = env.routing_grid[:, :, layer_idx]
        z = layer_idx
        
        trace_positions = np.where(grid == 0.5)
        if len(trace_positions[0]) > 0:
            ax_3d.scatter(trace_positions[0], trace_positions[1], 
                         [z] * len(trace_positions[0]),
                         c=layer_colors[layer_idx % len(layer_colors)],
                         marker='s', s=15, alpha=0.6)
        
        pin_positions = np.where(grid == 1.0)
        if len(pin_positions[0]) > 0:
            ax_3d.scatter(pin_positions[0], pin_positions[1],
                         [z] * len(pin_positions[0]),
                         c='red', marker='*', s=80, alpha=0.9)
    
    for via in env.vias:
        z_coords = list(range(via.from_layer, via.to_layer + 1))
        ax_3d.plot([via.x]*len(z_coords), [via.y]*len(z_coords), z_coords, 
                   'ko-', linewidth=2, alpha=0.7)
    
    ax_3d.set_xlabel('X', fontsize=10)
    ax_3d.set_ylabel('Y', fontsize=10)
    ax_3d.set_zlabel('Layer', fontsize=10)
    ax_3d.set_title('3D Multi-layer View', fontsize=12, fontweight='bold')
    
    # ===== Performance Metrics (Row 2) =====
    
    # Wire Length
    ax1 = fig.add_subplot(gs[2, 0:2])
    ax1.plot(metrics_history['steps'], metrics_history['wire_length'], 
             linewidth=2, color='blue', label='Wire Length')
    ax1.axhline(env.wire_length_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.wire_length_threshold}')
    ax1.fill_between(metrics_history['steps'], metrics_history['wire_length'], 
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Step', fontsize=10)
    ax1.set_ylabel('Wire Length', fontsize=10)
    ax1.set_title('Wire Length Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # SI/PI Performance
    ax2 = fig.add_subplot(gs[2, 2:4])
    ax2.plot(metrics_history['steps'], metrics_history['si_pi_performance'], 
             linewidth=2, color='green', label='SI/PI Performance')
    ax2.axhline(env.si_pi_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.si_pi_threshold}')
    ax2.fill_between(metrics_history['steps'], metrics_history['si_pi_performance'],
                     alpha=0.3, color='green')
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('SI/PI Performance', fontsize=10)
    ax2.set_title('SI/PI Performance Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # DRC Violations
    ax3 = fig.add_subplot(gs[2, 4])
    ax3.plot(metrics_history['steps'], metrics_history['drc_violations'], 
             linewidth=2, color='red', label='DRC Violations')
    ax3.fill_between(metrics_history['steps'], metrics_history['drc_violations'],
                     alpha=0.3, color='red')
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Violations', fontsize=10)
    ax3.set_title('DRC Violations', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ===== Additional Metrics (Row 3) =====
    
    # Rewards per step
    ax4 = fig.add_subplot(gs[3, 0:2])
    ax4.plot(metrics_history['steps'], metrics_history['rewards'], 
             linewidth=1.5, color='purple', alpha=0.7)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('Reward', fontsize=10)
    ax4.set_title('Reward per Step', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Unrouted Nets
    ax5 = fig.add_subplot(gs[3, 2])
    ax5.plot(metrics_history['steps'], metrics_history['unrouted_nets'], 
             linewidth=2, color='orange')
    ax5.fill_between(metrics_history['steps'], metrics_history['unrouted_nets'],
                     alpha=0.3, color='orange')
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('Unrouted Nets', fontsize=10)
    ax5.set_title('Unrouted Nets', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Total Vias
    ax6 = fig.add_subplot(gs[3, 3])
    ax6.plot(metrics_history['steps'], metrics_history['total_vias'], 
             linewidth=2, color='brown')
    ax6.fill_between(metrics_history['steps'], metrics_history['total_vias'],
                     alpha=0.3, color='brown')
    ax6.set_xlabel('Step', fontsize=10)
    ax6.set_ylabel('Total Vias', fontsize=10)
    ax6.set_title('Via Count', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # ===== Statistics Text Box =====
    ax_stats = fig.add_subplot(gs[3, 4])
    ax_stats.axis('off')
    
    stats_text = f"""
📊 Final Statistics

🔧 Configuration:
  Grid: {env.grid_size[0]}×{env.grid_size[1]}
  Layers: {env.num_layers}
  Components: {env.num_components}
  Pins: {len(env.pins)}
  Nets: {len(env.nets)}

📏 Metrics:
  Wire: {env.wire_length:.1f}/{env.wire_length_threshold:.1f}
  SI/PI: {env.si_pi_performance:.3f}/{env.si_pi_threshold:.3f}
  Vias: {env.total_vias}
  Unrouted: {env.unrouted_nets}

⚠️ DRC:
  Violations: {sum(env.drc_violations.values())}

✅ Result:
  {'SUCCESS' if env._check_design_rules() and env._check_performance_threshold() else 'FAILED'}
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Add legend
    legend_elements = [
        Patch(facecolor='lightgreen', label='Trace'),
        Patch(facecolor='blue', label='Via'),
        Patch(facecolor='red', label='Pin'),
        Circle((0, 0), radius=0.1, color='red', label='Component'),
        Circle((0, 0), radius=0.1, color='yellow', edgecolor='black', label='Via Point')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comprehensive visualization saved to: {save_path}")
    
    plt.show()


def compare_algorithms(results_dict, save_path):
    """Create comparison plot for multiple algorithms"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=18, fontweight='bold')
    
    algorithms = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    # 1. Total Rewards
    ax = axes[0, 0]
    rewards = [results_dict[algo]['total_reward'] for algo in algorithms]
    bars = ax.bar(algorithms, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Total Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Final Wire Length
    ax = axes[0, 1]
    wire_lengths = [results_dict[algo]['final_metrics']['wire_length'] for algo in algorithms]
    bars = ax.bar(algorithms, wire_lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Wire Length', fontsize=12)
    ax.set_title('Final Wire Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, wl in zip(bars, wire_lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{wl:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Final SI/PI Performance
    ax = axes[0, 2]
    si_pi = [results_dict[algo]['final_metrics']['si_pi_performance'] for algo in algorithms]
    bars = ax.bar(algorithms, si_pi, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('SI/PI Performance', fontsize=12)
    ax.set_title('Final SI/PI Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, sp in zip(bars, si_pi):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sp:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Wire Length Evolution
    ax = axes[1, 0]
    for i, algo in enumerate(algorithms):
        metrics = results_dict[algo]['metrics']
        ax.plot(metrics['wire_length'], label=algo, linewidth=2, color=colors[i])
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Wire Length', fontsize=12)
    ax.set_title('Wire Length Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. SI/PI Evolution
    ax = axes[1, 1]
    for i, algo in enumerate(algorithms):
        metrics = results_dict[algo]['metrics']
        ax.plot(metrics['si_pi_performance'], label=algo, linewidth=2, color=colors[i])
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('SI/PI Performance', fontsize=12)
    ax.set_title('SI/PI Performance Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 6. DRC Violations
    ax = axes[1, 2]
    violations = [results_dict[algo]['final_metrics']['drc_violations'] for algo in algorithms]
    colors_viol = ['green' if v == 0 else 'red' for v in violations]
    bars = ax.bar(algorithms, violations, color=colors_viol, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('DRC Violations', fontsize=12)
    ax.set_title('Final DRC Violations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, violations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{v}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {save_path}")
    
    plt.show()


# ==================== Main Functions ====================

def visualize_single_agent(model_path, task, agent_type, save_dir):
    """Visualize single agent comprehensively"""
    
    print(f"\n{'='*70}")
    print(f"🎨 Visualizing {agent_type} Agent")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Task: {task}")
    print(f"{'='*70}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load environment and agent
    env = CircuitRoutingEnv(task=task)
    agent = load_agent(model_path, env, agent_type)
    
    # Run episode
    print(f"🏃 Running episode...")
    total_reward, metrics_history, final_env = run_episode(agent, env, agent_type)
    
    # Print statistics
    success = print_detailed_statistics(final_env, agent_type, total_reward)
    
    # Create comprehensive visualization
    save_path = os.path.join(save_dir, f'{agent_type}_comprehensive.png')
    create_comprehensive_visualization(final_env, metrics_history, agent_type, 
                                      total_reward, save_path)
    
    # Save metrics to JSON
    json_path = os.path.join(save_dir, f'{agent_type}_results.json')
    results = {
        'agent_type': agent_type,
        'task': task,
        'total_reward': float(total_reward),
        'success': success,
        'final_metrics': {
            'wire_length': float(final_env.wire_length),
            'si_pi_performance': float(final_env.si_pi_performance),
            'total_vias': int(final_env.total_vias),
            'unrouted_nets': int(final_env.unrouted_nets),
            'drc_violations': int(sum(final_env.drc_violations.values()))
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {json_path}")
    
    return results, metrics_history


def compare_all_agents(agent_configs, task, save_dir):
    """Compare multiple agents"""
    
    print(f"\n{'='*70}")
    print(f"🔬 Comparing Multiple Agents")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Agents: {[cfg['name'] for cfg in agent_configs]}")
    print(f"{'='*70}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_dict = {}
    
    for config in agent_configs:
        agent_name = config['name']
        model_path = config['model_path']
        agent_type = config['type']
        
        print(f"\n{'─'*70}")
        print(f"Processing: {agent_name}")
        print(f"{'─'*70}")
        
        agent_dir = os.path.join(save_dir, agent_name)
        results, metrics = visualize_single_agent(model_path, task, agent_type, agent_dir)
        
        results_dict[agent_name] = {
            'total_reward': results['total_reward'],
            'final_metrics': results['final_metrics'],
            'metrics': metrics
        }
    
    # Create comparison plot
    comparison_path = os.path.join(save_dir, 'algorithm_comparison.png')
    compare_algorithms(results_dict, comparison_path)
    
    # Save comparison JSON
    json_path = os.path.join(save_dir, 'comparison_results.json')
    comparison_results = {
        agent_name: {
            'total_reward': data['total_reward'],
            'final_metrics': data['final_metrics']
        }
        for agent_name, data in results_dict.items()
    }
    
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Comparison Complete!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Circuit Routing Visualization')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare'],
                        help='Visualization mode')
    parser.add_argument('--model_path', type=str,
                        default='./ppo_circuit_routing/ppo_circuit_final.pt',
                        help='Path to model (for single mode)')
    parser.add_argument('--agent_type', type=str, default='PPO',
                        choices=['PPO', 'DQN'],
                        help='Agent type (for single mode)')
    parser.add_argument('--task', type=str, default='circuit_routing_easy',
                        help='Task name')
    parser.add_argument('--save_dir', type=str, default='./unified_visualizations',
                        help='Save directory')
    parser.add_argument('--compare_config', type=str, default=None,
                        help='JSON config file for comparison mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        visualize_single_agent(args.model_path, args.task, args.agent_type, args.save_dir)
    
    elif args.mode == 'compare':
        if args.compare_config and os.path.exists(args.compare_config):
            with open(args.compare_config, 'r') as f:
                agent_configs = json.load(f)
        else:
            # Default comparison config
            agent_configs = [
                {
                    'name': 'PPO',
                    'model_path': './ppo_circuit_routing/ppo_circuit_final.pt',
                    'type': 'PPO'
                },
                # Add more agents as needed
            ]
        
        compare_all_agents(agent_configs, args.task, args.save_dir)


if __name__ == '__main__':
    main()