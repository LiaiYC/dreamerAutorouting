# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 12:52:59 2025

@author: user
"""

"""
Individual Model Review Tool
Simple interface to review each trained model separately
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import json
import sys
from datetime import datetime

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


def review_ppo_model(model_path, task, save_dir='./review_ppo'):
    """Review PPO model"""
    from ppo_train_circuit import PPOAgent
    
    print("\n" + "="*70)
    print("🤖 REVIEWING PPO MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Task: {task}")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load environment
    env = CircuitRoutingEnv(task=task)
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load agent
    agent = PPOAgent(image_shape, vector_dim, action_dim, device=device)
    
    try:
        agent.load(model_path)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run episode
    print("🏃 Running test episode...")
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    action_history = []
    reward_history = []
    state_history = {
        'wire_length': [],
        'si_pi': [],
        'drc_violations': [],
        'unrouted_nets': []
    }
    
    while not done and step < env.max_iterations:
        action, log_prob, value = agent.select_action(obs, deterministic=True)
        action_history.append(action)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        reward_history.append(reward)
        
        state_history['wire_length'].append(env.wire_length)
        state_history['si_pi'].append(env.si_pi_performance)
        state_history['drc_violations'].append(sum(env.drc_violations.values()))
        state_history['unrouted_nets'].append(env.unrouted_nets)
        
        step += 1
    
    print(f"✅ Episode completed in {step} steps\n")
    
    # Print detailed results
    print_results(env, total_reward, step, state_history)
    
    # Visualize
    visualize_results(env, state_history, reward_history, 'PPO', save_dir)
    
    # Save summary
    save_summary(env, total_reward, step, state_history, 'PPO', save_dir)
    
    return env, state_history


def review_dqn_model(model_path, task, save_dir='./review_dqn'):
    """Review DQN model"""
    from baseline_algorithms import DQNAgent
    
    print("\n" + "="*70)
    print("🤖 REVIEWING DQN MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Task: {task}")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load environment
    env = CircuitRoutingEnv(task=task)
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load agent
    agent = DQNAgent(image_shape, vector_dim, action_dim, device=device)
    
    try:
        checkpoint = torch.load(model_path)
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run episode
    print("🏃 Running test episode...")
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    action_history = []
    reward_history = []
    state_history = {
        'wire_length': [],
        'si_pi': [],
        'drc_violations': [],
        'unrouted_nets': []
    }
    
    while not done and step < env.max_iterations:
        action = agent.select_action(obs, eval_mode=True)
        action_history.append(action)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        reward_history.append(reward)
        
        state_history['wire_length'].append(env.wire_length)
        state_history['si_pi'].append(env.si_pi_performance)
        state_history['drc_violations'].append(sum(env.drc_violations.values()))
        state_history['unrouted_nets'].append(env.unrouted_nets)
        
        step += 1
    
    print(f"✅ Episode completed in {step} steps\n")
    
    # Print detailed results
    print_results(env, total_reward, step, state_history)
    
    # Visualize
    visualize_results(env, state_history, reward_history, 'DQN', save_dir)
    
    # Save summary
    save_summary(env, total_reward, step, state_history, 'DQN', save_dir)
    
    return env, state_history


def print_results(env, total_reward, steps, state_history):
    """Print detailed results"""
    print("="*70)
    print("📊 DETAILED RESULTS")
    print("="*70)
    
    print(f"\n🎮 Episode Summary:")
    print(f"  Total Reward:      {total_reward:.2f}")
    print(f"  Total Steps:       {steps}")
    print(f"  Avg Reward/Step:   {total_reward/steps:.2f}")
    
    print(f"\n🔧 Circuit Configuration:")
    print(f"  Grid Size:         {env.grid_size[0]} × {env.grid_size[1]}")
    print(f"  Number of Layers:  {env.num_layers}")
    print(f"  Components:        {env.num_components}")
    print(f"  Total Pins:        {len(env.pins)}")
    print(f"  Total Nets:        {len(env.nets)}")
    
    print(f"\n📏 Final Routing Metrics:")
    print(f"  Wire Length:       {env.wire_length:.2f} / {env.wire_length_threshold:.2f}")
    wire_pass = "✅ PASS" if env.wire_length <= env.wire_length_threshold else "❌ FAIL"
    print(f"                     {wire_pass}")
    
    print(f"  SI/PI Performance: {env.si_pi_performance:.3f} / {env.si_pi_threshold:.3f}")
    sipi_pass = "✅ PASS" if env.si_pi_performance >= env.si_pi_threshold else "❌ FAIL"
    print(f"                     {sipi_pass}")
    
    print(f"  Total Vias:        {env.total_vias}")
    print(f"  Unrouted Nets:     {env.unrouted_nets}")
    
    print(f"\n⚠️  DRC Violations Detail:")
    total_viol = 0
    for viol_type, count in env.drc_violations.items():
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {viol_type:20s}: {count:3d}")
        total_viol += count
    print(f"  {'─'*50}")
    print(f"  {'Total Violations:':<24} {total_viol:3d}")
    
    print(f"\n📈 Performance Trajectory:")
    print(f"  Initial Wire Length:   {state_history['wire_length'][0]:.2f}")
    print(f"  Final Wire Length:     {state_history['wire_length'][-1]:.2f}")
    print(f"  Improvement:           {state_history['wire_length'][0] - state_history['wire_length'][-1]:.2f}")
    
    print(f"\n  Initial SI/PI:         {state_history['si_pi'][0]:.3f}")
    print(f"  Final SI/PI:           {state_history['si_pi'][-1]:.3f}")
    print(f"  Improvement:           {state_history['si_pi'][-1] - state_history['si_pi'][0]:+.3f}")
    
    print(f"\n  Initial DRC Violations: {state_history['drc_violations'][0]}")
    print(f"  Final DRC Violations:   {state_history['drc_violations'][-1]}")
    
    print(f"\n📍 Pin Distribution by Layer:")
    for layer in range(env.num_layers):
        pins_on_layer = [p for p in env.pins if p.layer == layer]
        print(f"  Layer {layer + 1}: {len(pins_on_layer):3d} pins")
    
    print(f"\n✅ Success Criteria:")
    drc_pass = env._check_design_rules()
    perf_pass = env._check_performance_threshold()
    overall = drc_pass and perf_pass
    
    print(f"  DRC Rules Satisfied:   {'✅ YES' if drc_pass else '❌ NO'}")
    print(f"  Performance Met:       {'✅ YES' if perf_pass else '❌ NO'}")
    print(f"  Overall Result:        {'✅ SUCCESS' if overall else '❌ FAILED'}")
    
    print("="*70 + "\n")


def visualize_results(env, state_history, reward_history, model_name, save_dir):
    """Create comprehensive visualization matching baseline format"""
    
    # Windows-safe filename
    safe_model_name = model_name.replace('*', 'Star')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    drc_pass = env._check_design_rules()
    perf_pass = env._check_performance_threshold()
    success = drc_pass and perf_pass
    total_reward = sum(reward_history)
    
    fig.suptitle(f'{model_name} Circuit Routing Results\nTotal Reward: {total_reward:.2f}',
                 fontsize=18, fontweight='bold')
    
    # ===== PCB Layer Views (Top 2 rows, columns 0-2) =====
    num_layers = min(env.num_layers, 6)
    for layer_idx in range(num_layers):
        row = layer_idx // 3
        col = layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        grid = env.routing_grid[:, :, layer_idx]
        im = ax.imshow(grid.T, cmap='viridis', origin='lower', interpolation='nearest', vmin=0, vmax=1)
        
        # Components (red circles)
        for comp_pos in env.component_positions:
            x, y = int(comp_pos[0]), int(comp_pos[1])
            circle = Circle((x, y), radius=2, color='red', alpha=0.7, zorder=5)
            ax.add_patch(circle)
        
        # Pins (red stars)
        pins_on_layer = [p for p in env.pins if p.layer == layer_idx]
        for pin in pins_on_layer:
            ax.plot(pin.x, pin.y, 'r*', markersize=6, zorder=10)
        
        # Vias (yellow circles)
        vias_on_layer = [v for v in env.vias if v.from_layer <= layer_idx <= v.to_layer]
        for via in vias_on_layer:
            circle = Circle((via.x, via.y), radius=1, color='yellow', 
                           edgecolor='black', linewidth=1, zorder=8)
            ax.add_patch(circle)
        
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # ===== Performance Metrics (Right side, column 3) =====
    steps = list(range(len(state_history['wire_length'])))
    
    # Wire Length Over Time
    ax1 = fig.add_subplot(gs[0, 3])
    ax1.plot(steps, state_history['wire_length'], 'b-', linewidth=2)
    ax1.axhline(env.wire_length_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.wire_length_threshold}')
    ax1.set_ylabel('Wire Length', fontsize=10)
    ax1.set_title('Wire Length Over Time', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # SI/PI Performance
    ax2 = fig.add_subplot(gs[1, 3])
    ax2.plot(steps, state_history['si_pi'], 'g-', linewidth=2)
    ax2.axhline(env.si_pi_threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold: {env.si_pi_threshold}')
    ax2.set_ylabel('SI/PI Performance', fontsize=10)
    ax2.set_title('SI/PI Performance', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # DRC Violations
    ax3 = fig.add_subplot(gs[2, 3])
    ax3.plot(steps, state_history['drc_violations'], 'r-', linewidth=2)
    ax3.fill_between(steps, state_history['drc_violations'], alpha=0.3, color='red')
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Violations', fontsize=10)
    ax3.set_title('DRC Violations', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== Statistics Text Box (Bottom, columns 0-2) =====
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
  • DRC Rules: {'✅ PASS' if drc_pass else '❌ FAIL'}
  • Performance: {'✅ PASS' if perf_pass else '❌ FAIL'}
  • Overall: {'✅ SUCCESS' if success else '❌ FAILED'}
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    save_path = os.path.join(save_dir, f'{safe_model_name}_review.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved to: {save_path}")
    
    plt.show()


def save_summary(env, total_reward, steps, state_history, model_name, save_dir):
    """Save summary to JSON"""
    
    # Windows-safe filename
    safe_model_name = model_name.replace('*', 'Star')
    
    summary = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'episode_summary': {
            'total_reward': float(total_reward),
            'total_steps': int(steps),
            'avg_reward_per_step': float(total_reward / steps)
        },
        'circuit_config': {
            'grid_size': env.grid_size,
            'num_layers': env.num_layers,
            'num_components': env.num_components,
            'total_pins': len(env.pins),
            'total_nets': len(env.nets)
        },
        'final_metrics': {
            'wire_length': float(env.wire_length),
            'wire_length_threshold': float(env.wire_length_threshold),
            'si_pi_performance': float(env.si_pi_performance),
            'si_pi_threshold': float(env.si_pi_threshold),
            'total_vias': int(env.total_vias),
            'unrouted_nets': int(env.unrouted_nets),
            'drc_violations': {k: int(v) for k, v in env.drc_violations.items()},
            'total_violations': int(sum(env.drc_violations.values()))
        },
        'success_criteria': {
            'drc_rules_satisfied': bool(env._check_design_rules()),
            'performance_met': bool(env._check_performance_threshold()),
            'overall_success': bool(env._check_design_rules() and env._check_performance_threshold())
        },
        'trajectory': {
            'initial_wire_length': float(state_history['wire_length'][0]),
            'final_wire_length': float(state_history['wire_length'][-1]),
            'wire_length_improvement': float(state_history['wire_length'][0] - state_history['wire_length'][-1]),
            'initial_si_pi': float(state_history['si_pi'][0]),
            'final_si_pi': float(state_history['si_pi'][-1]),
            'si_pi_improvement': float(state_history['si_pi'][-1] - state_history['si_pi'][0]),
            'initial_drc_violations': int(state_history['drc_violations'][0]),
            'final_drc_violations': int(state_history['drc_violations'][-1])
        }
    }
    
    json_path = os.path.join(save_dir, f'{safe_model_name}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved to: {json_path}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Review Individual Model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['ppo', 'dqn'],
                        help='Model type to review')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file')
    parser.add_argument('--task', type=str, default='circuit_routing_easy',
                        help='Task name')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Auto-generate save directory
    if args.save_dir is None:
        args.save_dir = f'./review_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    if args.model == 'ppo':
        review_ppo_model(args.model_path, args.task, args.save_dir)
    elif args.model == 'dqn':
        review_dqn_model(args.model_path, args.task, args.save_dir)


if __name__ == '__main__':
    main()