# -*- coding: utf-8 -*-
"""
DreamerV3 Optimized Visualizer - 完整版
強化 DRC 懲罰 + 階層式 Reward Shaping
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys
import json

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv


def load_dreamerv3_model(checkpoint_path, env):
    """Load DreamerV3 model"""
    print("⚠️  No checkpoint - Using random policy for demonstration")
    return None


def hierarchical_reward_shaping(env, prev_drc=0, drc_weight=25.0):
    """
    階層式 Reward Shaping
    優先級：DRC > SI/PI > Wire Length > Completion
    """
    shaped_reward = 0
    breakdown = {}
    
    drc_count = sum(env.drc_violations.values())
    
    # 1. DRC 懲罰/獎勵 (最高優先)
    if drc_count > 0:
        penalty = -drc_weight * drc_count
        # 連續違規加重
        if prev_drc > 0 and drc_count >= prev_drc:
            penalty *= 1.5
        shaped_reward += penalty
        breakdown['drc_penalty'] = penalty
    else:
        bonus = 20.0
        shaped_reward += bonus
        breakdown['drc_bonus'] = bonus
    
    # 2. SI/PI 獎勵
    if env.si_pi_performance >= env.si_pi_threshold:
        bonus = 10.0 * (env.si_pi_performance - env.si_pi_threshold)
        shaped_reward += bonus
        breakdown['si_pi_bonus'] = bonus
    
    # 3. Wire Length 效率
    if env.wire_length <= env.wire_length_threshold:
        efficiency = 1.0 - (env.wire_length / env.wire_length_threshold)
        bonus = 15.0 * efficiency
        shaped_reward += bonus
        breakdown['wire_bonus'] = bonus
    
    # 4. 完成度
    if env.unrouted_nets == 0:
        shaped_reward += 25.0
        breakdown['completion'] = 25.0
    
    return shaped_reward, breakdown


def run_episode(model, env, episode_num, total_episodes, drc_weight=25.0):
    """Run one episode with optimized reward shaping"""
    
    # 自適應 DRC 權重：隨訓練進度增加
    progress = episode_num / total_episodes
    adaptive_weight = drc_weight + (50.0 - drc_weight) * progress
    
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    metrics = {
        'wire_length': [],
        'si_pi': [],
        'drc_violations': [],
        'rewards': [],
        'breakdowns': []
    }
    
    prev_drc = 0
    
    while not done and step < env.max_iterations:
        # Random action (replace with your model)
        if model is None:
            action = np.random.randint(0, env.act_space['action']['discrete'])
        else:
            action = model.select_action(obs, deterministic=True)
        
        obs, orig_reward, done, _ = env.step(action)
        
        # Apply reward shaping
        shaped, breakdown = hierarchical_reward_shaping(env, prev_drc, adaptive_weight)
        
        total_reward += shaped
        prev_drc = sum(env.drc_violations.values())
        
        # Record
        metrics['wire_length'].append(env.wire_length)
        metrics['si_pi'].append(env.si_pi_performance)
        metrics['drc_violations'].append(prev_drc)
        metrics['rewards'].append(shaped)
        metrics['breakdowns'].append(breakdown)
        
        step += 1
    
    return total_reward, metrics, env


def create_visualization(env, metrics, reward, save_dir, ep_num):
    """Create detailed visualization"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    drc_total = sum(env.drc_violations.values())
    title = f'DreamerV3 Optimized - Episode {ep_num}\n'
    title += f'Reward: {reward:.2f} | DRC: {drc_total} | Wire: {env.wire_length:.2f} | SI/PI: {env.si_pi_performance:.3f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Layers (2x3 grid)
    for layer_idx in range(min(6, env.num_layers)):
        row, col = layer_idx // 3, layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        grid = env.routing_grid[:, :, layer_idx]
        im = ax.imshow(grid.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
        
        # Components
        for comp in env.component_positions:
            x, y = int(comp[0]), int(comp[1])
            ax.add_patch(Circle((x, y), 2, color='red', alpha=0.7, zorder=5))
        
        # Pins
        for pin in [p for p in env.pins if p.layer == layer_idx]:
            ax.plot(pin.x, pin.y, 'r*', markersize=6, zorder=10)
        
        # Vias
        for via in env.vias:
            if via.from_layer <= layer_idx <= via.to_layer:
                ax.add_patch(Circle((via.x, via.y), 1, color='yellow', 
                           edgecolor='black', zorder=8))
        
        ax.set_title(f'Layer {layer_idx+1}', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    steps = list(range(len(metrics['rewards'])))
    
    # Reward breakdown
    ax = fig.add_subplot(gs[0, 3])
    drc_pen = [b.get('drc_penalty', 0) for b in metrics['breakdowns']]
    drc_bon = [b.get('drc_bonus', 0) for b in metrics['breakdowns']]
    ax.plot(steps, drc_pen, 'r-', linewidth=2, label='DRC Penalty')
    ax.plot(steps, drc_bon, 'g-', linewidth=2, label='DRC Bonus')
    ax.set_title('Reward Breakdown', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # DRC violations
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(steps, metrics['drc_violations'], 'r-', linewidth=2, marker='o')
    ax.fill_between(steps, metrics['drc_violations'], alpha=0.3, color='red')
    ax.set_title('DRC Violations', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    # SI/PI
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(steps, metrics['si_pi'], 'g-', linewidth=2)
    ax.axhline(env.si_pi_threshold, color='r', linestyle='--', linewidth=2)
    ax.set_title('SI/PI Performance', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Stats
    ax = fig.add_subplot(gs[2, 0:3])
    ax.axis('off')
    
    total_drc_pen = sum([b.get('drc_penalty', 0) for b in metrics['breakdowns']])
    total_drc_bon = sum([b.get('drc_bonus', 0) for b in metrics['breakdowns']])
    
    stats = f"""
📊 Episode {ep_num} Summary:

🔧 Config: Grid {env.grid_size[0]}×{env.grid_size[1]}, Layers {env.num_layers}, Components {env.num_components}

📏 Metrics:
  • Wire Length: {env.wire_length:.2f} / {env.wire_length_threshold:.2f} {'✅' if env.wire_length <= env.wire_length_threshold else '❌'}
  • SI/PI: {env.si_pi_performance:.3f} / {env.si_pi_threshold:.3f} {'✅' if env.si_pi_performance >= env.si_pi_threshold else '❌'}
  • Vias: {env.total_vias}, Unrouted: {env.unrouted_nets}

⚠️ DRC: {drc_total} violations {'✅ ZERO!' if drc_total == 0 else '❌'}

💰 Rewards:
  • Total Shaped: {reward:.2f}
  • DRC Penalty: {total_drc_pen:.2f}
  • DRC Bonus: {total_drc_bon:.2f}

✅ Status: {'✅ SUCCESS' if (env._check_design_rules() and env._check_performance_threshold()) else '❌ FAILED'}
    """
    
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(f'{save_dir}/ep{ep_num}_detail.png', dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: ep{ep_num}_detail.png")
    plt.close()


def create_summary_plot(all_data, save_dir):
    """Create summary across all episodes"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DreamerV3 Optimized - Multi-Episode Summary', fontsize=16, fontweight='bold')
    
    n = len(all_data)
    episodes = list(range(1, n+1))
    
    rewards = [d['reward'] for d in all_data]
    drc_viols = [d['drc'] for d in all_data]
    wire_lens = [d['wire'] for d in all_data]
    si_pis = [d['si_pi'] for d in all_data]
    success = [d['success'] for d in all_data]
    
    # Rewards
    axes[0, 0].plot(episodes, rewards, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 0].axhline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    axes[0, 0].set_title('Episode Rewards', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # DRC
    colors = ['green' if v == 0 else 'red' for v in drc_viols]
    axes[0, 1].bar(episodes, drc_viols, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('DRC Violations', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Success rate
    success_rate = sum(success) / n * 100
    colors_s = ['green' if s else 'red' for s in success]
    axes[0, 2].bar(episodes, [1 if s else 0 for s in success], color=colors_s, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title(f'Success (Rate: {success_rate:.1f}%)', fontweight='bold')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylim([-0.1, 1.1])
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Wire length
    axes[1, 0].plot(episodes, wire_lens, 's-', linewidth=2, markersize=8, color='blue')
    axes[1, 0].axhline(np.mean(wire_lens), color='r', linestyle='--', label=f'Mean: {np.mean(wire_lens):.1f}')
    axes[1, 0].set_title('Wire Length', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # SI/PI
    axes[1, 1].plot(episodes, si_pis, 'D-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axhline(0.8, color='orange', linestyle='--', label='Target', linewidth=2)
    axes[1, 1].set_title('SI/PI Performance', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary stats
    axes[1, 2].axis('off')
    zero_drc = sum([1 for v in drc_viols if v == 0])
    summary_text = f"""
📊 Overall Summary

Episodes: {n}

Success Rate: {success_rate:.1f}%
Zero DRC Rate: {zero_drc/n*100:.1f}%

Mean Reward: {np.mean(rewards):.2f}
Mean DRC: {np.mean(drc_viols):.2f}
Mean Wire: {np.mean(wire_lens):.2f}
Mean SI/PI: {np.mean(si_pis):.3f}
    """
    axes[1, 2].text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/summary.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Summary saved: summary.png")
    plt.close()


def evaluate(model, task, n_episodes=20, drc_weight=25.0, save_dir='./dreamerv3_optimized'):
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("🚀 DREAMERV3 OPTIMIZED EVALUATION")
    print("="*70)
    print(f"Task: {task}")
    print(f"Episodes: {n_episodes}")
    print(f"DRC Weight: {drc_weight} (adaptive)")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    env = CircuitRoutingEnv(task=task)
    
    all_data = []
    success_count = 0
    zero_drc_count = 0
    
    for ep in range(n_episodes):
        print(f"Episode {ep+1}/{n_episodes}...", end=' ')
        
        reward, metrics, final_env = run_episode(model, env, ep+1, n_episodes, drc_weight)
        
        drc = sum(final_env.drc_violations.values())
        success = final_env._check_design_rules() and final_env._check_performance_threshold()
        
        if success:
            success_count += 1
        if drc == 0:
            zero_drc_count += 1
        
        all_data.append({
            'reward': reward,
            'drc': drc,
            'wire': final_env.wire_length,
            'si_pi': final_env.si_pi_performance,
            'success': success
        })
        
        status = "✅" if success else "❌"
        drc_str = "✅" if drc == 0 else f"⚠️{drc}"
        print(f"Reward={reward:.1f}, DRC={drc_str}, {status}")
        
        # Detailed viz for first 5
        if ep < 500:
            create_visualization(final_env, metrics, reward, save_dir, ep+1)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Success Rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"Zero DRC Rate: {zero_drc_count}/{n_episodes} ({zero_drc_count/n_episodes*100:.1f}%)")
    
    rewards = [d['reward'] for d in all_data]
    drcs = [d['drc'] for d in all_data]
    wires = [d['wire'] for d in all_data]
    si_pis = [d['si_pi'] for d in all_data]
    
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean DRC: {np.mean(drcs):.2f} ± {np.std(drcs):.2f}")
    print(f"Mean Wire: {np.mean(wires):.2f} ± {np.std(wires):.2f}")
    print(f"Mean SI/PI: {np.mean(si_pis):.3f} ± {np.std(si_pis):.3f}")
    print("="*70 + "\n")
    
    create_summary_plot(all_data, save_dir)
    
    # Save JSON
    summary = {
        'task': task,
        'n_episodes': n_episodes,
        'success_rate': success_count / n_episodes,
        'zero_drc_rate': zero_drc_count / n_episodes,
        'mean_reward': float(np.mean(rewards)),
        'mean_drc': float(np.mean(drcs)),
        'mean_wire': float(np.mean(wires)),
        'mean_si_pi': float(np.mean(si_pis))
    }
    
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 JSON saved: summary.json\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DreamerV3 Optimized Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--task', type=str, default='circuit_routing_easy')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--drc_weight', type=float, default=25.0)
    parser.add_argument('--save_dir', type=str, default='./dreamerv3_optimized')
    
    args = parser.parse_args()
    
    env = CircuitRoutingEnv(task=args.task)
    model = load_dreamerv3_model(args.checkpoint, env) if args.checkpoint else None
    
    evaluate(model, args.task, args.n_episodes, args.drc_weight, args.save_dir)