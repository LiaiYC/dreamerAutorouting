"""
Visualize Baseline Algorithms (DQN, GA, A*, SA) Circuit Routing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import json
import sys

sys.path.append('.')
from envs.circuit_routing import CircuitRoutingEnv
from baseline_algorithms import DQNAgent, GeneticAlgorithm, AStarAgent, SimulatedAnnealingAgent


def visualize_algorithm_result(agent, agent_type, env, save_dir):
    """Visualize single algorithm result"""
    print(f"\n🎨 Visualizing {agent_type}...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Windows-safe filename (replace * with Star)
    safe_agent_name = agent_type.replace('*', 'Star')
    
    # Run episode
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    # Metrics tracking
    metrics_history = {
        'wire_length': [],
        'si_pi_performance': [],
        'drc_violations': [],
        'unrouted_nets': [],
        'rewards': []
    }
    
    if agent_type == 'DQN':
        # DQN step-by-step
        while not done and step < env.max_iterations:
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Record metrics
            metrics_history['wire_length'].append(env.wire_length)
            metrics_history['si_pi_performance'].append(env.si_pi_performance)
            metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
            metrics_history['unrouted_nets'].append(env.unrouted_nets)
            metrics_history['rewards'].append(reward)
            
            step += 1
            
    elif agent_type == 'GA':
        # GA evolution
        agent.initialize_population(sequence_length=20)
        best_actions, best_reward = agent.evolve(env, generations=10)
        
        # Execute best actions and track
        obs = env.reset()
        for action in best_actions:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            metrics_history['wire_length'].append(env.wire_length)
            metrics_history['si_pi_performance'].append(env.si_pi_performance)
            metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
            metrics_history['unrouted_nets'].append(env.unrouted_nets)
            metrics_history['rewards'].append(reward)
            
            if done:
                break
                
    elif agent_type == 'A*':
        # A* search
        actions = agent.search(env, max_nodes=500)
        
        # Execute and track
        obs = env.reset()
        for action in actions:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            metrics_history['wire_length'].append(env.wire_length)
            metrics_history['si_pi_performance'].append(env.si_pi_performance)
            metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
            metrics_history['unrouted_nets'].append(env.unrouted_nets)
            metrics_history['rewards'].append(reward)
            
            if done:
                break
                
    elif agent_type == 'SA':
        # Simulated Annealing
        actions, _ = agent.optimize(env, max_iterations=500)
        
        # Execute and track
        obs = env.reset()
        for action in actions:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            metrics_history['wire_length'].append(env.wire_length)
            metrics_history['si_pi_performance'].append(env.si_pi_performance)
            metrics_history['drc_violations'].append(sum(env.drc_violations.values()))
            metrics_history['unrouted_nets'].append(env.unrouted_nets)
            metrics_history['rewards'].append(reward)
            
            if done:
                break
    
    # Create visualizations
    create_visualization_plots(env, metrics_history, safe_agent_name, total_reward, save_dir)
    
    return total_reward, metrics_history


def create_visualization_plots(env, metrics_history, agent_type, total_reward, save_dir):
    """Create comprehensive visualization plots"""
    
    # Windows-safe filename
    safe_agent_name = agent_type.replace('*', 'Star')
    
    # 1. Main figure with routing result and metrics
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{agent_type} Circuit Routing Results\nTotal Reward: {total_reward:.2f}',
                 fontsize=18, fontweight='bold')
    
    # Layer visualizations (top 2 rows, left side)
    num_layers = min(env.num_layers, 6)
    for layer_idx in range(num_layers):
        row = layer_idx // 3
        col = layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        grid = env.routing_grid[:, :, layer_idx]
        im = ax.imshow(grid.T, cmap='viridis', origin='lower', interpolation='nearest')
        
        # Mark components
        for comp_pos in env.component_positions:
            x, y = int(comp_pos[0]), int(comp_pos[1])
            circle = Circle((x, y), radius=2, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # Mark pins
        pins_on_layer = [p for p in env.pins if p.layer == layer_idx]
        for pin in pins_on_layer:
            ax.plot(pin.x, pin.y, 'r*', markersize=6)
        
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Metrics plots (right side)
    # Wire Length
    ax1 = fig.add_subplot(gs[0, 3])
    ax1.plot(metrics_history['wire_length'], linewidth=2, color='blue')
    ax1.axhline(env.wire_length_threshold, color='r', linestyle='--', 
                label=f'Threshold: {env.wire_length_threshold}', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Wire Length')
    ax1.set_title('Wire Length Over Time', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # SI/PI Performance
    ax2 = fig.add_subplot(gs[1, 3])
    ax2.plot(metrics_history['si_pi_performance'], linewidth=2, color='green')
    ax2.axhline(env.si_pi_threshold, color='r', linestyle='--',
                label=f'Threshold: {env.si_pi_threshold}', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('SI/PI Performance')
    ax2.set_title('SI/PI Performance', fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # DRC Violations
    ax3 = fig.add_subplot(gs[2, 3])
    ax3.plot(metrics_history['drc_violations'], linewidth=2, color='red')
    ax3.fill_between(range(len(metrics_history['drc_violations'])),
                     metrics_history['drc_violations'], alpha=0.3, color='red')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Violations')
    ax3.set_title('DRC Violations', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Statistics box
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
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    save_path = os.path.join(save_dir, f'{agent_type}_routing_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to: {save_path}")
    
    plt.show()


def visualize_3d_routing(env, agent_type, save_dir):
    """3D visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Windows-safe filename
    safe_agent_name = agent_type.replace('*', 'Star')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    layer_colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    # Plot traces on each layer
    for layer_idx in range(env.num_layers):
        grid = env.routing_grid[:, :, layer_idx]
        z = layer_idx
        
        trace_positions = np.where(grid == 0.5)
        if len(trace_positions[0]) > 0:
            ax.scatter(trace_positions[0], trace_positions[1], 
                      [z] * len(trace_positions[0]),
                      c=layer_colors[layer_idx % len(layer_colors)],
                      marker='s', s=20, alpha=0.6,
                      label=f'Layer {layer_idx + 1}')
        
        pin_positions = np.where(grid == 1.0)
        if len(pin_positions[0]) > 0:
            ax.scatter(pin_positions[0], pin_positions[1],
                      [z] * len(pin_positions[0]),
                      c='red', marker='*', s=100, alpha=0.9)
    
    # Plot vias
    for via in env.vias:
        z_coords = list(range(via.from_layer, via.to_layer + 1))
        x_coords = [via.x] * len(z_coords)
        y_coords = [via.y] * len(z_coords)
        ax.plot(x_coords, y_coords, z_coords, 'ko-', linewidth=3, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer')
    ax.set_title(f'{agent_type} - 3D Multi-layer Routing', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    save_path = os.path.join(save_dir, f'{safe_agent_name}_3d_routing.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 3D view saved to: {save_path}")
    
    plt.show()


def compare_all_algorithms_visual(task, save_dir='./baseline_visualizations'):
    """Compare all algorithms with visualizations"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔬 Baseline Algorithms Visualization Comparison")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"{'='*70}\n")
    
    env = CircuitRoutingEnv(task=task)
    image_shape = env.obs_space['image']['shape']
    vector_dim = env.obs_space['vector']['shape'][0]
    action_dim = env.act_space['action']['discrete']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    # DQN
    print("="*70)
    print("1️⃣ Training and Visualizing DQN...")
    print("="*70)
    dqn_agent = DQNAgent(image_shape, vector_dim, action_dim, device=device)
    
    # Quick training
    obs = env.reset()
    for step in range(50000):
        action = dqn_agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        dqn_agent.store_transition(obs, action, reward, next_obs, done)
        
        if step > 1000 and step % 4 == 0:
            dqn_agent.update()
        
        obs = next_obs if not done else env.reset()
        
        if step % 10000 == 0:
            print(f"  Training step {step}/50000, epsilon={dqn_agent.epsilon:.3f}")
    
    dqn_dir = os.path.join(save_dir, 'DQN')
    reward, metrics = visualize_algorithm_result(dqn_agent, 'DQN', env, dqn_dir)
    visualize_3d_routing(env, 'DQN', dqn_dir)
    results['DQN'] = {'reward': reward, 'metrics': metrics}
    
    # GA
    print("\n" + "="*70)
    print("2️⃣ Visualizing Genetic Algorithm...")
    print("="*70)
    ga_agent = GeneticAlgorithm(action_dim)
    ga_dir = os.path.join(save_dir, 'GA')
    reward, metrics = visualize_algorithm_result(ga_agent, 'GA', env, ga_dir)
    visualize_3d_routing(env, 'GA', ga_dir)
    results['GA'] = {'reward': reward, 'metrics': metrics}
    
    # A*
    print("\n" + "="*70)
    print("3️⃣ Visualizing A* Search...")
    print("="*70)
    astar_agent = AStarAgent(action_dim)
    astar_dir = os.path.join(save_dir, 'AStar')
    reward, metrics = visualize_algorithm_result(astar_agent, 'A*', env, astar_dir)
    visualize_3d_routing(env, 'A*', astar_dir)
    results['A*'] = {'reward': reward, 'metrics': metrics}
    
    # SA
    print("\n" + "="*70)
    print("4️⃣ Visualizing Simulated Annealing...")
    print("="*70)
    sa_agent = SimulatedAnnealingAgent(action_dim)
    sa_dir = os.path.join(save_dir, 'SA')
    reward, metrics = visualize_algorithm_result(sa_agent, 'SA', env, sa_dir)
    visualize_3d_routing(env, 'SA', sa_dir)
    results['SA'] = {'reward': reward, 'metrics': metrics}
    
    # Create comparison plot
    create_comparison_plot(results, save_dir)
    
    # Save results
    results_file = os.path.join(save_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for algo, data in results.items():
            json_results[algo] = {
                'reward': float(data['reward']),
                'final_wire_length': float(data['metrics']['wire_length'][-1]),
                'final_si_pi': float(data['metrics']['si_pi_performance'][-1]),
                'final_violations': int(data['metrics']['drc_violations'][-1]),
                'final_unrouted': int(data['metrics']['unrouted_nets'][-1])
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations completed!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"{'='*70}\n")
    
    return results


def create_comparison_plot(results, save_dir):
    """Create comparison plot across all algorithms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Comparison', fontsize=18, fontweight='bold')
    
    algorithms = list(results.keys())
    colors = {'DQN': 'blue', 'GA': 'green', 'A*': 'red', 'SA': 'purple'}
    
    # 1. Total Rewards
    ax = axes[0, 0]
    rewards = [results[algo]['reward'] for algo in algorithms]
    bars = ax.bar(algorithms, rewards, color=[colors[a] for a in algorithms], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Total Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Wire Length Evolution
    ax = axes[0, 1]
    for algo in algorithms:
        metrics = results[algo]['metrics']
        ax.plot(metrics['wire_length'], label=algo, linewidth=2, color=colors[algo])
    ax.set_xlabel('Step')
    ax.set_ylabel('Wire Length')
    ax.set_title('Wire Length Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. SI/PI Performance
    ax = axes[1, 0]
    for algo in algorithms:
        metrics = results[algo]['metrics']
        ax.plot(metrics['si_pi_performance'], label=algo, linewidth=2, color=colors[algo])
    ax.set_xlabel('Step')
    ax.set_ylabel('SI/PI Performance')
    ax.set_title('SI/PI Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. DRC Violations
    ax = axes[1, 1]
    for algo in algorithms:
        metrics = results[algo]['metrics']
        ax.plot(metrics['drc_violations'], label=algo, linewidth=2, color=colors[algo])
    ax.set_xlabel('Step')
    ax.set_ylabel('DRC Violations')
    ax.set_title('DRC Violations Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'algorithm_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Baseline Algorithms')
    parser.add_argument('--task', type=str, default='circuit_routing_easy',
                        help='Task name')
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['DQN', 'GA', 'AStar', 'SA', 'all'],
                        help='Which algorithm to visualize')
    parser.add_argument('--save_dir', type=str, default='./baseline_visualizations',
                        help='Save directory')
    
    args = parser.parse_args()
    
    if args.algorithm == 'all':
        compare_all_algorithms_visual(args.task, args.save_dir)
    else:
        # Visualize single algorithm
        env = CircuitRoutingEnv(task=args.task)
        image_shape = env.obs_space['image']['shape']
        vector_dim = env.obs_space['vector']['shape'][0]
        action_dim = env.act_space['action']['discrete']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        algo_dir = os.path.join(args.save_dir, args.algorithm)
        
        if args.algorithm == 'DQN':
            agent = DQNAgent(image_shape, vector_dim, action_dim, device=device)
            # Quick training
            print("Training DQN...")
            obs = env.reset()
            for step in range(50000):
                action = agent.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.store_transition(obs, action, reward, next_obs, done)
                if step > 1000 and step % 4 == 0:
                    agent.update()
                obs = next_obs if not done else env.reset()
            visualize_algorithm_result(agent, 'DQN', env, algo_dir)
            visualize_3d_routing(env, 'DQN', algo_dir)
            
        elif args.algorithm == 'GA':
            agent = GeneticAlgorithm(action_dim)
            visualize_algorithm_result(agent, 'GA', env, algo_dir)
            visualize_3d_routing(env, 'GA', algo_dir)
            
        elif args.algorithm == 'AStar':
            agent = AStarAgent(action_dim)
            visualize_algorithm_result(agent, 'A*', env, algo_dir)
            visualize_3d_routing(env, 'A*', algo_dir)
            
        elif args.algorithm == 'SA':
            agent = SimulatedAnnealingAgent(action_dim)
            visualize_algorithm_result(agent, 'SA', env, algo_dir)
            visualize_3d_routing(env, 'SA', algo_dir)


if __name__ == '__main__':
    main()