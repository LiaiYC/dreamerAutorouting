#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit Routing 3D Visualization Tool
"""

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Chinese font settings
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Circuit3DVisualizer:
    def __init__(self, logdir):
        self.logdir = pathlib.Path(logdir)
        self.episodes = []
        self.load_episodes()
    
    def load_episodes(self):
        """Load episode data from training logs"""
        print(f"Loading from {self.logdir}...")
        
        train_dir = self.logdir / 'train_eps'
        eval_dir = self.logdir / 'eval_eps'
        
        episode_files = []
        if train_dir.exists():
            episode_files.extend(list(train_dir.glob("*.npz")))
        if eval_dir.exists():
            episode_files.extend(list(eval_dir.glob("*.npz")))
        
        if not episode_files:
            print("Warning: No episode files found, using demo data")
            self.create_demo_episode()
            return
        
        print(f"Found {len(episode_files)} episode files")
        
        episode_files = sorted(episode_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        for ep_file in episode_files[:5]:
            try:
                data = np.load(ep_file)
                episode = {
                    'image': data.get('image', None),
                    'vector': data.get('vector', None),
                    'action': data.get('action', None),
                    'reward': data.get('reward', None),
                }
                self.episodes.append(episode)
                print(f"  Loaded: {ep_file.name}")
            except Exception as e:
                print(f"  Failed to load {ep_file.name}: {e}")
        
        if not self.episodes:
            print("Warning: Cannot load any episode, using demo data")
            self.create_demo_episode()
    
    def create_demo_episode(self):
        """Create demo circuit data"""
        print("Generating demo circuit data...")
        
        grid_size = (20, 20)
        num_components = 10
        
        episode = {
            'components': [],
            'routes': [],
            'grid_size': grid_size,
        }
        
        np.random.seed(42)
        for i in range(num_components):
            x = np.random.randint(2, grid_size[0] - 2)
            y = np.random.randint(2, grid_size[1] - 2)
            z = 0
            episode['components'].append({
                'id': i,
                'position': (x, y, z),
                'name': f'C{i}'
            })
        
        for i in range(num_components - 1):
            start = episode['components'][i]['position']
            end = episode['components'][i + 1]['position']
            path = self._manhattan_route(start, end)
            episode['routes'].append({
                'start': i,
                'end': i + 1,
                'path': path,
                'wire_length': len(path)
            })
        
        self.demo_episode = episode
        print(f"Generated {num_components} components and {len(episode['routes'])} routes")
    
    def _manhattan_route(self, start, end):
        """Generate Manhattan path"""
        path = [start]
        x, y, z = start
        ex, ey, ez = end
        
        while x != ex:
            x += 1 if x < ex else -1
            path.append((x, y, z))
        
        while y != ey:
            y += 1 if y < ey else -1
            path.append((x, y, z))
        
        return path
    
    def visualize_3d_static(self, save_path=None):
        """Static 3D visualization"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        print("\nGenerating 3D visualization...")
        
        if hasattr(self, 'demo_episode'):
            episode = self.demo_episode
        else:
            self.create_demo_episode()
            episode = self.demo_episode
        
        # Draw grid base
        grid_size = episode['grid_size']
        x_grid = np.arange(0, grid_size[0] + 1)
        y_grid = np.arange(0, grid_size[1] + 1)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        ax.plot_surface(X, Y, Z, alpha=0.1, color='lightgray')
        
        # Draw components
        for comp in episode['components']:
            x, y, z = comp['position']
            ax.scatter(x, y, z, c='blue', s=200, marker='o', 
                      edgecolors='black', linewidths=2, alpha=0.8)
            ax.text(x, y, z + 0.5, comp['name'], fontsize=10, 
                   ha='center', fontweight='bold')
        
        # Draw routing paths
        colors = plt.cm.rainbow(np.linspace(0, 1, len(episode['routes'])))
        
        for route, color in zip(episode['routes'], colors):
            path = np.array(route['path'])
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   color=color, linewidth=3, alpha=0.7, 
                   label=f"Route {route['start']}->{route['end']}")
            
            if len(path) > 1:
                mid = len(path) // 2
                dx = path[mid, 0] - path[mid - 1, 0]
                dy = path[mid, 1] - path[mid - 1, 1]
                dz = path[mid, 2] - path[mid - 1, 2]
                ax.quiver(path[mid - 1, 0], path[mid - 1, 1], path[mid - 1, 2],
                         dx, dy, dz, color=color, arrow_length_ratio=0.3, 
                         linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X (Grid)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Grid)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (Layer)', fontsize=12, fontweight='bold')
        ax.set_title('電路佈線 3D 可視化', fontsize=16, fontweight='bold', pad=20)
        
        ax.view_init(elev=25, azim=45)
        
        if len(episode['routes']) <= 10:
            ax.legend(loc='upper left', fontsize=8)
        
        total_wire_length = sum(r['wire_length'] for r in episode['routes'])
        stats_text = f"元件數: {len(episode['components'])}\n"
        stats_text += f"佈線數: {len(episode['routes'])}\n"
        stats_text += f"總線長: {total_wire_length}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def visualize_3d_multiple_views(self, save_path=None):
        """Multiple view 3D visualization"""
        fig = plt.figure(figsize=(20, 5))
        
        if hasattr(self, 'demo_episode'):
            episode = self.demo_episode
        else:
            self.create_demo_episode()
            episode = self.demo_episode
        
        views = [
            {'elev': 90, 'azim': 0, 'title': '俯視圖 (Top)'},
            {'elev': 0, 'azim': 0, 'title': '前視圖 (Front)'},
            {'elev': 0, 'azim': 90, 'title': '側視圖 (Side)'},
            {'elev': 30, 'azim': 45, 'title': '透視圖 (3D)'},
        ]
        
        for idx, view in enumerate(views, 1):
            ax = fig.add_subplot(1, 4, idx, projection='3d')
            
            for comp in episode['components']:
                x, y, z = comp['position']
                ax.scatter(x, y, z, c='blue', s=100, marker='o', 
                          edgecolors='black', linewidths=1.5, alpha=0.8)
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(episode['routes'])))
            for route, color in zip(episode['routes'], colors):
                path = np.array(route['path'])
                ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                       color=color, linewidth=2, alpha=0.7)
            
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.set_title(view['title'], fontsize=10, fontweight='bold')
            ax.view_init(elev=view['elev'], azim=view['azim'])
        
        plt.suptitle('電路佈線多視角展示', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def visualize_interactive(self):
        """Interactive 3D visualization"""
        print("\nLaunching interactive 3D window...")
        print("Tip: Use mouse to rotate the view")
        self.visualize_3d_static(save_path=None)
    
    def create_animation(self, save_path='circuit_animation.gif'):
        """Create rotation animation"""
        print("\nGenerating rotation animation...")
        
        if hasattr(self, 'demo_episode'):
            episode = self.demo_episode
        else:
            self.create_demo_episode()
            episode = self.demo_episode
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for comp in episode['components']:
            x, y, z = comp['position']
            ax.scatter(x, y, z, c='blue', s=200, marker='o', 
                      edgecolors='black', linewidths=2, alpha=0.8)
            ax.text(x, y, z + 0.5, comp['name'], fontsize=10, ha='center')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(episode['routes'])))
        for route, color in zip(episode['routes'], colors):
            path = np.array(route['path'])
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   color=color, linewidth=3, alpha=0.7)
        
        ax.set_xlabel('X (Grid)', fontsize=12)
        ax.set_ylabel('Y (Grid)', fontsize=12)
        ax.set_zlabel('Z (Layer)', fontsize=12)
        ax.set_title('電路佈線 3D 動畫', fontsize=14, fontweight='bold')
        
        def update(frame):
            ax.view_init(elev=25, azim=frame)
            return fig,
        
        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), 
                           interval=50, blit=False)
        
        try:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved: {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            print("   Hint: Install pillow with: pip install pillow")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Circuit Routing 3D Visualization')
    parser.add_argument('--logdir', type=str, default='./logs/easy_stable')
    parser.add_argument('--output', type=str, default='./vis_3d')
    parser.add_argument('--mode', type=str, default='interactive',
                      choices=['interactive', 'static', 'multi', 'animation'])
    parser.add_argument('--demo', action='store_true')
    
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Circuit Routing 3D Visualization Tool")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")
    
    viz = Circuit3DVisualizer(args.logdir)
    
    if args.demo:
        viz.create_demo_episode()
    
    if args.mode == 'interactive':
        viz.visualize_interactive()
    elif args.mode == 'static':
        viz.visualize_3d_static(save_path=output_path / 'circuit_3d.png')
    elif args.mode == 'multi':
        viz.visualize_3d_multiple_views(save_path=output_path / 'circuit_3d_views.png')
    elif args.mode == 'animation':
        viz.create_animation(save_path=output_path / 'circuit_animation.gif')
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()