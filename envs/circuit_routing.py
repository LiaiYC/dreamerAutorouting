# envs/circuit_routing.py
# Enhanced Multi-layer PCB Routing with DRC Rules

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass


class RoutingAlgorithm(Enum):
    """Available AI routing algorithms"""
    MCTS = 0
    GA = 1
    PSO = 2
    DE = 3
    DQN = 4


@dataclass
class DRCRules:
    """Design Rule Check parameters"""
    min_trace_width: float = 0.15  # mm
    min_trace_spacing: float = 0.15  # mm
    min_via_diameter: float = 0.3  # mm
    min_via_to_via_spacing: float = 0.25  # mm
    min_via_to_trace_spacing: float = 0.2  # mm
    min_pad_to_trace_spacing: float = 0.2  # mm
    min_copper_to_board_edge: float = 0.5  # mm
    max_via_aspect_ratio: float = 8.0  # depth/diameter
    
    def check_trace_width(self, width: float) -> bool:
        return width >= self.min_trace_width
    
    def check_trace_spacing(self, spacing: float) -> bool:
        return spacing >= self.min_trace_spacing
    
    def check_via_diameter(self, diameter: float) -> bool:
        return diameter >= self.min_via_diameter
    
    def check_via_spacing(self, spacing: float) -> bool:
        return spacing >= self.min_via_to_via_spacing


@dataclass
class Via:
    """Via representation"""
    x: int
    y: int
    from_layer: int
    to_layer: int
    diameter: float = 0.3
    
    def __hash__(self):
        return hash((self.x, self.y, self.from_layer, self.to_layer))


@dataclass
class Pin:
    """Component pin"""
    pin_id: int
    component_id: int
    x: int
    y: int
    layer: int
    net_id: int
    
    def __hash__(self):
        return hash((self.pin_id, self.component_id))


@dataclass
class Net:
    """Routing net"""
    net_id: int
    pins: List[Pin]
    routed: bool = False
    wire_length: float = 0.0
    num_vias: int = 0
    num_violations: int = 0


class CircuitRoutingEnv:
    """
    Multi-layer PCB Routing Environment with DRC Rules
    """
    
    def __init__(self, task, action_repeat=1, size=(64, 64)):
        """
        Initialize environment
        
        Args:
            task: Task name (e.g., 'circuit_routing_basic')
            action_repeat: Action repeat (handled by wrapper)
            size: Image size (height, width)
        """
        self._task = task
        self._size = size
        
        # Setup task parameters
        self._setup_task_params(task)
        
        # DRC rules
        self.drc_rules = DRCRules()
        
        # Algorithm properties
        self.algorithm_properties = {
            RoutingAlgorithm.MCTS: {
                'complexity': 2, 'convergence_time': 50,
                'wire_length_performance': 0.7, 'computation_cost': 200,
                'prone_to_local_optima': True, 'via_efficiency': 0.6
            },
            RoutingAlgorithm.GA: {
                'complexity': 2, 'convergence_time': 60,
                'wire_length_performance': 0.75, 'computation_cost': 180,
                'prone_to_local_optima': True, 'via_efficiency': 0.7
            },
            RoutingAlgorithm.PSO: {
                'complexity': 2, 'convergence_time': 40,
                'wire_length_performance': 0.9, 'computation_cost': 250,
                'prone_to_local_optima': False, 'via_efficiency': 0.85
            },
            RoutingAlgorithm.DE: {
                'complexity': 2, 'convergence_time': 70,
                'wire_length_performance': 0.95, 'computation_cost': 400,
                'prone_to_local_optima': False, 'via_efficiency': 0.9
            },
            RoutingAlgorithm.DQN: {
                'complexity': 1, 'convergence_time': 80,
                'wire_length_performance': 0.92, 'computation_cost': 350,
                'prone_to_local_optima': False, 'via_efficiency': 0.88
            }
        }
        
        self.reset()
    
    def _setup_task_params(self, task):
        """Setup parameters based on task name"""
        if 'easy' in task:
            self.grid_size = (40, 40)  # 增大網格以容納更多針腳
            self.num_layers = 2
            self.num_components = 8
            self.pins_per_component = 4
            self.min_pins_per_layer = 20
            self.si_pi_threshold = 0.7
            self.wire_length_threshold = 150.0
            self.max_iterations = 30
            self.computation_budget = 800.0
        elif 'hard' in task:
            self.grid_size = (60, 60)
            self.num_layers = 6
            self.num_components = 20
            self.pins_per_component = 8
            self.min_pins_per_layer = 30
            self.si_pi_threshold = 0.85
            self.wire_length_threshold = 250.0
            self.max_iterations = 80
            self.computation_budget = 1500.0
        elif 'expert' in task:
            self.grid_size = (80, 80)
            self.num_layers = 8
            self.num_components = 30
            self.pins_per_component = 12
            self.min_pins_per_layer = 40
            self.si_pi_threshold = 0.9
            self.wire_length_threshold = 350.0
            self.max_iterations = 100
            self.computation_budget = 2000.0
        else:  # medium or basic
            self.grid_size = (50, 50)
            self.num_layers = 4
            self.num_components = 12
            self.pins_per_component = 6
            self.min_pins_per_layer = 25
            self.si_pi_threshold = 0.8
            self.wire_length_threshold = 200.0
            self.max_iterations = 50
            self.computation_budget = 1000.0
        
        self.total_pins = self.num_components * self.pins_per_component
    
    @property
    def obs_space(self):
        """Observation space definition"""
        return {
            'image': {'shape': (*self._size, 3), 'dtype': np.uint8},
            'vector': {'shape': (64,), 'dtype': np.float32},
        }
    
    @property
    def act_space(self):
        """Action space definition"""
        return {'action': {'shape': (), 'dtype': np.int32, 'discrete': 60}}
    
    @property
    def observation_space(self):
        """Gym-style observation space"""
        import gym
        return gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(*self._size, 3), dtype=np.uint8),
            'vector': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32),
        })
    
    @property
    def action_space(self):
        """Gym-style action space"""
        import gym
        return gym.spaces.Discrete(60)
    
    def reset(self):
        """Reset environment"""
        # Circuit parameters
        self.circuit_size = np.random.randint(80, 120)
        self.si_pi_requirement = np.random.uniform(0.7, 0.9)
        self.design_time_limit = np.random.uniform(100, 200)
        
        # Generate components and pins
        self.component_positions = self._generate_component_placement()
        self.pins, self.nets = self._generate_pins_and_nets()
        
        # Multi-layer routing grid
        self.routing_grid = np.zeros((*self.grid_size, self.num_layers), dtype=np.float32)
        self.vias: Set[Via] = set()
        
        # Metrics
        self.wire_length = self.circuit_size * 1.5
        self.si_pi_performance = 0.5
        self.total_vias = 0
        self.drc_violations = {
            'trace_width': 0,
            'trace_spacing': 0,
            'via_spacing': 0,
            'via_to_trace': 0,
            'clearance': 0
        }
        self.unrouted_nets = len(self.nets)
        
        # State tracking
        self.computation_used = 0.0
        self.iterations = 0
        self.reroute_count = 0
        self.rearrange_count = 0
        self.performance_history = []
        self.last_algorithm = None
        self._is_first = True
        
        return self._get_obs()
    
    def _generate_component_placement(self) -> np.ndarray:
        """Generate component positions with minimum spacing"""
        positions = []
        min_spacing = 5
        
        for _ in range(self.num_components):
            max_attempts = 100
            for _ in range(max_attempts):
                x = np.random.randint(min_spacing, self.grid_size[0] - min_spacing)
                y = np.random.randint(min_spacing, self.grid_size[1] - min_spacing)
                
                valid = True
                for px, py in positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_spacing:
                        valid = False
                        break
                
                if valid:
                    positions.append([x, y])
                    break
        
        return np.array(positions, dtype=np.float32)
    
    def _generate_pins_and_nets(self) -> Tuple[List[Pin], List[Net]]:
        """Generate pins and nets ensuring minimum pins per layer"""
        pins = []
        pin_id = 0
        
        # Generate pins for each component
        for comp_id, pos in enumerate(self.component_positions):
            x, y = int(pos[0]), int(pos[1])
            
            for i in range(self.pins_per_component):
                # Offset pins around component
                offset_x = (i % 2) * 2 - 1
                offset_y = (i // 2) * 2 - 1
                pin_x = max(0, min(self.grid_size[0] - 1, x + offset_x))
                pin_y = max(0, min(self.grid_size[1] - 1, y + offset_y))
                
                # Assign layer (distribute across layers)
                layer = i % self.num_layers
                
                pin = Pin(
                    pin_id=pin_id,
                    component_id=comp_id,
                    x=pin_x,
                    y=pin_y,
                    layer=layer,
                    net_id=-1  # Will be assigned
                )
                pins.append(pin)
                pin_id += 1
        
        # Ensure minimum pins per layer
        pins_per_layer = [sum(1 for p in pins if p.layer == l) for l in range(self.num_layers)]
        for layer_idx, count in enumerate(pins_per_layer):
            while count < self.min_pins_per_layer:
                # Add extra pins to this layer
                comp_id = np.random.randint(0, self.num_components)
                x, y = self.component_positions[comp_id].astype(int)
                offset_x = np.random.randint(-2, 3)
                offset_y = np.random.randint(-2, 3)
                pin_x = max(0, min(self.grid_size[0] - 1, x + offset_x))
                pin_y = max(0, min(self.grid_size[1] - 1, y + offset_y))
                
                pin = Pin(
                    pin_id=pin_id,
                    component_id=comp_id,
                    x=pin_x,
                    y=pin_y,
                    layer=layer_idx,
                    net_id=-1
                )
                pins.append(pin)
                pin_id += 1
                count += 1
        
        # Create nets (connect random pins)
        nets = []
        num_nets = len(pins) // 3  # Average 3 pins per net
        available_pins = list(range(len(pins)))
        np.random.shuffle(available_pins)
        
        net_id = 0
        while available_pins and net_id < num_nets:
            pins_in_net = min(np.random.randint(2, 5), len(available_pins))
            net_pin_indices = [available_pins.pop() for _ in range(pins_in_net)]
            
            net_pins = [pins[i] for i in net_pin_indices]
            for p in net_pins:
                p.net_id = net_id
            
            net = Net(net_id=net_id, pins=net_pins)
            nets.append(net)
            net_id += 1
        
        return pins, nets
    
    def step(self, action):
        """Execute action"""
        if isinstance(action, dict):
            action = action['action']
        
        # Convert to integer
        if isinstance(action, (int, np.integer)):
            action = int(action)
        elif isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action.flatten()[0])
        elif hasattr(action, 'item'):
            try:
                action = int(action.item())
            except:
                action = int(action.flatten()[0].item())
        else:
            action = int(action)
        
        # Decode action
        algorithm_idx, reroute_flag, rearrange_flag, partition_strategy = \
            self._action_to_components(action)
        
        algorithms = list(RoutingAlgorithm)
        selected_algorithm = algorithms[algorithm_idx]
        self.last_algorithm = selected_algorithm
        
        reward = 0.0
        self._is_first = False
        
        # Execute routing
        if rearrange_flag == 1:
            self._rearrange_components()
            self.rearrange_count += 1
            reward -= 5
        
        if reroute_flag == 1 or self.iterations == 0:
            routing_result = self._execute_routing(selected_algorithm, partition_strategy)
            self.wire_length = routing_result['wire_length']
            self.si_pi_performance = routing_result['si_pi_performance']
            self.total_vias = routing_result['num_vias']
            self.unrouted_nets = routing_result['unrouted_nets']
            self.drc_violations = routing_result['drc_violations']
            self.computation_used += routing_result['computation_cost']
            if reroute_flag == 1:
                self.reroute_count += 1
        
        self.iterations += 1
        
        # Evaluate performance
        design_rules_satisfied = self._check_design_rules()
        performance_threshold_met = self._check_performance_threshold()
        
        reward += self._calculate_reward(design_rules_satisfied, performance_threshold_met, selected_algorithm)
        
        self.performance_history.append({
            'iteration': self.iterations,
            'wire_length': self.wire_length,
            'si_pi_performance': self.si_pi_performance,
            'algorithm': selected_algorithm.value,
            'drc_violations': sum(self.drc_violations.values())
        })
        
        # Check if episode is done
        is_last = False
        if design_rules_satisfied and performance_threshold_met:
            is_last = True
            reward += 150
        elif self.iterations >= self.max_iterations:
            is_last = True
            reward -= 50
        elif self.computation_used >= self.computation_budget:
            is_last = True
            reward -= 30
        
        obs = self._get_obs()
        info = {'is_last': is_last, 'is_terminal': is_last}
        
        return obs, reward, is_last, info
    
    def _action_to_components(self, action: int) -> Tuple[int, int, int, int]:
        """Decode action into components"""
        algorithm_idx = action // 12
        remaining = action % 12
        reroute_flag = remaining // 6
        remaining = remaining % 6
        rearrange_flag = remaining // 3
        partition_strategy = remaining % 3
        return algorithm_idx, reroute_flag, rearrange_flag, partition_strategy
    
    def _execute_routing(self, algorithm: RoutingAlgorithm, partition_strategy: int) -> Dict:
        """Execute routing with DRC checking"""
        props = self.algorithm_properties[algorithm]
        
        # Calculate wire length
        base_wire_length = self._calculate_minimum_wire_length()
        wire_length_improvement = props['wire_length_performance']
        
        if props['prone_to_local_optima'] and np.random.rand() < 0.3:
            wire_length_improvement *= 0.8
        
        wire_length = base_wire_length * (1.2 - wire_length_improvement * 0.4)
        
        # Calculate SI/PI performance
        si_pi_base = 0.5 + np.random.uniform(0, 0.3)
        si_pi_improvement = min(wire_length_improvement + 0.1, 0.95)
        si_pi_performance = si_pi_base + (1 - si_pi_base) * si_pi_improvement
        
        # Via count (using via_efficiency)
        via_efficiency = props['via_efficiency']
        base_vias = len(self.nets) * 2  # Average 2 vias per net
        num_vias = int(base_vias * (1.5 - via_efficiency * 0.5))
        
        # DRC violations
        drc_violations = self._simulate_drc_violations(wire_length_improvement, via_efficiency)
        
        # Unrouted nets
        routing_success_rate = wire_length_improvement
        unrouted_nets = max(0, int(len(self.nets) * (1 - routing_success_rate)))
        
        # Update routing grid and vias
        self._update_routing_grid(wire_length_improvement, num_vias)
        
        return {
            'wire_length': wire_length,
            'si_pi_performance': si_pi_performance,
            'num_vias': num_vias,
            'unrouted_nets': unrouted_nets,
            'drc_violations': drc_violations,
            'computation_cost': props['computation_cost']
        }
    
    def _calculate_minimum_wire_length(self) -> float:
        """Calculate theoretical minimum wire length for all nets"""
        total_length = 0.0
        for net in self.nets:
            if len(net.pins) < 2:
                continue
            # Minimum spanning tree approximation
            pins = net.pins
            for i in range(len(pins) - 1):
                p1, p2 = pins[i], pins[i + 1]
                manhattan = abs(p1.x - p2.x) + abs(p1.y - p2.y)
                layer_change = abs(p1.layer - p2.layer) * 2  # Via penalty
                total_length += manhattan + layer_change
        return total_length
    
    def _simulate_drc_violations(self, routing_quality: float, via_efficiency: float) -> Dict[str, int]:
        """Simulate DRC violations based on routing quality"""
        violation_base = 10 * (1 - routing_quality)
        
        return {
            'trace_width': max(0, int(np.random.poisson(violation_base * 0.2))),
            'trace_spacing': max(0, int(np.random.poisson(violation_base * 0.4))),
            'via_spacing': max(0, int(np.random.poisson(violation_base * 0.3 * (1 - via_efficiency)))),
            'via_to_trace': max(0, int(np.random.poisson(violation_base * 0.2))),
            'clearance': max(0, int(np.random.poisson(violation_base * 0.3)))
        }
    
    def _update_routing_grid(self, improvement: float, num_vias: int):
        """Update multi-layer routing grid with traces and vias"""
        self.routing_grid.fill(0.0)
        self.vias.clear()
        
        # Place component pins
        for pin in self.pins:
            if 0 <= pin.x < self.grid_size[0] and 0 <= pin.y < self.grid_size[1]:
                self.routing_grid[pin.x, pin.y, pin.layer] = 1.0
        
        # Route nets
        num_nets_to_route = int(len(self.nets) * improvement)
        for i in range(num_nets_to_route):
            net = self.nets[i % len(self.nets)]
            if len(net.pins) >= 2:
                self._draw_net_route(net)
        
        # Add vias
        for _ in range(num_vias):
            x = np.random.randint(2, self.grid_size[0] - 2)
            y = np.random.randint(2, self.grid_size[1] - 2)
            from_layer = np.random.randint(0, self.num_layers - 1)
            to_layer = np.random.randint(from_layer + 1, self.num_layers)
            
            via = Via(x, y, from_layer, to_layer)
            self.vias.add(via)
            
            # Mark via on grid
            for layer in range(from_layer, to_layer + 1):
                self.routing_grid[x, y, layer] = 0.8
    
    def _draw_net_route(self, net: Net):
        """Draw route for a net across multiple layers"""
        pins = net.pins
        if len(pins) < 2:
            return
        
        for i in range(len(pins) - 1):
            start_pin = pins[i]
            end_pin = pins[i + 1]
            
            # Route on starting layer
            self._draw_manhattan_route(
                start_pin.x, start_pin.y, start_pin.layer,
                end_pin.x, end_pin.y, start_pin.layer
            )
            
            # Add via if layers differ
            if start_pin.layer != end_pin.layer:
                via = Via(end_pin.x, end_pin.y, start_pin.layer, end_pin.layer)
                self.vias.add(via)
                
                # Route on ending layer
                self._draw_manhattan_route(
                    end_pin.x, end_pin.y, end_pin.layer,
                    end_pin.x, end_pin.y, end_pin.layer
                )
    
    def _draw_manhattan_route(self, x1: int, y1: int, layer1: int, 
                               x2: int, y2: int, layer2: int):
        """Draw Manhattan routing on a layer"""
        x, y = x1, y1
        
        # Route horizontally
        while x != x2:
            x += 1 if x < x2 else -1
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.routing_grid[x, y, layer1] = 0.5
        
        # Route vertically
        while y != y2:
            y += 1 if y < y2 else -1
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.routing_grid[x, y, layer1] = 0.5
    
    def _rearrange_components(self):
        """Rearrange component positions"""
        num_to_move = max(1, self.num_components // 4)
        indices = np.random.choice(self.num_components, num_to_move, replace=False)
        
        for idx in indices:
            x = np.random.randint(5, self.grid_size[0] - 5)
            y = np.random.randint(5, self.grid_size[1] - 5)
            self.component_positions[idx] = [x, y]
    
    def _check_design_rules(self) -> bool:
        """Check if all DRC rules are satisfied"""
        total_violations = sum(self.drc_violations.values())
        return total_violations == 0 and self.unrouted_nets == 0
    
    def _check_performance_threshold(self) -> bool:
        """Check if performance meets requirements"""
        return (self.wire_length <= self.wire_length_threshold and 
                self.si_pi_performance >= self.si_pi_threshold)
    
    def _calculate_reward(self, design_rules_satisfied: bool, 
                         performance_threshold_met: bool, 
                         algorithm: RoutingAlgorithm) -> float:
        """Calculate reward based on routing quality"""
        reward = 0.0
        
        # Performance reward
        if performance_threshold_met:
            reward += 30
        else:
            wire_ratio = min(1.0, self.wire_length / self.wire_length_threshold)
            si_pi_ratio = min(1.0, self.si_pi_performance / self.si_pi_threshold)
            reward += 15 * (2 - wire_ratio) + 15 * si_pi_ratio
        
        # DRC compliance reward
        if design_rules_satisfied:
            reward += 25
        else:
            # Penalize violations
            total_violations = sum(self.drc_violations.values())
            reward -= total_violations * 2
            reward -= self.unrouted_nets * 5
        
        # Via count penalty (too many vias is bad)
        optimal_vias = len(self.nets) * 1.5
        via_ratio = abs(self.total_vias - optimal_vias) / optimal_vias
        reward -= 5 * via_ratio
        
        # Efficiency rewards
        if self.computation_budget > 0:
            computation_efficiency = max(0, 1 - (self.computation_used / self.computation_budget))
            reward += 8 * computation_efficiency
        
        time_efficiency = max(0, 1 - (self.iterations / self.max_iterations))
        reward += 7 * time_efficiency
        
        return reward
    
    def _render_grid_to_image(self) -> np.ndarray:
        """Render multi-layer grid to RGB image"""
        h, w = self._size
        scale_x = h // self.grid_size[0]
        scale_y = w // self.grid_size[1]
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Render each layer with different colors
        layer_colors = [
            [255, 0, 0],    # Red - Top layer
            [0, 255, 0],    # Green - Layer 2
            [0, 0, 255],    # Blue - Layer 3
            [255, 255, 0],  # Yellow - Layer 4
            [255, 0, 255],  # Magenta - Layer 5
            [0, 255, 255],  # Cyan - Layer 6
            [255, 128, 0],  # Orange - Layer 7
            [128, 0, 255],  # Purple - Layer 8
        ]
        
        # Composite all layers
        for layer in range(self.num_layers):
            color = layer_colors[layer % len(layer_colors)]
            layer_weight = 1.0 / (layer + 1)  # Top layers more visible
            
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    value = self.routing_grid[i, j, layer]
                    if value > 0:
                        x_start = i * scale_x
                        y_start = j * scale_y
                        x_end = min((i + 1) * scale_x, h)
                        y_end = min((j + 1) * scale_y, w)
                        
                        intensity = int(255 * value * layer_weight)
                        layer_color = [int(c * intensity / 255) for c in color]
                        
                        # Blend with existing
                        current = image[x_start:x_end, y_start:y_end]
                        blended = np.clip(current + layer_color, 0, 255).astype(np.uint8)
                        image[x_start:x_end, y_start:y_end] = blended
        
        # Mark vias with white dots
        for via in self.vias:
            x_start = via.x * scale_x
            y_start = via.y * scale_y
            x_end = min((via.x + 1) * scale_x, h)
            y_end = min((via.y + 1) * scale_y, w)
            image[x_start:x_end, y_start:y_end] = [255, 255, 255]
        
        return image
    
    def _get_obs(self):
        """Get observation"""
        image = self._render_grid_to_image()
        
        # Calculate statistics
        total_violations = sum(self.drc_violations.values())
        pins_per_layer = [sum(1 for p in self.pins if p.layer == l) for l in range(self.num_layers)]
        
        # 確保 pins_per_layer 有 8 個元素
        while len(pins_per_layer) < 8:
            pins_per_layer.append(0)
        
        vector_components = [
            # Performance metrics (4)
            self.wire_length / self.wire_length_threshold,
            self.si_pi_performance,
            self.total_vias / max(1, len(self.nets) * 2),
            self.unrouted_nets / max(1, len(self.nets)),
            
            # DRC violations (6)
            self.drc_violations['trace_width'] / max(1, len(self.nets)),
            self.drc_violations['trace_spacing'] / max(1, len(self.nets)),
            self.drc_violations['via_spacing'] / max(1, self.total_vias + 1),
            self.drc_violations['via_to_trace'] / max(1, len(self.nets)),
            self.drc_violations['clearance'] / max(1, len(self.nets)),
            total_violations / max(1, len(self.nets) * 5),
            
            # Resource usage (4)
            self.computation_used / self.computation_budget,
            self.iterations / self.max_iterations,
            self.reroute_count / max(1, self.iterations + 1),
            self.rearrange_count / max(1, self.iterations + 1),
            
            # Circuit properties (6)
            self.circuit_size / 120.0,
            self.si_pi_requirement,
            self.design_time_limit / 200.0,
            self.num_layers / 8.0,
            len(self.nets) / 50.0,
            len(self.pins) / 400.0,
            
            # Component statistics (4)
            self.component_positions[:, 0].mean() / self.grid_size[0],
            self.component_positions[:, 1].mean() / self.grid_size[1],
            self.component_positions[:, 0].std() / self.grid_size[0] if len(self.component_positions) > 1 else 0.0,
            self.component_positions[:, 1].std() / self.grid_size[1] if len(self.component_positions) > 1 else 0.0,
            
            # Pins per layer distribution (8)
            pins_per_layer[0] / max(1, self.total_pins),
            pins_per_layer[1] / max(1, self.total_pins),
            pins_per_layer[2] / max(1, self.total_pins),
            pins_per_layer[3] / max(1, self.total_pins),
            pins_per_layer[4] / max(1, self.total_pins),
            pins_per_layer[5] / max(1, self.total_pins),
            pins_per_layer[6] / max(1, self.total_pins),
            pins_per_layer[7] / max(1, self.total_pins),
            
            # Layer utilization (1)
            np.mean([np.sum(self.routing_grid[:, :, l] > 0) / (self.grid_size[0] * self.grid_size[1]) 
                     for l in range(self.num_layers)]),
            
            # Via statistics (1)
            len(self.vias) / max(1, len(self.nets) * 3),
            
            # Historical performance (1)
            len(self.performance_history) / self.max_iterations,
            
            # Algorithm indicator (5)
            1.0 if self.last_algorithm == RoutingAlgorithm.MCTS else 0.0,
            1.0 if self.last_algorithm == RoutingAlgorithm.GA else 0.0,
            1.0 if self.last_algorithm == RoutingAlgorithm.PSO else 0.0,
            1.0 if self.last_algorithm == RoutingAlgorithm.DE else 0.0,
            1.0 if self.last_algorithm == RoutingAlgorithm.DQN else 0.0,
            
            # Grid occupancy statistics (4)
            np.sum(self.routing_grid > 0) / max(1, self.grid_size[0] * self.grid_size[1] * self.num_layers),
            np.sum(self.routing_grid == 1.0) / max(1, np.sum(self.routing_grid > 0)),
            np.sum(self.routing_grid == 0.5) / max(1, np.sum(self.routing_grid > 0)),
            np.sum(self.routing_grid == 0.8) / max(1, np.sum(self.routing_grid > 0)),
            
            # Success indicators (2)
            1.0 if self._check_design_rules() else 0.0,
            1.0 if self._check_performance_threshold() else 0.0,
        ]
        
        # 計算當前有多少個特徵 (應該是 46)
        current_count = len(vector_components)
        
        # 補齊到 64 維
        padding_needed = 64 - current_count
        vector_components.extend([0.0] * padding_needed)
        
        vector = np.array(vector_components, dtype=np.float32)
        
        # 確保正好是 64 維
        assert vector.shape[0] == 64, f"Vector dimension mismatch: got {vector.shape[0]}, expected 64"
        
        return {
            'image': image,
            'vector': vector,
            'is_first': self._is_first,
            'is_last': False,
            'is_terminal': False,
        }