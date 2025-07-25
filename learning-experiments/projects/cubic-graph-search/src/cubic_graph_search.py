"""
Cubic Graph Search with USDZ Visualization
==========================================

A 3D cube mesh graph implementation using AIMA search algorithms with USDZ visualization.
This project demonstrates graph search algorithms on a cubic lattice structure where
each node is color-coded by spatial position for easy navigation and algorithm visualization.

Author: Jesse B.
Course: Stanford AI Fundamentals
Date: July 2025
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import json
from pathlib import Path

# Simple fix: Add AIMA-python root to path
aima_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, aima_root)

# Import AIMA modules
from search import Problem, breadth_first_graph_search, depth_first_graph_search, astar_search
from search import breadth_first_graph_search, uniform_cost_search
from utils import PriorityQueue

class CubicNode:
    """
    Represents a node in the 3D cubic mesh.
    
    Each node has:
    - 3D coordinates (x, y, z)
    - Color coding based on spatial position
    - Connections to adjacent nodes
    """
    
    def __init__(self, x: int, y: int, z: int, cube_size: int = 5):
        self.x, self.y, self.z = x, y, z
        self.cube_size = cube_size
        self.position = (x, y, z)
        self.color = self._calculate_color()
        self.neighbors: Set[Tuple[int, int, int]] = set()
        
    def _calculate_color(self) -> str:
        """
        Calculate color based on position in cube for spatial orientation.
        
        Color coding system:
        - Red: Front face (z = max)
        - Blue: Back face (z = 0)
        - Green: Right face (x = max)
        - Yellow: Left face (x = 0)
        - White: Top face (y = max)
        - Orange: Bottom face (y = 0)
        - Gray: Interior nodes
        """
        colors = []
        
        # Check if node is on any face
        if self.x == 0:
            colors.append("Yellow")  # Left face
        elif self.x == self.cube_size - 1:
            colors.append("Green")   # Right face
            
        if self.y == 0:
            colors.append("Orange")  # Bottom face
        elif self.y == self.cube_size - 1:
            colors.append("White")   # Top face
            
        if self.z == 0:
            colors.append("Blue")    # Back face
        elif self.z == self.cube_size - 1:
            colors.append("Red")     # Front face
        
        # Return primary color or gray for interior
        if len(colors) == 0:
            return "Gray"  # Interior node
        elif len(colors) == 1:
            return colors[0]  # Face node
        else:
            return f"{'-'.join(colors)}"  # Edge/corner node
    
    def add_neighbor(self, neighbor_pos: Tuple[int, int, int]):
        """Add a neighboring node position."""
        self.neighbors.add(neighbor_pos)
    
    def __str__(self):
        return f"Node({self.x},{self.y},{self.z})-{self.color}"
    
    def __repr__(self):
        return self.__str__()

class CubicMesh:
    """
    3D cubic mesh graph structure with spatial indexing and neighbor relationships.
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.nodes: Dict[Tuple[int, int, int], CubicNode] = {}
        self.edges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
        self._build_mesh()
    
    def _build_mesh(self):
        """Build the complete cubic mesh with nodes and edges."""
        print(f"Building {self.size}x{self.size}x{self.size} cubic mesh...")
        
        # Create all nodes
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    node = CubicNode(x, y, z, self.size)
                    self.nodes[(x, y, z)] = node
        
        # Create edges (6-connectivity: up, down, left, right, front, back)
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for pos, node in self.nodes.items():
            x, y, z = pos
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Check bounds
                if (0 <= nx < self.size and 
                    0 <= ny < self.size and 
                    0 <= nz < self.size):
                    
                    neighbor_pos = (nx, ny, nz)
                    node.add_neighbor(neighbor_pos)
                    self.edges.append((pos, neighbor_pos))
        
        print(f"Created {len(self.nodes)} nodes and {len(self.edges)} directed edges")
    
    def get_node(self, position: Tuple[int, int, int]) -> Optional[CubicNode]:
        """Get node at specific position."""
        return self.nodes.get(position)
    
    def get_neighbors(self, position: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get neighbor positions for a given node."""
        node = self.get_node(position)
        return list(node.neighbors) if node else []
    
    def distance(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

class CubicGraphProblem(Problem):
    """
    AIMA Problem formulation for searching through the cubic mesh.
    
    This adapts our cube mesh to work with AIMA search algorithms.
    """
    
    def __init__(self, mesh: CubicMesh, initial: Tuple[int, int, int], 
                 goal: Tuple[int, int, int]):
        """
        Initialize the search problem.
        
        Args:
            mesh: The cubic mesh graph
            initial: Starting position (x, y, z)
            goal: Target position (x, y, z)
        """
        super().__init__(initial, goal)
        self.mesh = mesh
        self.goal_position = goal
        
        print(f"Search Problem: {initial} -> {goal}")
        print(f"Start node: {mesh.get_node(initial)}")
        print(f"Goal node: {mesh.get_node(goal)}")
    
    def actions(self, state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Return available actions (neighboring positions) from current state."""
        return self.mesh.get_neighbors(state)
    
    def result(self, state: Tuple[int, int, int], 
               action: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Return the result of taking an action from a state."""
        return action  # Action is the destination position
    
    def goal_test(self, state: Tuple[int, int, int]) -> bool:
        """Test if we've reached the goal."""
        return state == self.goal_position
    
    def path_cost(self, c: float, state1: Tuple[int, int, int], 
                  action: Tuple[int, int, int], state2: Tuple[int, int, int]) -> float:
        """Calculate cost of path (uniform cost for adjacent moves)."""
        return c + 1  # Each move costs 1
    
    def h(self, node) -> float:
        """Heuristic function for A* search (Manhattan distance)."""
        state = node.state
        return sum(abs(a - b) for a, b in zip(state, self.goal_position))

class SearchResultAnalyzer:
    """
    Analyzes and compares results from different search algorithms.
    """
    
    def __init__(self, mesh: CubicMesh):
        self.mesh = mesh
        self.results: Dict[str, Dict] = {}
    
    def run_algorithm(self, algorithm_name: str, algorithm_func, problem: CubicGraphProblem):
        """Run a search algorithm and store results."""
        print(f"\n--- Running {algorithm_name} ---")
        
        try:
            # Run the algorithm
            solution = algorithm_func(problem)
            
            if solution:
                path = solution.path()
                path_positions = [node.state for node in path]
                
                result = {
                    'success': True,
                    'path_length': len(path_positions),
                    'path_positions': path_positions,
                    'path_colors': [self.mesh.get_node(pos).color for pos in path_positions],
                    'cost': solution.path_cost,
                    'solution_node': solution
                }
                
                print(f"✓ Solution found! Path length: {result['path_length']}")
                print(f"  Path: {' -> '.join([str(pos) for pos in path_positions[:5]])}{'...' if len(path_positions) > 5 else ''}")
                print(f"  Colors: {' -> '.join(result['path_colors'][:5])}{'...' if len(result['path_colors']) > 5 else ''}")
                
            else:
                result = {
                    'success': False,
                    'error': 'No solution found'
                }
                print("✗ No solution found")
                
        except Exception as e:
            result = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ Error: {e}")
        
        self.results[algorithm_name] = result
        return result
    
    def compare_algorithms(self) -> Dict:
        """Compare performance of different algorithms."""
        comparison = {
            'successful_algorithms': [],
            'path_lengths': {},
            'efficiency_ranking': []
        }
        
        for name, result in self.results.items():
            if result['success']:
                comparison['successful_algorithms'].append(name)
                comparison['path_lengths'][name] = result['path_length']
        
        # Rank by path length (shorter = better)
        if comparison['path_lengths']:
            comparison['efficiency_ranking'] = sorted(
                comparison['path_lengths'].items(), 
                key=lambda x: x[1]
            )
        
        return comparison

def generate_usdz_model(mesh: CubicMesh, search_results: Dict[str, Dict], 
                       output_file: str = "cubic_search_visualization.usdz"):
    """
    Generate USDZ 3D model showing the cube mesh and search paths.
    
    Note: This is a simplified representation. In practice, you'd use
    libraries like USD-Python or export to formats that can be converted to USDZ.
    """
    print(f"\n--- Generating USDZ Visualization ---")
    
    # For this example, we'll create a JSON representation that describes
    # the 3D model structure. In practice, you'd use USD tools.
    
    model_data = {
        "cube_size": mesh.size,
        "nodes": {},
        "edges": [],
        "search_paths": {},
        "color_legend": {
            "Red": "Front face (z=max)",
            "Blue": "Back face (z=0)", 
            "Green": "Right face (x=max)",
            "Yellow": "Left face (x=0)",
            "White": "Top face (y=max)",
            "Orange": "Bottom face (y=0)",
            "Gray": "Interior nodes"
        }
    }
    
    # Add node data
    for pos, node in mesh.nodes.items():
        model_data["nodes"][str(pos)] = {
            "position": pos,
            "color": node.color,
            "neighbors": list(node.neighbors)
        }
    
    # Add edge data
    for edge in mesh.edges:
        model_data["edges"].append([edge[0], edge[1]])
    
    # Add search path data
    for alg_name, result in search_results.items():
        if result['success']:
            model_data["search_paths"][alg_name] = {
                "path": result['path_positions'],
                "colors": result['path_colors'],
                "length": result['path_length']
            }
    
    # Save as JSON (would be converted to USDZ in production)
    with open(output_file.replace('.usdz', '.json'), 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"✓ Model data saved to {output_file.replace('.usdz', '.json')}")
    print("  (In production, this would be converted to USDZ format)")
    
    return model_data

def main():
    """
    Main execution function demonstrating cubic graph search algorithms.
    """
    print("="*60)
    print("CUBIC GRAPH SEARCH WITH AIMA ALGORITHMS")
    print("Stanford AI Class Preparation Project")
    print("="*60)
    
    # 1. Create the cubic mesh
    print("\n1. BUILDING CUBIC MESH")
    mesh = CubicMesh(size=5)  # 5x5x5 cube
    
    # 2. Define search problem
    print("\n2. DEFINING SEARCH PROBLEM")
    start_pos = (0, 0, 0)  # Bottom-left-back corner (Yellow-Orange-Blue)
    goal_pos = (4, 4, 4)   # Top-right-front corner (Green-White-Red)
    
    problem = CubicGraphProblem(mesh, start_pos, goal_pos)
    
    # 3. Run different search algorithms
    print("\n3. RUNNING SEARCH ALGORITHMS")
    analyzer = SearchResultAnalyzer(mesh)
    
    # Test various AIMA search algorithms
    algorithms = {
        'Breadth-First Search': breadth_first_graph_search,
        'Depth-First Search': depth_first_graph_search,
        'Uniform Cost Search': uniform_cost_search,
        'A* Search': astar_search,
    }
    
    for name, algorithm in algorithms.items():
        analyzer.run_algorithm(name, algorithm, problem)
    
    # 4. Analyze results
    print("\n4. ANALYZING RESULTS")
    comparison = analyzer.compare_algorithms()
    
    print(f"Successful algorithms: {comparison['successful_algorithms']}")
    print("Efficiency ranking (by path length):")
    for rank, (alg, length) in enumerate(comparison['efficiency_ranking'], 1):
        print(f"  {rank}. {alg}: {length} steps")
    
    # 5. Generate visualization
    print("\n5. GENERATING VISUALIZATION")
    model_data = generate_usdz_model(mesh, analyzer.results)
    
    # 6. Generate report
    print("\n6. GENERATING REPORT")
    generate_report(mesh, analyzer.results, comparison)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("Check the 'reports' directory for detailed analysis.")
    print("="*60)

def generate_report(mesh: CubicMesh, results: Dict, comparison: Dict):
    """Generate a detailed analysis report."""
    
    report_content = f"""
# Cubic Graph Search Algorithm Analysis Report

## Experiment Overview
- **Cube Size**: {mesh.size}x{mesh.size}x{mesh.size} ({len(mesh.nodes)} nodes)
- **Search Space**: 3D lattice with 6-connectivity
- **Start Position**: (0,0,0) - Yellow-Orange-Blue corner
- **Goal Position**: ({mesh.size-1},{mesh.size-1},{mesh.size-1}) - Green-White-Red corner

## Algorithm Performance Comparison

"""
    
    for alg_name, result in results.items():
        if result['success']:
            report_content += f"""
### {alg_name}
- **Status**: ✓ Success
- **Path Length**: {result['path_length']} steps
- **Path Cost**: {result['cost']}
- **Color Sequence**: {' → '.join(result['path_colors'][:10])}{'...' if len(result['path_colors']) > 10 else ''}

"""
        else:
            report_content += f"""
### {alg_name}
- **Status**: ✗ Failed
- **Error**: {result.get('error', 'Unknown error')}

"""
    
    report_content += f"""
## Key Findings

1. **Most Efficient Algorithm**: {comparison['efficiency_ranking'][0][0] if comparison['efficiency_ranking'] else 'None'}
2. **Optimal Path Length**: {comparison['efficiency_ranking'][0][1] if comparison['efficiency_ranking'] else 'N/A'} steps
3. **Success Rate**: {len(comparison['successful_algorithms'])}/{len(results)} algorithms found solutions

## Color-Coded Navigation System

The cubic mesh uses a spatial color coding system:
- **Red**: Front face (z=max) - "approaching" 
- **Blue**: Back face (z=0) - "receding"
- **Green**: Right face (x=max) - "rightward"
- **Yellow**: Left face (x=0) - "leftward" 
- **White**: Top face (y=max) - "upward"
- **Orange**: Bottom face (y=0) - "downward"
- **Gray**: Interior nodes - "internal"

This system enables intuitive spatial reasoning about algorithm behavior and path characteristics.

## Applications and Extensions

This framework can be extended for:
1. **Pathfinding in 3D environments**
2. **Robot navigation in structured spaces**
3. **Network routing optimization**
4. **Game AI spatial reasoning**
5. **3D maze solving algorithms**

---
Generated: {os.path.basename(__file__)} - Stanford AI Class Preparation
"""
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/cubic_search_analysis.md', 'w') as f:
        f.write(report_content)
    
    print("✓ Report saved to reports/cubic_search_analysis.md")

if __name__ == "__main__":
    main()
