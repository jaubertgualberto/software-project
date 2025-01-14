import heapq
import numpy as np
from typing import List, Tuple

class DStarLite:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the D* Lite algorithm with enhanced error handling
        """
        self.grid = grid.copy() 
        self.start = start
        self.goal = goal
        self.g = {}
        self.rhs = {}
        self.U = []
        self.rows, self.cols = grid.shape
        
        # Initialize costs
        for x in range(self.rows):
            for y in range(self.cols):
                self.g[(x, y)] = float('inf')
                self.rhs[(x, y)] = float('inf')

        self.rhs[self.goal] = 0
        key = self.calculate_key(self.goal)
        heapq.heappush(self.U, (key, self.goal))
        
        # Verify initial conditions
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is outside grid bounds")
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            raise ValueError(f"Goal position {goal} is outside grid bounds")

    def update_vertex(self, state: Tuple[int, int]):
        """
        Update vertex with enhanced error handling
        """
        if not self.is_valid_coordinates(state):
            return  # Silently ignore invalid states
            
        if state != self.goal:
            valid_neighbors = [n for n in self.get_neighbors(state) if self.is_valid_coordinates(n)]
            if valid_neighbors:
                self.rhs[state] = min(
                    self.g[neighbor] + self.get_edge_cost(state, neighbor)
                    for neighbor in valid_neighbors
                )
            else:
                self.rhs[state] = float('inf')

        self.remove_from_queue(state)
        if self.g[state] != self.rhs[state]:
            heapq.heappush(self.U, (self.calculate_key(state), state))

    def get_edge_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """
        Calculate edge cost with penalties for obstacles
        """
        if not self.is_valid_coordinates(neighbor):
            return float('inf')
            
        base_cost = 1.0
        obstacle_penalty = 10.0
        
        # Apply penalty if moving through obstacle
        if self.grid[neighbor] == 1:
            if neighbor == self.goal:  # Allow reaching goal even if blocked
                return base_cost * 2  # Small penalty for blocked goal
            return base_cost * obstacle_penalty
            
        return base_cost

    def is_valid_coordinates(self, state: Tuple[int, int]) -> bool:
        """
        Check if coordinates are within grid bounds
        """
        return (0 <= state[0] < self.rows and 
                0 <= state[1] < self.cols)

    def compute_shortest_path(self):
        """
        Compute shortest path with timeout and error handling
        """
        max_iterations = self.rows * self.cols * 4  # Reasonable upper bound
        iteration = 0
        
        while self.U and iteration < max_iterations:
            k_old, state = heapq.heappop(self.U)
            
            if self.g[state] > self.rhs[state]:
                self.g[state] = self.rhs[state]
                for neighbor in self.get_neighbors(state):
                    if self.is_valid_coordinates(neighbor):
                        self.update_vertex(neighbor)
            else:
                self.g[state] = float('inf')
                self.update_vertex(state)
                for neighbor in self.get_neighbors(state):
                    if self.is_valid_coordinates(neighbor):
                        self.update_vertex(neighbor)
                        
            iteration += 1
            
        if iteration >= max_iterations:
            print("Warning: Path computation reached maximum iterations")

    def reconstruct_path(self) -> List[Tuple[int, int]]:
        """
        Reconstruct path with enhanced error handling
        """
        if self.g[self.start] == float('inf'):
            print("Warning: No valid path to goal exists")
            return [self.start]  # Return current position if no path exists
            
        path = []
        current = self.start
        max_path_length = self.rows * self.cols  # Maximum possible path length
        
        while current != self.goal and len(path) < max_path_length:
            path.append(current)
            neighbors = [n for n in self.get_neighbors(current) 
                       if self.is_valid_coordinates(n)]
            
            if not neighbors:
                break
                
            # Choose next step based on lowest cost
            current = min(neighbors, 
                        key=lambda n: (self.g[n] + self.get_edge_cost(current, n)))
            
        path.append(current)  # Add final position
        
        if current != self.goal:
            print("Warning: Path does not reach goal")
            
        return path
    
    def calculate_key(self, state: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate the priority key for a given state.

        Args:
            state: Tuple representing the cell (x, y)

        Returns:
            Tuple representing the priority key.
        """
        g_rhs = min(self.g[state], self.rhs[state])
        return (g_rhs + self.heuristic(self.start, state), g_rhs)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Compute the heuristic (Manhattan distance).

        Args:
            a: Tuple representing the first cell (x, y)
            b: Tuple representing the second cell (x, y)

        Returns:
            Heuristic value.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

   
    def remove_from_queue(self, state: Tuple[int, int]):
        """
        Remove a state from the priority queue if it exists.

        Args:
            state: Tuple representing the cell (x, y)
        """
        self.U = [(k, s) for k, s in self.U if s != state]
        heapq.heapify(self.U)

   
    def get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the neighbors of a given cell.

        Args:
            state: Tuple representing the cell (x, y)

        Returns:
            List of neighboring cells.
        """
        x, y = state
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return neighbors

    def is_valid(self, state: Tuple[int, int]) -> bool:
        """
        Check if a cell is valid (inside grid and not an obstacle), except for the goal.
        
        Args:
            state: Tuple representing the cell (x, y)

        Returns:
            True if valid, False otherwise.
        """
        x, y = state
        # Allow the goal to be valid even if it's an obstacle
        if state == self.goal:
            return True
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] == 0
    
  
    

