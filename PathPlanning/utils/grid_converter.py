from typing import Tuple, List
import numpy as np
from utils.Point import Point
from rsoccer_gym.Entities import Robot
import matplotlib.pyplot as plt


class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 24):
        """
        Initialize grid converter
        
        Args:
            field_length: Length of the field in meters
            field_width: Width of the field in meters
            grid_size: Number of cells in each dimension
        """
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate cell dimensions
        self.cell_length = field_length / grid_size
        self.cell_width = field_width / grid_size
        
        self.neighbors = {}



    def create_grid(self, obstacles: dict, robot_radius: float) -> np.ndarray:
        """
        Create a grid representation of the environment
        
        Args:
            obstacles: Dictionary of robot obstacles {id: robot}
            robot_radius: Radius of robots for collision checking
            
        Returns:
            Binary grid where 1 represents obstacles and 0 represents free space
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # For each cell, check if it intersects with any obstacle
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_x = -self.field_length/2 + i * self.cell_length + self.cell_length/2
                cell_y = -self.field_width/2 + j * self.cell_width + self.cell_width/2
                
                # Check each obstacle
                for robot in obstacles.values():
                    # Calculate distance to robot center
                    dist = np.hypot(robot.x - cell_x, robot.y - cell_y)
                    if dist < robot_radius + self.cell_length/2:  # Add cell radius for conservative estimate
                        grid[i, j] = 1
                        break


        self.neighbors = {
            (x, y): [
                (x + dx, y + dy)
                for dx in range(-2, 2) for dy in range(-2, 2)
                if (dx != 0 or dy != 0) and 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
            ]
            for x in range(self.grid_size) for y in range(self.grid_size)
        }
        return grid
    
    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates"""
        grid_x = int((x + self.field_length/2) / self.cell_length)
        grid_y = int((y + self.field_width/2) / self.cell_width)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size-1))
        grid_y = max(0, min(grid_y, self.grid_size-1))
        
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates"""
        x = -self.field_length/2 + grid_x * self.cell_length + self.cell_length/2
        y = -self.field_width/2 + grid_y * self.cell_width + self.cell_width/2
        return x, y
    
    def is_unoccupied(self, pos) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """

        row, col = self.continuous_to_grid(pos[0], pos[1])

        return self.occupancy_grid_map[row][col] == 1
    
    def create_grids(self, current_target: Point, opponents: dict[int, Robot], 
                        robot_radius: float, robot: Robot) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Create the grid representation of the environment and get start and goal cells.
        """
        
        current_grid = self.create_grid(opponents, robot_radius )
        start_cell = self.continuous_to_grid(
            robot.x, robot.y
        )  
        goal_cell = self.continuous_to_grid(
            current_target.x, current_target.y
        )  
        return current_grid, start_cell, goal_cell



    def detect_changes_allong_path(self, old_grid: np.ndarray, new_grid: np.ndarray, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Detect changes in the grid along the path between the previous and current state.

        Args:
            old_grid: Previous grid state.
            new_grid: Current grid state.
        """

        changes = []
        for cell in path:
            changes += self.detect_changes(old_grid, new_grid, cell)

        # print(f"Changes: {changes}")

        return changes

    def detect_changes(self, old_grid: np.ndarray, new_grid: np.ndarray, current_cell: Tuple[int, int] 
                       ) -> List[Tuple[int, int]]:
        """
        Detect important changes in the grid, considering only the neighborhood of the current cell.
        (analogy to "Scan graph for changed edge costs)
        """

        if old_grid is None:
            # If old_grid is None, assume all cells in the neighborhood have changed
            rows, cols = new_grid.shape
            return [(x, y) for x in range(rows) for y in range(cols)] 

        changes = []
        rows, cols = old_grid.shape

        # Get neighbors of the current cell, including the cell itself
        neighbors = self.get_neighbors(current_cell) + [current_cell]

        # Check for changes only in the neighborhood
        for cell in neighbors:
            x, y = cell
            if 0 <= x < rows and 0 <= y < cols:  # Ensure within bounds
                if old_grid[x, y] != new_grid[x, y]:
                    changes.append((x, y))

        return changes



    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self.neighbors.get(cell, []) #dict.get() method returns the value for the specified key if key is in dictionary.
        
    @staticmethod
    def visualize_grid( grid: np.ndarray, robot_pos: Tuple[int, int], 
                    target_pos: Tuple[int, int], path: List[Tuple[int, int]] = None):
        """
        Visualize the grid, robot position, target, and path, rotated 90° counterclockwise.

        Args:
            grid (np.ndarray): The grid representing the environment (0 for free, 1 for obstacles).
            robot_pos (Tuple[int, int]): The current position of the robot (x, y).
            target_pos (Tuple[int, int]): The target position (x, y).
            path (List[Tuple[int, int]], optional): The planned path as a list of (x, y) coordinates.
        """
        # Rotate the grid 90° counterclockwise
        rotated_grid = np.rot90(grid)

        # Transform coordinates to match the rotated grid
        grid_height, grid_width = rotated_grid.shape
        rotated_robot_pos = (grid_width - robot_pos[1] - 1, robot_pos[0])
        rotated_target_pos = (grid_width - target_pos[1] - 1, target_pos[0])
        rotated_path = [(grid_width - y - 1, x) for x, y in path] if path else []

        # Set up the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(rotated_grid, cmap="Greys", origin="upper")
        
        # Mark robot's position
        plt.scatter(rotated_robot_pos[1], rotated_robot_pos[0], color="blue", label="Robot", s=100)
        
        # Mark target's position
        plt.scatter(rotated_target_pos[1], rotated_target_pos[0], color="green", label="Target", s=100)
        
        # Mark the path
        if rotated_path:
            path_x = [cell[1] for cell in rotated_path]
            path_y = [cell[0] for cell in rotated_path]
            plt.plot(path_x, path_y, color="orange", label="Path", linewidth=2)
            plt.scatter(path_x, path_y, color="orange", s=20)  # Mark individual path cells
        
        # Add grid lines
        plt.grid(color="black", linestyle="--", linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid_width, 1), [])
        plt.yticks(np.arange(-0.5, grid_height, 1), [])
        plt.gca().set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        plt.gca().grid(which="minor", color="black", linestyle="--", linewidth=0.5)
        plt.gca().set_xticks(np.arange(0, grid_width, 1))
        plt.gca().set_yticks(np.arange(0, grid_height, 1))
        
        # Add labels and legend
        plt.legend(loc="upper right")
        plt.title("Grid Visualization")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.gca().invert_yaxis()  # Match the field's origin at the bottom-left corner
        plt.show()
