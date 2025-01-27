from typing import Tuple, List
import numpy as np
from utils.Point import Point
from rsoccer_gym.Entities import Robot
import matplotlib.pyplot as plt



class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 24):
        """
        Initialize GridConverter with field dimensions and grid size
        
        The grid is created to maintain the field's aspect ratio while 
        keeping the total grid size close to the specified grid_size
        
        Args:
            field_length (float): Length of the soccer field
            field_width (float): Width of the soccer field
            grid_size (int): Approximate total grid size, defaults to 24
        """
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate grid sizes proportional to field dimensions
        aspect_ratio = field_length / field_width
        self.grid_size_x = grid_size
        self.grid_size_y = int(grid_size / aspect_ratio)


        # Calculate cell dimensions
        self.cell_length = field_length / self.grid_size_x
        self.cell_width = field_width / self.grid_size_y
        
        
        # Precompute neighbors for all grid cells
        self.neighbors = self.compute_neighbors()

    def compute_neighbors(self):
        """
        Precompute valid neighbors for each grid cell
        
        Returns:
            dict: A dictionary mapping each cell to its valid neighboring cells
        """
        return {
            (x, y): [
                (x + dx, y + dy)
                for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                if (dx != 0 or dy != 0) and 
                   0 <= x + dx < self.grid_size_x and 
                   0 <= y + dy < self.grid_size_y
            ]
            for x in range(self.grid_size_x) for y in range(self.grid_size_y)
        }


    def _get_cells_covered_by_robot(self, center_x: float, center_y: float, radius: float) -> List[Tuple[int, int]]:
        """
        Get all grid cells that intersect with the robot radius. 
        Assure that all cells that might overlap with the robot are considered as obstacles. 
        
        Args:
            center_x (float): X-coordinate of circle center
            center_y (float): Y-coordinate of circle center
            radius (float): Radius of the circle
        
        Returns:
            List[Tuple[int, int]]: List of grid cells covered by the circle
        """

        # get grid coordinates of the circle center
        grid_center_x, grid_center_y = self.continuous_to_grid(center_x, center_y)
        # get the number of cells that the circle radius spans
        cells_radius = int(np.ceil(radius / min(self.cell_length, self.cell_width))) + 1
        
        covered_cells = []
        for dx in range(-cells_radius, cells_radius + 1):
            for dy in range(-cells_radius, cells_radius + 1):
                grid_x = grid_center_x + dx
                grid_y = grid_center_y + dy
                

                # Get cell center in continuous coordinates
                cell_x, cell_y = self.grid_to_continuous(grid_x, grid_y)
                
                # Check if cell center is within robot radius + cell diagonal/2
                cell_diagonal = np.hypot(self.cell_length, self.cell_width) / 2
                if np.hypot(cell_x - center_x, cell_y - center_y) <= radius + cell_diagonal:
                    covered_cells.append((grid_x, grid_y))
                        
        return covered_cells

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
    

    def create_grid(self, obstacles: dict[int, Robot], robot_radius: float) -> np.ndarray:
        """
        Create grid representation with obstacles
        
        Args:
            obstacles (dict[int, Robot]): Dictionary of obstacle robots
            robot_radius (float): Radius of robots to consider as obstacles
        
        Returns:
            np.ndarray: Grid with obstacles marked
        """
        grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=int)
        
        for robot in obstacles.values():
            covered_cells = self._get_cells_covered_by_robot(robot.x, robot.y, robot_radius)
            for grid_x, grid_y in covered_cells:
                if 0 <= grid_x < self.grid_size_x and 0 <= grid_y < self.grid_size_y:
                    grid[grid_x, grid_y] = 1
        
        
        return grid





    def create_grids(self, current_target: Point, opponents: dict[int, Robot],
                    robot_radius: float, robot: Robot) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Create grid representation and return grid cells for start and goal
        
        Args:
            current_target (Point): Target point
            opponents (dict[int, Robot]): Dictionary of opponent robots
            robot_radius (float): Radius of robots
            robot (Robot): Current robot
        
        Returns:
            Tuple containing:
            - Occupancy grid
            - Start cell coordinates
            - Goal cell coordinates
        """
        # Create occupancy grid
        current_grid = self.create_grid(opponents, robot_radius)
        
        # Get start and goal cells
        start_cell = self.continuous_to_grid(robot.x, robot.y)
        goal_cell = self.continuous_to_grid(current_target.x, current_target.y)
        
        return current_grid, start_cell, goal_cell

    def is_unoccupied(self, pos, occupancy_grid_map) -> bool:
        """
        Check if a given position is unoccupied in the grid
        
        Args:
            pos (Tuple[float, float]): Position to check
            occupancy_grid_map (np.ndarray): Grid representing occupied cells
        
        Returns:
            bool: True if cell is unoccupied, False otherwise
        """
        row, col = self.continuous_to_grid(pos[0], pos[1])
        return occupancy_grid_map[row][col] == 0

    def detect_changes_allong_path(self, old_grid: np.ndarray, new_grid: np.ndarray, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Detect changes in the grid along the path between the previous and current state.

        Args:
            old_grid: Previous grid state.
            new_grid: Current grid state.
        """

        changes = []
        for cell in path:
            changes += self.detect_changes_neighbors(old_grid, new_grid, cell)

        # print(f"Changes: {changes}")

        return changes

    def detect_changes_neighbors(self, old_grid: np.ndarray, new_grid: np.ndarray, current_cell: Tuple[int, int] 
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

    def detect_changes(self, old_grid: np.ndarray, new_grid: np.ndarray) -> List[Tuple[int, int]]:
        if old_grid is None:
            # If old_grid is None, assume all cells in the neighborhood have changed
            rows, cols = new_grid.shape
            return [(x, y) for x in range(rows) for y in range(cols)] 

        changes = []
        rows, cols = old_grid.shape

        for x in range(rows):
            for y in range(cols):
                if old_grid[x, y] != new_grid[x, y]:
                    changes.append((x, y))

        return changes

    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self.neighbors.get(cell, []) #dict.get() method returns the value for the specified key if key is in dictionary.

    @staticmethod
    def visualize_grid(grid: np.ndarray, robot_pos: Tuple[int, int], 
                       target_pos: Tuple[int, int], path: List[Tuple[int, int]] = []):
        """
        Visualize the grid.

        Args:
            grid (np.ndarray): The grid representing the environment (0 for free, 1 for obstacles).
            robot_pos (Tuple[int, int]): The current position of the robot (x, y).
            target_pos (Tuple[int, int]): The target position (x, y).
            path (List[Tuple[int, int]], optional): The planned path as a list of (x, y) coordinates.
        """

        # Create a figure with an aspect ratio matching the grid
        plt.figure(figsize=(10, 10 * (grid.shape[0] / grid.shape[1])))
        
        # Create a masked array to visualize obstacles
        masked_grid = np.ma.masked_where(grid == 0, grid)
        
        # Plot the grid with obstacles
        plt.imshow(masked_grid, cmap='gist_gray', alpha=0.5, interpolation='nearest')
        plt.imshow(grid == 0, cmap='gray', alpha=0.2, interpolation='nearest')
        
        # Plot grid lines
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
        
        # Mark robot's position
        plt.scatter(robot_pos[1], robot_pos[0], color='green', s=100, label='Robot', zorder=10)
        
        # Mark target position
        plt.scatter(target_pos[1], target_pos[0], color='red', s=100, label='Target', zorder=10)
        
        # Plot path if provided
        # if path:
        #     path_x = [pos[1] for pos in path]
        #     path_y = [pos[0] for pos in path]
        #     plt.plot(path_x, path_y, color='orange', linewidth=2, label='Path')
        #     plt.scatter(path_x, path_y, color='orange', s=50)
        
        # Set up the plot
        plt.title('Grid Visualization')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        
        # Customize ticks to show cell boundaries
        plt.xticks(range(grid.shape[1]))
        plt.yticks(range(grid.shape[0]))
        
        # Add legend
        plt.legend()

        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()