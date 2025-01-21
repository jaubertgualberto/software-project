from typing import Tuple, List
import numpy as np
from utils.Point import Point
from rsoccer_gym.Entities import Robot
import matplotlib.pyplot as plt


class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 24):
        """Initialize GridConverter with field dimensions and grid size"""
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate cell dimensions
        self.cell_length = field_length / grid_size
        self.cell_width = field_width / grid_size
        
        # Will store the grid offset to align target with cell center
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.neighbors = {}
    
    def calculate_grid_offset(self, target_x: float, target_y: float):
        """
        Calculate offset needed to align target with cell center
        """
        # Find what fraction of a cell the target is offset from cell centers
        target_cell_x = (target_x + self.field_length/2) / self.cell_length
        target_cell_y = (target_y + self.field_width/2) / self.cell_width
        
        # Calculate the fractional offset from cell center
        frac_x = target_cell_x - np.floor(target_cell_x) - 0.5
        frac_y = target_cell_y - np.floor(target_cell_y) - 0.5
        
        # Convert to actual distance offset
        self.offset_x = -frac_x * self.cell_length
        self.offset_y = -frac_y * self.cell_width

    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates with target-aligned grid"""
        # Apply offset before conversion
        x_shifted = x + self.offset_x
        y_shifted = y + self.offset_y
        
        grid_x = int((x_shifted + self.field_length/2) / self.cell_length)
        grid_y = int((y_shifted + self.field_width/2) / self.cell_width)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size-1))
        grid_y = max(0, min(grid_y, self.grid_size-1))
        
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates with target-aligned grid"""
        x = -self.field_length/2 + grid_x * self.cell_length + self.cell_length/2
        y = -self.field_width/2 + grid_y * self.cell_width + self.cell_width/2
        
        # Remove offset for final coordinates
        return x - self.offset_x, y - self.offset_y

    def get_cells_covered_by_circle(self, center_x: float, center_y: float, radius: float) -> List[Tuple[int, int]]:
        """Get all grid cells that intersect with a circle"""
        grid_center_x, grid_center_y = self.continuous_to_grid(center_x, center_y)
        cells_radius = int(np.ceil(radius / min(self.cell_length, self.cell_width))) + 1
        
        covered_cells = []
        for dx in range(-cells_radius, cells_radius + 1):
            for dy in range(-cells_radius, cells_radius + 1):
                grid_x = grid_center_x + dx
                grid_y = grid_center_y + dy
                
                if not (0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size):
                    continue
                
                # Get cell center in continuous coordinates
                cell_x, cell_y = self.grid_to_continuous(grid_x, grid_y)
                
                # Check if cell center is within robot radius + cell diagonal/2
                cell_diagonal = np.hypot(self.cell_length, self.cell_width) / 2
                if np.hypot(cell_x - center_x, cell_y - center_y) <= radius + cell_diagonal:
                    covered_cells.append((grid_x, grid_y))
                        
        return covered_cells

    def create_grid(self, obstacles: dict[int, Robot], robot_radius: float) -> np.ndarray:
        """Create grid representation with obstacles"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for robot in obstacles.values():
            covered_cells = self.get_cells_covered_by_circle(robot.x, robot.y, robot_radius)
            for grid_x, grid_y in covered_cells:
                grid[grid_x, grid_y] = 1
                
        return grid

    def create_grids(self, current_target: Point, opponents: dict   [int, Robot],
                    robot_radius: float, robot: Robot) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Create grid representation and return grid cells for start and goal,
        with grid aligned so target is at cell center
        """
        # Calculate grid offset to align target with cell center
        self.calculate_grid_offset(current_target.x, current_target.y)
        
        # Create occupancy grid
        # current_grid = self.create_grid(opponents, robot_radius)
        
        # Get start and goal cells
        start_cell = self.continuous_to_grid(robot.x, robot.y)
        goal_cell = self.continuous_to_grid(current_target.x, current_target.y)
        
        # Update neighbors dict
        self.neighbors = {
            (x, y): [
                (x + dx, y + dy)
                for dx in range(-1, 1) for dy in range(-1, 1)
                if (dx != 0 or dy != 0) and 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
            ]
            for x in range(self.grid_size) for y in range(self.grid_size)
        }
        
        return current_grid, start_cell, goal_cell

    def verify_target_alignment(self, target: Point) -> bool:
        """
        Verify that target point aligns with cell center
        (useful for debugging)
        """
        grid_x, grid_y = self.continuous_to_grid(target.x, target.y)
        center_x, center_y = self.grid_to_continuous(grid_x, grid_y)
        error = np.hypot(center_x - target.x, center_y - target.y)
        return error < 1e-10  # Allow for floating point error
    
    def is_unoccupied(self, pos) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """

        row, col = self.continuous_to_grid(pos[0], pos[1])

        return self.occupancy_grid_map[row][col] == 1
    
    # def create_grids(self, current_target: Point, opponents: dict[int, Robot], 
    #                     robot_radius: float, robot: Robot) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    #     """
    #     Create the grid representation of the environment and get start and goal cells.
    #     """
        
    #     current_grid = self.create_grid(opponents, robot_radius )
    #     start_cell = self.continuous_to_grid(
    #         robot.x, robot.y
    #     )  
    #     goal_cell = self.continuous_to_grid(
    #         current_target.x, current_target.y
    #     )  
    #     return current_grid, start_cell, goal_cell



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
    def visualize_grid( grid: np.ndarray, robot_pos: Tuple[int, int], 
                    target_pos: Tuple[int, int], path: List[Tuple[int, int]] = []):
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
