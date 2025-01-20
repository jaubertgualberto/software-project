from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from utils.grid_converter import GridConverter
import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.d_star_lite import DStarLite

import matplotlib.pyplot as plt



colors = [  
    "RED",
    "WHITE",
    "BG_GREEN",
    "GREEN",
    "ORANGE",
    "BLUE",
    "YELLOW",
    "OLIVE",
    "PURPLE",
    "HAZEL",
    ]

class DStarLiteAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.06
        self.grid_size = 27
        self.grid_converter = GridConverter(field_length=6.0, field_width=4.0, grid_size=self.grid_size)
        self.grid = None
        self.dstar = None
        self.previous_target = None
        self.render_path = []

        self.max_speed = 4.0    
        self.min_speed = 0.4

        self.has_target = False
        self.current_target = None
        self.color = colors[id]
        self.standing_point = None
        self.safety_radius = 0.4 

        self.path = []
        self.last_start  = []

        self.wayPoint = 0

        # Precompute neighbors for each cell (avoid recalculating every step)
        self.neighbors = {
            (x, y): [
                (x + dx, y + dy)
                for dx in range(-2, 2) for dy in range(-2, 2)
                if (dx != 0 or dy != 0) and 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
            ]
            for x in range(self.grid_size) for y in range(self.grid_size)
        }



    def visualize_grid(self, grid: np.ndarray, robot_pos: Tuple[int, int], 
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


    def get_evasive_velocity(self) -> Point:
        """Calculate evasive velocity when robots get too close while standing."""
        evasive_x, evasive_y = 0.0, 0.0
        
        for opponent in self.opponents.values():
            dx = self.robot.x - opponent.x
            dy = self.robot.y - opponent.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < self.safety_radius:
                force = (self.safety_radius - distance) / self.safety_radius
                # Normalize direction
                if distance > 0:
                    dx /= distance
                    dy /= distance
                else: 
                    angle = np.random.random() * 2 * np.pi
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                
                
                evasive_x += dx * force * self.max_speed * 0.5
                evasive_y += dy * force * self.max_speed * 0.5
        
        # If no evasion needed, return zero velocity
        if evasive_x == 0 and evasive_y == 0:
            return Point(0.0, 0.0)
            
        # Normalize and scale evasive velocity
        magnitude = np.sqrt(evasive_x*evasive_x + evasive_y*evasive_y)
        if magnitude > self.max_speed:
            evasive_x = (evasive_x / magnitude) * self.max_speed
            evasive_y = (evasive_y / magnitude) * self.max_speed
            
        return Point(evasive_x, evasive_y)

    def goTo(self, point: Point):
        target_velocity, target_angle_velocity = Navigation.goToPoint(
            self.robot, point
        )
        
        # If we're close to our target point, check for evasive action
        if Navigation.distance_to_point(self.robot, point) < 0.08:
            evasive_vel = self.get_evasive_velocity()
            if evasive_vel.x != 0 or evasive_vel.y != 0:
                self.set_vel(evasive_vel)
                return
                
        self.set_vel(target_velocity)
        self.set_angle_vel(target_angle_velocity)

    def wait(self):
        if self.standing_point is None:
            # Get evasive action if needed while standing still
            safe_vel = self.get_evasive_velocity()
            self.set_vel(safe_vel)
            self.set_angle_vel(0.0)
        else:
            self.goTo(self.standing_point)

    def update(self,  
             self_robot: Robot, 
             opponents: dict[int, Robot] = dict(), 
             teammates: dict[int, Robot] = dict(),):
        
        self.robot = self_robot
        self.opponents = opponents.copy()
        self.teammates = teammates.copy()   

    def step(self, current_target: Point = None) -> Robot:

        self.current_target = current_target

        if current_target and self.has_target:
            self.decision( current_target)
        else:
            self.wait()
            
        self.post_decision()
        
        return Robot(id=self.id, yellow=self.yellow,
                    v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    def set_standing_point(self, point: Point):
        self.standing_point = point

    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self.neighbors.get(cell, [])

    def is_valid_coordinates(self, u: Tuple[int, int]) -> bool:
        """
        Check if coordinates are within grid bounds
        """
        return (0 <= u[0] < self.rows and 
                0 <= u[1] < self.cols)



    def detect_changes_allong_path(self, old_grid: np.ndarray, new_grid: np.ndarray, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Detect changes in the grid along the path between the previous and current state.

        Args:
            old_grid: Previous grid state.
            new_grid: Current grid state.
        """

        changes = []
        for cell in path:
            changes += self.detect_important_changes(old_grid, new_grid, cell)

        # print(f"Changes: {changes}")

        return changes

    def detect_important_changes(self, old_grid: np.ndarray, new_grid: np.ndarray, current_cell: Tuple[int, int]) -> List[Tuple[int, int]]:
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


    def decision(self, current_target = None):
        if not current_target:
            self.wait()
            return 

        
        # Create Grid, get start and goal cells
        current_grid, start_cell, goal_cell = self.create_grids(current_target, self.opponents)
        impoartant_changes = len(self.detect_important_changes(self.grid, current_grid, start_cell)) > 0
  

        if self.dstar:
            # changes = self.detect_changes(self.dstar.grid, current_grid)
            changes = self.detect_changes_allong_path(self.dstar.grid, current_grid, self.path)
        else:
            changes = []



        if not self.dstar or self.hasTargetChanged(current_target):
            self.dstar = DStarLite(grid=current_grid, start=start_cell, goal=goal_cell, id=self.id)
            self.dstar.compute_shortest_path()
            self.path = self.dstar.reconstruct_path()
            self.wayPoint = 1 

            self.last_start = self.dstar.start

            # self.visualize_grid(self.dstar.grid, self.dstar.start, self.dstar.goal, self.path)
            # self.visualize_grid(current_grid, start_cell, goal_cell, self.path)


        
        
        # If any edge cost has changed
        elif (impoartant_changes) or goal_cell not in self.path:
            # Update D* Lite K_m value

            print("Updating Path")

            # self.dstar.Km +=  self.dstar.heuristic(self.dstar.start, self.last_start)

            self.dstar.update_grid(changes, current_grid)

            self.dstar.compute_shortest_path()
            self.path = self.dstar.reconstruct_path()

            self.last_start = self.dstar.start

            # self.visualize_grid(self.dstar.grid, self.dstar.start, self.dstar.goal, self.path)
            # self.visualize_grid(current_grid, start_cell, goal_cell, self.path)

            
        # Convert to continuous coordinates
        continuous_path = [
            self.grid_converter.grid_to_continuous(grid_x, grid_y)
            for grid_x, grid_y in self.path
        ]

        # Move towards next point
        next_point = self.get_next_waypoint()
        if next_point:
            self.goTo(next_point)

            # Update waypoint if reached
            if self.reachedWayPoint(next_point):
                self.path.pop(0)
                
        
        self.dstar.start = self.grid_converter.continuous_to_grid(self.robot.x, self.robot.y)
        
        self.render_path = [Point(x, y) for x, y in continuous_path]
        self.previous_target = current_target
        self.grid = current_grid





    def reachedWayPoint(self, point: Point):
        return Navigation.distance_to_point(self.robot, point) < 0.1

    def hasTargetChanged(self, current_target):
        return  self.previous_target is None \
        or abs(current_target.x - self.previous_target.x) > 1e-6 \
        or abs(current_target.y - self.previous_target.y) > 1e-6
        
    def get_next_waypoint(self):
        if not self.path or self.wayPoint >= len(self.path):
            return None
        # Directly get the waypoint
        return Point(*self.grid_converter.grid_to_continuous(*self.path[self.wayPoint]))

    def create_grids(self, current_target, opponents):
        """
        Create the grid representation of the environment and get start and goal cells.
        """
        current_grid = self.grid_converter.create_grid(opponents, self.robot_radius )

        start_cell = self.grid_converter.continuous_to_grid(
            self.robot.x, self.robot.y
        )  
        goal_cell = self.grid_converter.continuous_to_grid(
            current_target.x, current_target.y
        )  
        return current_grid, start_cell, goal_cell


    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.render_path