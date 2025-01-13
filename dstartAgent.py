from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from utils.grid_converter import GridConverter
import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.DStarLite.d_star_lite import DStarLite, Node

class DStarLiteAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.15
        self.planned_path = []
        self.grid_size = 35
        self.grid_converter = GridConverter(field_length=8.0, field_width=5.0, grid_size=self.grid_size)
        self.dstar = None
        self.previous_obstacle_cells: Set[Tuple[int, int]] = set()
        self.current_target = None
        self.render_path = []
        
    def step(self, 
             self_robot: Robot, 
             opponents: dict[int, Robot] = dict(), 
             teammates: dict[int, Robot] = dict(), 
             targets: list[Point] = [], 
             keep_targets=False,
             path=None) -> Robot:
        
        self.targets = targets.copy()
        self.robot = self_robot
        self.opponents = opponents.copy()
        self.teammates = teammates.copy()

        self.decision(opponents)
        self.post_decision()

        return Robot(id=self.id, yellow=self.yellow,
                    v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    def get_obstacle_cells(self, grid: np.ndarray) -> Set[Tuple[int, int]]:
        """Get set of obstacle cell coordinates"""
        return {(i, j) for i in range(grid.shape[0]) 
                for j in range(grid.shape[1]) if grid[i, j]}

    def get_changed_cells(self, current_cells: Set[Tuple[int, int]]) -> List[Node]:
        """Get list of cells that changed state since last update"""
        # Find cells that changed state (free->obstacle or obstacle->free)
        changed = current_cells.symmetric_difference(self.previous_obstacle_cells)
        return [Node(x=x, y=y) for x, y in changed]

    def decision(self, opponents: dict[int, Robot] = dict()):
        if len(self.targets) == 0:
            return

        current_target = self.targets[0]

        # Create current obstacle grid
        current_grid = self.grid_converter.create_grid(opponents, self.robot_radius)
        current_obstacle_cells = self.get_obstacle_cells(current_grid)
        
        # Convert positions to grid coordinates
        start_grid_x, start_grid_y = self.grid_converter.continuous_to_grid(
            self.robot.x, self.robot.y)
        goal_grid_x, goal_grid_y = self.grid_converter.continuous_to_grid(
            current_target.x, current_target.y)
        try:     
            # Check if target has changed
            target_changed = (self.current_target is None or 
                            abs(current_target.x - self.current_target.x) > 1e-6 or 
                            abs(current_target.y - self.current_target.y) > 1e-6)
            
            # Reinitialize if target has changes or if D* Lite is not initialized
            if self.dstar is None or target_changed:
                ox, oy = zip(*current_obstacle_cells) if current_obstacle_cells else ([], [])
                self.dstar = DStarLite(list(ox), list(oy))
                self.dstar.x_max = self.grid_size
                self.dstar.y_max = self.grid_size
                self.current_target = current_target

            else:
                # Update changed cells for replanning
                changed_cells = self.get_changed_cells(current_obstacle_cells)
                if changed_cells:
                    # Create spoofed obstacles for cells that changed
                    spoofed_ox = [[node.x] for node in changed_cells]
                    spoofed_oy = [[node.y] for node in changed_cells]
                else:
                    spoofed_ox, spoofed_oy = [], []

            # Run D* Lite planning
            found, path_x, path_y = self.dstar.main(
                Node(x=start_grid_x, y=start_grid_y),
                Node(x=goal_grid_x, y=goal_grid_y),
                spoofed_ox=spoofed_ox if not target_changed else [],
                spoofed_oy=spoofed_oy if not target_changed else []
            )

            if found and len(path_x) > 0:
                # Convert path to continuous coordinates
                continuous_path = []
                for grid_x, grid_y in zip(path_x, path_y):
                    x, y = self.grid_converter.grid_to_continuous(
                        int(grid_x), int(grid_y))
                    continuous_path.append((x, y))
                
                self.planned_path = continuous_path
                self.render_path = [Point(x, y) for x, y in continuous_path]
                

                if len(continuous_path) > 1:
                    next_point = Point(continuous_path[1][0], continuous_path[1][1])
                else:
                    next_point = current_target
                
                target_velocity, target_angle_velocity = Navigation.goToPoint(
                    self.robot, next_point)
                self.set_vel(target_velocity)
                self.set_angle_vel(target_angle_velocity)
        
        except Exception as e:
            print(f"Planning error: {e}")
            ox, oy = zip(*current_obstacle_cells) if current_obstacle_cells else ([], [])
            self.dstar = DStarLite(list(ox), list(oy))
            self.dstar.x_max = self.grid_size
            self.dstar.y_max = self.grid_size
            self.current_target = current_target

            target_velocity, target_angle_velocity = Navigation.goToPoint(
                    self.robot, self.robot)  
            self.set_vel(target_velocity)
            self.set_angle_vel(target_angle_velocity)
        
            
        # Store current obstacle cells for next iteration
        self.previous_obstacle_cells = current_obstacle_cells
        
    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.render_path