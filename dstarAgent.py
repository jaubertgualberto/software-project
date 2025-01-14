from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from utils.grid_converter import GridConverter
import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.DStarLite.d_star_lite import DStarLite


class DStarLiteAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.06
        self.planned_path = []
        self.grid_size = 35
        self.grid_converter = GridConverter(field_length=8.0, field_width=5.0, grid_size=self.grid_size)
        self.grid = None
        self.dstar = None
        self.previous_obstacle_cells: Set[Tuple[int, int]] = set()
        self.current_target = None
        self.render_path = []
        self.replan_threshold = 3
        self.update_counter = 0  
        
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

    def detect_changes(self, old_grid: np.ndarray, new_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect changes in the grid between the previous and current state.

        Args:
            old_grid: Previous grid state.
            new_grid: Current grid state.

        Returns:
            List of cells that have changed.
        """
        if old_grid is None:
            # If old_grid is None, assume all cells in new_grid have changed
            rows, cols = new_grid.shape
            return [(x, y) for x in range(rows) for y in range(cols)]
        
        changes = []
        rows, cols = old_grid.shape  # Get grid dimensions dynamically
        for x in range(rows):
            for y in range(cols):
                if old_grid[x, y] != new_grid[x, y]:
                    changes.append((x, y))
        return changes

    def decision(self, opponents: dict[int, Robot] = dict()):
        if len(self.targets) == 0:
            return

        current_target = self.targets[0]
        current_grid, start_grid, goal_grid = self.create_grids(current_target, opponents)
        target_changed = self.hasTargetChanged(current_target)

        try:
            # Only replan if target changed or every 5 updates
            self.update_counter += 1
            needs_replan = (self.dstar is None or 
                          target_changed or 
                          self.update_counter >= 5)

            if needs_replan:
                self.update_counter = 0
                self.dstar = DStarLite(grid=current_grid, start=start_grid, goal=goal_grid)
                self.dstar.compute_shortest_path()
                
                # If goal is blocked, just wait /:
                if not self.dstar.is_valid(goal_grid):
                    self.goTo(self.robot)

            path = self.dstar.reconstruct_path()
            
            # Convert to continuous coordinates
            continuous_path = [
                self.grid_converter.grid_to_continuous(grid_x, grid_y)
                for grid_x, grid_y in path
            ]

            self.planned_path = continuous_path
            self.render_path = [Point(x, y) for x, y in continuous_path]
            
            # Move towards next point
            if len(continuous_path) > 1:
                next_point = Point(continuous_path[1][0], continuous_path[1][1])
            else:
                next_point = current_target

            self.goTo(next_point)

        except ValueError:
            self.goTo(self.robot)

        self.current_target = current_target
        self.grid = current_grid

    def hasTargetChanged(self, current_target):
        return  self.current_target is None \
        or abs(current_target.x - self.current_target.x) > 1e-6 \
        or abs(current_target.y - self.current_target.y) > 1e-6
        
    def create_grids(self, current_target, opponents):
        current_grid = self.grid_converter.create_grid(opponents, self.robot_radius + 0.1)

        start_grid = self.grid_converter.continuous_to_grid(
            self.robot.x, self.robot.y
        )
        goal_grid = self.grid_converter.continuous_to_grid(
            current_target.x, current_target.y
        )
        return current_grid, start_grid, goal_grid

    def goTo(self, point: Point):
            target_velocity, target_angle_velocity = Navigation.goToPoint(
                self.robot, point
            )
            self.set_vel(target_velocity)
            self.set_angle_vel(target_angle_velocity)

    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.render_path