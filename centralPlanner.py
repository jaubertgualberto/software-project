import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from dStarAgent import DStarLiteAgent
from utils.ssl.Navigation import Navigation
from PathPlanning.utils.grid_converter import GridConverter




"""

To do:
    "Add parallel path planning for agents",
    "Implement error handling in step method",
    "Create monitoring for completed targets"

    "Create shared grid"
    "Handle case when one agent reach another agent target" ok

"""

class CentralPlanner:
    
    def __init__(self,
                agents: dict[int, DStarLiteAgent] = dict()):
        self.grid_size = 20
        
        self.agents = agents
        self.targets = []
        self.pursued_targets = []
        self.grid_converter = GridConverter(field_length=6.0, field_width=4.0, grid_size=self.grid_size)
        self.robot_radius = 0.06
        self.grid = None
        self.last_grid = None

    
    def add_target(self, target: Point):
        new_target = Point(target.x, target.y)
        self.targets.append((new_target, False))

    def get_agent_from_target(self, target: Point):
        for agent in self.agents.values():
            if agent.current_target == target:
                return agent

    def agent_reached_target(self, target: Point):
        try:
            agent_target = self.get_agent_from_target(target)
            agent_target.has_target = False
            self.targets.remove((target, True))
        except:
            print("Error: Agent not found")


    def num_targets_left(self):
        """
        Returns the number of targets that have not been pursued yet.
        """
        return len([target for target, is_pursued in self.targets if not is_pursued])

    def step(self, 
             robots_blue: List[Robot],
             teammates: dict[int, Robot] = dict(), 
             obstacles: dict[int, Robot] = []) -> List[Robot]:
        
        self.teammates = teammates.copy()

        remove_self = lambda robots, selfId: {id: robot for id, robot in robots.items() if id != selfId}
        # Create global shared grid
        # current_grid, start_cell, goal_cell = self.grid_converter.create_grids(current_target, self.opponents,  self.robot_radius, self.robot)
        


        # goal_cell = self.continuous_to_grid(current_target.x, current_target.y)
        if self.grid is not None:
            global_changes = self.grid_converter.detect_changes(self.last_grid, self.grid)
        else:
            global_changes = []

        my_actions = []
        for agent in self.agents.values():
            current_grid = self.grid_converter.create_grid(remove_self(obstacles, agent.id), self.robot_radius)

            agent.update(robots_blue[agent.id], remove_self(obstacles, agent.id), teammates)

            if agent.has_target:
                assigned_target = agent.current_target
            # Checks if there's any target left that the agent can pursue
            elif self.num_targets_left() > 0:
                # Pursue the target left
                assigned_target = self.assign_target(agent)
                

                # Reset standing point
                agent.set_standing_point(None)
            else:
                # No target left, stay in place
                assigned_target = None
                if agent.standing_point is None:
                    point = Point(agent.robot.x, agent.robot.y)
                    agent.set_standing_point(point)
            
            
            # if assigned_target is not None:
            #     agent_start_cell, goal_cell = self.grid_converter.create_grids(assigned_target, agent.robot)
            #     assert self.grid_converter.verify_target_alignment(assigned_target)
            
            #     agent.set_grids(current_grid, agent_start_cell, goal_cell)
            #     agent.set_global_changes(global_changes)

            action = agent.step(current_target=assigned_target)
            my_actions.append(action)


        self.last_grid = self.grid
        self.grid = current_grid

        return my_actions
    
    def assign_target(self, agent: DStarLiteAgent) -> Point:

        assigned_target = None

        if agent.current_target is None:
            available_targets = []
            available_target_indices = []
            
            for i, (target, is_pursued) in enumerate(self.targets):
                if not is_pursued:
                    available_targets.append(target)
                    available_target_indices.append(i)

            if available_targets:

                distances = [Navigation.distance_to_point(agent.robot, target) 
                           for target in available_targets]

                # Find the closest available target
                closest_idx = np.argmin(distances)
                target_idx = available_target_indices[closest_idx]
                
                # Assign this robot to the target
                assigned_target = self.targets[target_idx][0]
                self.targets[target_idx] = (assigned_target, True)
                agent.has_target = True
            else:
                # No available targets, stay in place
                agent.has_target = False
                assigned_target = Point(agent.robot.x, agent.robot.y)


        return assigned_target

