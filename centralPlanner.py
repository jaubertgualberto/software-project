import numpy as np
from typing import List
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from agent import MainAgent
from utils.ssl.Navigation import Navigation



class CentralPlanner:
    
    def __init__(self,
                agents: dict[int, MainAgent] = dict()):
        """
        Central Planner Class design to manage multiples agents and targets.
        """
        self.agents = agents
        self.targets = []
        self.pursued_targets = []
        self.robot_radius = 0.06

    
    def add_target(self, target: Point):
        """
        Add new target for central planner.
        Args:
            Target point: Point
        """
        new_target = Point(target.x, target.y)
        self.targets.append((new_target, False))

    def _get_agent_from_target(self, target: Point):
        """
        Returns the agent pursuing the target.

        Args:
            Target point: Point
        """
        
        for agent in self.agents.values():
            if agent.current_target == target:
                return agent

    def agent_reached_target(self, target: Point):
        """
        Handles when an agent reaches the target. 
        Args:
            Target point: Point
        """
        try:
            agent_target = self._get_agent_from_target(target)
            agent_target.has_target = False
            self.targets.remove((target, True))
        except:
            # ToDo: Implement this. 
            pass


    def _num_targets_left(self):
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

        my_actions = []
        for agent in self.agents.values():
            agent.update(robots_blue[agent.id], remove_self(obstacles, agent.id), teammates)

            if agent.has_target:
                assigned_target = agent.current_target
            # Checks if there's any target left that the agent can pursue
            elif self._num_targets_left() > 0:
                # Pursue the target left
                assigned_target = self._assign_target(agent)

                # Reset standing point
                agent.set_standing_point(None)
            else:
                # No target left, stay in place
                assigned_target = None
                if agent.standing_point is None:
                    point = Point(agent.robot.x, agent.robot.y)
                    agent.set_standing_point(point)
            
            action = agent.step(current_target=assigned_target)
            my_actions.append(action)

        return my_actions
    
    def _assign_target(self, agent: MainAgent) -> Point:
        """
        Assign a target to agent based on the closest available target.

        Args:
            agent: MainAgent object

        Returns:
            assigned_target: Point object
        """
        assigned_target = None

        if agent.current_target is None:
            available_targets = []
            available_target_indices = []
            
            for i, (target, is_pursued) in enumerate(self.targets):
                if not is_pursued:
                    available_targets.append(target)
                    available_target_indices.append(i)

            if available_targets:
                # Compute distances from agents to available targets
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

