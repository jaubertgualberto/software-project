import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from dstarAgent import DStarLiteAgent
from utils.ssl.Navigation import Navigation




"""

To do:
    "Add parallel path planning for agents",
    "Implement error handling in step method",
    "Create monitoring for completed targets"

    "Handle case when one agent reach another agent target" ok

"""

class CentralPlanner:
    
    def __init__(self,
                agents: dict[int, DStarLiteAgent] = dict()):
        
        self.agents = agents
        # print("Agents: ", self.agents)
        self.targets = []
        self.pursued_targets = []
        # print(f"Pursued targets: {self.pursued_targets}")

    
    def add_target(self, target: Point):
        new_target = Point(target.x, target.y)
        self.targets.append((new_target, False))

    def get_agent_from_target(self, target: Point):
        for agent in self.agents.values():
            if agent.current_target == target:
                return agent

    def agent_reached_target(self, target: Point):
            
            agent_target = self.get_agent_from_target(target)
            agent_target.has_target = False
            self.targets.remove((target, True))


    def num_targets_left(self):
        """
        Returns the number of targets that have not been pursued yet.
        """
        return len([target for target, is_pursued in self.targets if not is_pursued])

    def step(self, 
             robots_blue: List[Robot],
             teammates: dict[int, Robot] = dict(), 
             obstacles: dict[int, Robot] = []) -> List[Robot]:
        
        # self.targets = targets.copy()
        self.teammates = teammates.copy()



        remove_self = lambda robots, selfId: {id: robot for id, robot in robots.items() if id != selfId}

        # print("Agents: ", len(self.agents))

        my_actions = []
        for agent in self.agents.values():
            agent.update(robots_blue[agent.id], remove_self(obstacles, agent.id), teammates)
            # assigned_target = self.assign_target(agent) if agent.has_target == False else agent.current_target

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
                # print(f"Agent {agent.color} has no target left.")
            
            action = agent.step(current_target=assigned_target)
            # print(f"Action 2: {action}")
            my_actions.append(action)

        # print(f"Returned actions: {my_actions}")
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
                # self.pursued_targets[target_idx] = True
                agent.has_target = True
                # print("Teste")
            else:
                # No available targets, stay in place
                agent.has_target = False
                assigned_target = Point(agent.robot.x, agent.robot.y)


        return assigned_target

