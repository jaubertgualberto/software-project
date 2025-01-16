import numpy as np
from typing import List, Set, Tuple
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from dstarAgent import DStarLiteAgent
from utils.ssl.Navigation import Navigation


class CentralPlanner:
    
    def __init__(self,
                agents: dict[int, DStarLiteAgent] = dict()):
        
        self.agents = agents
        # print("Agents: ", self.agents)
        self.targets = []
        self.pursued_targets = []
        # print(f"Pursued targets: {self.pursued_targets}")

    def add_agent(self, agent:  DStarLiteAgent):
        # print("Agent added: ", agent)
        print("Teste")
        self.agents[len(self.agents)] = agent
    
    def add_target(self, target: Point):
        # print("Target added: ", target)
        self.targets.append(target)
        self.pursued_targets.append(False)

    


    def agent_reached_target(self, agent: DStarLiteAgent):
            # print(f"1- Targets: {self.targets}")
            self.pursued_targets.pop(self.targets.index(agent.current_target))
            agent.has_target = False

    def step(self, 
             robots_blue: List[Robot],
             teammates: dict[int, Robot] = dict(), 
             targets: list[Point] = [], 
             obstacles: dict[int, Robot] = []) -> List[Robot]:
        
        self.targets = targets.copy()
        self.teammates = teammates.copy()
        
        # print("Received targets: ", self.targets)
        # print("Targets: ", self.targets)

        remove_self = lambda robots, selfId: {id: robot for id, robot in robots.items() if id != selfId}

        my_actions = []
        for agent in self.agents.values():
            agent.update(robots_blue[agent.id], remove_self(obstacles, agent.id), teammates)
            assigned_target = self.assign_targets(agent) if agent.has_target == False else agent.current_target
            # print("Assigned target: ", assigned_target)
            
            action = agent.step(current_target=assigned_target)
            # print(f"Action 2: {action}")
            my_actions.append(action)

        # print(f"Returned actions: {my_actions}")
        return my_actions
    
    def assign_targets(self, agent: DStarLiteAgent) -> Point:

        assigned_target = None

        if agent.current_target is None:
            available_targets = []
            available_target_indices = []
            
            for i, (target, is_pursued) in enumerate(zip(self.targets, self.pursued_targets)):
                if not is_pursued:
                    available_targets.append(target)
                    available_target_indices.append(i)

            print("Available targets: ", self.pursued_targets)

            if available_targets:
                # Calculate distances to all available targets
                # print("Available targets: ", available_targets)
                # print("Agent robot position: ", agent.robot.x, agent.robot.y)

                distances = [Navigation.distance_to_point(agent.robot, target) 
                           for target in available_targets]
                

                # Find the closest available target
                closest_idx = np.argmin(distances)
                target_idx = available_target_indices[closest_idx]
                
                # Assign this robot to the target
                assigned_target = self.targets[target_idx]
                self.pursued_targets[target_idx] = True
                agent.has_target = True
                # print("Teste")
            else:
                # No available targets, stay in place
                agent.has_target = False
                assigned_target = Point(agent.robot.x, agent.robot.y)


        return assigned_target

