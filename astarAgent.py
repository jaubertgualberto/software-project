
from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from PathPlanning.utils.grid_converter import GridConverter
import numpy as np
from typing import List
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.a_star import AStarPlanner



class AStarAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.09
        self.planned_path = []  
        self.grid_converter = GridConverter(field_length=9.0, field_width=6.0)

    def step(self, 
             self_robot : Robot, 
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

        return Robot( id=self.id, yellow=self.yellow,
                      v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    

    def decision(self, opponents: dict[int, Robot] = dict()):
        if len(self.targets) == 0:
            return
        
        a_star = AStarPlanner([point.x for point in opponents.values()], [point.y for point in opponents.values()], 0.08, self.robot_radius + 0.1 )
        path_x, path_y = a_star.planning(self.robot.x, self.robot.y, self.targets[0].x, self.targets[0].y)

        self.planned_path = [Point(x, y) for x, y in zip(path_x, path_y)]

        if self.planned_path and len(self.planned_path) > 0:
            
            
            self.planned_path = self.planned_path[::-1]
            

            # Use first waypoint for immediate navigation
            i = 0 if len(self.planned_path) == 1 else 1 if len(self.planned_path) == 2 else 2

            next_point = Point(self.planned_path[i][0], self.planned_path[i][1])
            
            # if(a_star.open_set):
            #     next_point = self.robot
            #     a_star.open_set = False

            target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
            self.set_vel(target_velocity)
            self.set_angle_vel(target_angle_velocity)


    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.planned_path