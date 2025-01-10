from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
import numpy as np
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.RRT.rrt import RRT


class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.planned_path = [] 

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
        # print("Teste")

        return Robot( id=self.id, yellow=self.yellow,
                      v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    def decision(self, opponents: dict[int, Robot] = dict()):
        if len(self.targets) == 0:
            return
        
        # Max and min values for the random points (max and min field lenght )
        rand_area = [-3.1, 3.1]
        play_area = [-3.1, 3.1, -2.1, 2.1]

        print(f"Robots pos: {self.robot.x, self.robot.y}")

        # print("Planning path...")
        path = RRT([self.pos.x, self.pos.y], [self.targets[0].x, self.targets[0].y], 
                       opponents, rand_area=rand_area, play_area=play_area) \
                        .planning(animation=False)
        # print("Planning completed")

        if path and len(path) > 0:
            # Store full path
            self.planned_path = [
                Point(*points)
                for points in path
            ]
            # print("Path found")
            print("Path: ", self.planned_path)
            next_point = Point(path[0][0], path[0][1])

            target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
            # self.set_angle_vel(target_angle_velocity)
            # self.set_vel(target_velocity)
        else:
            # print("Invalid path!")
            self.planned_path = []
        return

    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.planned_path