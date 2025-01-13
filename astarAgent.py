
from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from utils.grid_converter import GridConverter
from typing import List
from utils.Point import Point
from rsoccer_gym.Entities import Robot
from PathPlanning.AStar.a_star import AStarPlanner


class AStarAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.09
        self.planned_path = []  
        self.grid_converter = GridConverter(field_length=9.0, field_width=6.0)
        self.render_path = []

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
        
        a_star = AStarPlanner([point.x for point in opponents.values()], [point.y for point in opponents.values()], 0.1, self.robot_radius + 0.21 )
        path_x, path_y = a_star.planning(self.robot.x, self.robot.y, self.targets[0].x, self.targets[0].y)


        self.planned_path = [Point(x, y) for x, y in zip(path_x, path_y)]
        self.render_path = self.planned_path

        if self.planned_path and len(self.planned_path) > 0:
            
            
            self.planned_path = self.planned_path[::-1]
            

            if len(self.planned_path) > 0:
                # Use first waypoint for immediate navigation
                i = 0 if len(self.planned_path) == 1 else 1 if len(self.planned_path) == 2 else 2

                next_point = Point(self.planned_path[i][0], self.planned_path[i][1])

                # print(f"Next point: {next_point}, Target point: {self.targets[0]}, robot pos: {self.robot.x, self.robot.y}")


                target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
                self.set_vel(target_velocity)
                self.set_angle_vel(target_angle_velocity)

        else:
            # print("Invalid path!")
            self.planned_path = []
        return

    def post_decision(self):
        pass

    def get_render_path(self):
        """Return the current planned path for rendering"""
        return self.render_path