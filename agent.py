from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
import numpy as np
from utils.Point import Point
# from RRTstar.rrt_star import RRTStar, Node

# def point_to_node(point: Point):
#     return Node(point.x, point.y)

class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)

    def decision(self, point):
        if len(self.targets) == 0:
            return

        if point is not None:
            # print("Teste")
            target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, target=self.targets[0])
            point_velocity, point_angle_velocity = Navigation.goToPoint(self.robot, target=point)

            # self.set_angle_vel(target_angle_velocity)
            # self.set_vel(target_velocity)

            # print(f"point pos {point.x}, {point.y}")
            # print(f"robot pos {self.pos.x}, {self.pos.y}")
            # print(f"Distancia target {self.pos.dist_to(self.targets[0])}")
            # print(f"Distancia point {self.pos.dist_to(point)}")
            # print("")


        return

    def post_decision(self):
        pass

