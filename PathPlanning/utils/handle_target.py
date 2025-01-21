import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from utils.Point import Point
from utils.ssl.Navigation import Navigation
from utils.Point import Point
from PathPlanning.utils.grid_converter import GridConverter
from rsoccer_gym.Entities import Robot

    
class TargetHelper:
    def __init__(self, grid_converter: GridConverter):
        self.grid_converter = grid_converter

    def get_next_waypoint(self, path: List[Tuple[int, int]], wayPoint: int, grid_converter: GridConverter) -> Point:
        """
        Get next way point that robot should reach.

        Args:
            path (List[Tuple[int, int]]): The planned path as a list of (x, y) coordinates.
            wayPoint (int): The index of the current way point.
            grid_converter (GridConverter): Grid converter object.

        Returns:
            Point: Next way point that robot should reach in continuous coordinates.
        """

        if not path or wayPoint >= len(path):
            return None
        # Directly get the waypoint
        return Point(*grid_converter.grid_to_continuous(*path[wayPoint]))

    @staticmethod
    def reachedWayPoint(point: Point, robot: Robot) -> bool:
        """
        Indicates whether the robot has reached the given point.

        Args:
            point (Point): Target point
            robot (Robot): Robot object

        Returns:
            True if the robot reached the point, false otherwise
        """
        return Navigation.distance_to_point(robot, point) < 0.1
    
    @staticmethod
    def hasTargetChanged(current_target: Point, previous_target: Point) -> bool:
        """
        Indicates whether the target has changed.

        Args:
            current_target (Point): Current target point
            previos_target (Point): Previos target point

        Returns:
            True if the target has changed, False otherwise.
        """
        return previous_target is None \
        or abs(current_target.x - previous_target.x) > 1e-6 \
        or abs(current_target.y - previous_target.y) > 1e-6
