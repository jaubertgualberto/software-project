from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from PathPlanning.utils.grid_converter import GridConverter
from utils.Point import Point

from PathPlanning.utils.robot_movement import get_evasive_velocity

from rsoccer_gym.Entities import Robot
from PathPlanning.d_star_lite import DStarLite
from PathPlanning.utils.handle_target import TargetHelper

from PathPlanning.velocity_obstacles import RVO_update

colors = [  
    "RED",
    "WHITE",
    "BG_GREEN",
    "GREEN",
    "ORANGE",
    "BLUE",
    "YELLOW",
    "OLIVE",
    "PURPLE",
    "HAZEL",
    ]

class DStarLiteAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.robot_radius = 0.06
        self.grid_size = 27
        self.grid_converter = GridConverter(field_length=6.0, field_width=4.0, grid_size=self.grid_size)
        self.target_helper = TargetHelper(grid_converter=self.grid_converter)

        self.grid = None
        self.dstar = None
        self.previous_target = None
        self.render_path = []

        self.max_speed = 4.0    
        self.min_speed = 0.4

        self.has_target = False
        self.current_target = None
        self.color = colors[id]
        self.standing_point = None
        self.safety_radius = 0.4 

        self.path = []
        self.last_start  = []

        self.wayPoint = 0


        # Initialize the VelocityObstacle instance
        # self.velocity_obstacle = VelocityObstacle(
        #     robot_radius=self.robot_radius,
        #     max_speed=self.max_speed,
        #     min_speed=self.min_speed
        # )

    
    # def compute_velocity(self, robot: Robot, target: Point, targetVel):
    #     RVO_update


    def update(self,  
             self_robot: Robot, 
             opponents: dict[int, Robot] = dict(), 
             teammates: dict[int, Robot] = dict(),):
        """
        Update the agent with the current env state.

        Args:
            self_robot: Robot object.
            opponents: Dictionary of opponent robots {id: robot}.
            teammates: Dictionary of teammate robots {id: robot}.
        """
        
        self.robot = self_robot
        self.opponents = opponents.copy()
        self.teammates = teammates.copy()   


    def step(self, current_target: Point = None) -> Robot:

        self.current_target = current_target

        if current_target and self.has_target:
            self.decision( current_target)
            self.post_decision()

        else:
            self.wait()
            
        
        return Robot(id=self.id, yellow=self.yellow,
                    v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)


    def decision(self, current_target = None):
        
        # Create Grid, get start and goal cells
        current_grid, start_cell, goal_cell = self.grid_converter.create_grids(current_target, self.opponents,  self.robot_radius, self.robot)
        changes = len(self.grid_converter.detect_changes(self.grid, current_grid, start_cell)) > 0
  

        if self.dstar:
            changes = self.grid_converter.detect_changes_allong_path(self.dstar.grid, current_grid, self.path)
        else:
            changes = []


        if not self.dstar or self.target_helper.hasTargetChanged(current_target, self.previous_target):
            self.dstar = DStarLite(grid=current_grid, start=start_cell, goal=goal_cell, id=self.id)
            self.dstar.compute_shortest_path()
            self.path = self.dstar.reconstruct_path()
            self.wayPoint = 1 

            self.last_start = self.dstar.start

        # If any edge cost has changed
        elif (changes) or goal_cell not in self.path:
            # Update D* Lite K_m value

            print("Updating Path")

            self.dstar.Km +=  self.dstar.heuristic(self.dstar.start, self.last_start)

            self.dstar.update_grid(changes, current_grid)

            self.dstar.compute_shortest_path()
            self.path = self.dstar.reconstruct_path()

            self.last_start = self.dstar.start

            
        # Convert to continuous coordinates
        continuous_path = [
            self.grid_converter.grid_to_continuous(grid_x, grid_y)
            for grid_x, grid_y in self.path
        ]

        # Move towards next point
        next_point = self.target_helper.get_next_waypoint(self.path, self.wayPoint, self.grid_converter)
        if next_point:
            self.goTo(next_point)

            # Update waypoint if reached
            if self.target_helper.reachedWayPoint(next_point, self.robot):
                self.path.pop(0)
                
        
        self.dstar.start = self.grid_converter.continuous_to_grid(self.robot.x, self.robot.y)
        
        self.render_path = [Point(x, y) for x, y in continuous_path]
        self.previous_target = current_target
        self.grid = current_grid


    def post_decision(self):
        # # Adjust velocity to avoid dynamic obstacles
        # adjusted_velocity = self.velocity_obstacle.avoid_collision(
        #     self.next_vel, self.robot, self.opponents
        # )
        # if adjusted_velocity is not None:
        #     self.set_vel(adjusted_velocity)
        pass


    def goTo(self, point: Point):
        """
        Define linear and angular velocity to robot reach a target point while avoiding imediate collisions.

        Args: 
            point: Target point to reach.
        """

        target_velocity, target_angle_velocity = Navigation.goToPoint(
            self.robot, point
        )
        
        # If we're close to our target point, check for evasive action
        if Navigation.distance_to_point(self.robot, point) < 0.08:
            evasive_vel = get_evasive_velocity(self.opponents, self.robot, self.safety_radius, self.max_speed)
            if evasive_vel.x != 0 or evasive_vel.y != 0:
                self.set_vel(evasive_vel)
                return
                
        self.set_vel(target_velocity)
        self.set_angle_vel(target_angle_velocity)


    def wait(self):
        """ Stand still and avoid collisions """

        if self.standing_point is None:
            # Get evasive action if needed while standing still
            safe_vel = get_evasive_velocity(self.opponents, self.robot, self.safety_radius, self.max_speed)
            self.set_vel(safe_vel)
            self.set_angle_vel(0.0)
        else:
            self.goTo(self.standing_point)
   
   
    def set_standing_point(self, point: Point):
        self.standing_point = point



    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.render_path