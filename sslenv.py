import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
from utils.Point import Point
from utils.FixedQueue import FixedQueue
from utils.ssl.small_field import SSLHRenderField
from agent import MainAgent

from random_agent import RandomAgent
import random
import pygame
from utils.CLI import Difficulty
from typing import Dict, List, Optional
from rsoccer_gym.Render import COLORS, SSLRenderField, SSLRobot
from rsoccer_gym.Render import Ball as BallRender
from rsoccer_gym.Simulators.rsim import RSimSSL

from centralPlanner import CentralPlanner


colors = {    
    0  : (151, 21, 0),
    1  : (220, 220, 220),
    2  : (0, 214, 190),
    3  : (0, 128, 0),
    4  : (253, 106, 2),
    5  : (0, 64, 255),
    6  : (255, 223, 0),
    7  : (128, 128, 0),
    8  : (102, 51, 153),
    9  : (220, 0, 220),
    10 : (250,165,112)
    }




class SSLExampleEnv(SSLBaseEnv):
    def __init__(self, render_mode="human", difficulty=Difficulty.EASY):
        field = 2   # 1: SSL Div B    2: SSL Software challenge
        super().__init__(
            field_type=field, 
            n_robots_blue=11,
            n_robots_yellow=11, 
            time_step=0.025,
            render_mode=render_mode)
        
        self.DYNAMIC_OBSTACLES, self.max_targets, self.max_rounds = Difficulty.parse(difficulty)

        n_obs = 4 # Ball x,y and Robot x, y
        self.action_space = Box(low=-1, high=1, shape=(2, ))
        self.observation_space = Box(low=-self.field.length/2,\
            high=self.field.length/2,shape=(n_obs, ))
        
        self.targets = []
        self.pursued_targets = []

        self.min_dist = 0.18
        self.all_points = FixedQueue(max(4, self.max_targets))
        self.robots_paths = [FixedQueue(40) for i in range(11)]

        self.rounds = self.max_rounds  ## because of the first round
        self.targets_per_round = 1

        self.my_agents = {0: MainAgent(0, False)}
        
        self.blue_agents = {i: RandomAgent(i, False) for i in range(1, 11)}
        self.yellow_agents = {i: RandomAgent(i, True) for i in range(0, 11)}

        self.gen_target_prob = 0.003

        self.central_planner = CentralPlanner(self.my_agents)

        if field == 2:
            self.field_renderer = SSLHRenderField()
            self.window_size = self.field_renderer.window_size
            
        goal_depth = 0.12  #at the goal center
        goal_length = 0.37
        self.goal_coordinates = [(-self.field.length/2-goal_depth, goal_length), 
                            (-self.field.length/2-goal_depth, -goal_length),
                            (self.field.length/2+goal_depth, goal_length),
                            (self.field.length/2+goal_depth, -goal_length),
                            (self.field.length/2, 0), (-self.field.length/2, 0),
                            (self.field.length/2, goal_length/2), (-self.field.length/2, goal_length/2),
                            (self.field.length/2, -goal_length/2), (-self.field.length/2, -goal_length/2),
                            ]
        
        
    def _frame_to_observations(self):
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array([ball.x, ball.y, robot.x, robot.y])

    def _get_commands(self, actions):

        obstacles = {id: robot for id, robot in self.frame.robots_blue.items()}
        for i in range(0, self.n_robots_yellow):
            obstacles[i + self.n_robots_blue] = self.frame.robots_yellow[i]
        
        # for coord in self.goal_coordinates:
        #     obstacles[len(obstacles)] = Robot(x=coord[0], y=coord[1])

        teammates = {id: self.frame.robots_blue[id] for id in self.my_agents.keys()}
        


        remove_self = lambda robots, selfId: {id: robot for id, robot in robots.items() if id != selfId}


        # print("Sent targets: ", self.targets)
        myActions = self.central_planner.step(self.frame.robots_blue, teammates, obstacles=obstacles)


        others_actions = []
        if self.DYNAMIC_OBSTACLES:
            for i in self.blue_agents.keys():
                random_target = []
                if random.uniform(0.0, 1.0) < self.gen_target_prob:
                    random_target.append(Point(x=self.x(), y=self.y()))
                    
                others_actions.append(self.blue_agents[i].step(self.frame.robots_blue[i], obstacles, dict(), random_target, True))

            for i in self.yellow_agents.keys():
                random_target = []
                if random.uniform(0.0, 1.0) < self.gen_target_prob:
                    random_target.append(Point(x=self.x(), y=self.y()))

                others_actions.append(self.yellow_agents[i].step(self.frame.robots_yellow[i], obstacles, dict(), random_target, True))



        return myActions + others_actions

    def update_path_and_targets(self):
        # Keep only the last M target points
        for target in self.targets:
            if target not in self.all_points:
                self.all_points.push(target)
                
        # Visible path drawing control
        for i in self.my_agents:
            self.robots_paths[i].push(Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y))

        # Check if the robot is close to the target
        for j in range(len(self.targets) - 1, -1, -1):
            for i in self.my_agents:
                if Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y).dist_to(self.targets[j]) < self.min_dist:
                    self.central_planner.agent_reached_target(self.targets[j])
                    
                    self.targets.pop(j)
                    break
        
        # Check if there are no more targets
        if len(self.targets) == 0:
            self.rounds -= 1

        # Finish the phase and increase the number of targets for the next phase
        if self.rounds == 0:
            self.rounds = self.max_rounds
            if self.targets_per_round < self.max_targets:
                self.targets_per_round += 1
                self.blue_agents.pop(len(self.my_agents))

                len_agents = len(self.my_agents)
                agent = MainAgent(len_agents, False)
                self.my_agents[len_agents] = agent



        # Generate new targets
        if len(self.targets) == 0:
            for i in range(self.targets_per_round):
                new_target = Point(self.x(), self.y())
                self.targets.append(new_target)
                self.central_planner.add_target(new_target)
                

    def step(self, action):
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        # print(f"Command: {commands}")
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        self.update_path_and_targets()

        if self.render_mode == "human":
            self.render()
        return observation, reward, done, False, {}

    def _calculate_reward_and_done(self):
        return 0, False
    
    def x(self):
        return random.uniform(-self.field.length/2 + self.min_dist, self.field.length/2 - self.min_dist)

    def y(self):
        return random.uniform(-self.field.width/2 + self.min_dist, self.field.width/2 - self.min_dist)
    
    def _get_initial_positions_frame(self):

        def theta():
            return random.uniform(0, 360)
    
        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=self.x(), y=self.y())

        pos_frame.robots_blue[0] = Robot(x=self.x(), y=self.y(), theta=theta())

        self.targets = [Point(x=self.x(), y=self.y())]


        target = self.targets[0]
        self.central_planner.add_target(target)

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (self.x(), self.y())
            while places.get_nearest(pos)[1] < self.min_dist:
                pos = (self.x(), self.y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())
        

        for i in range(0, self.n_robots_yellow):
            pos = (self.x(), self.y())
            while places.get_nearest(pos)[1] < self.min_dist:
                pos = (self.x(), self.y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    """
    "
    "RED": (151, 21, 0),
    WHITE": (220, 220, 220),
    "BG_GREEN": (20, 90, 45),
    "GREEN": (0, 128, 0),
    "ORANGE": (255, 106, 2),
    "BLUE": (0, 64, 255),
    "YELLOW": (250, 218, 94),
    "OLIVE": (128, 128, 0),
    "PURPLE": (102, 51, 153),
    "HAZEL": 	(250,165,112),
    """

    def _render_modified(self):


        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        ball = BallRender(
            *pos_transform(self.frame.ball.x, self.frame.ball.y),
            self.field_renderer.scale
        )
        self.field_renderer.draw(self.window_surface)

        for i in range(self.n_robots_blue):
            robot = self.frame.robots_blue[i]
            x, y = pos_transform(robot.x, robot.y)
            rbt = SSLRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["BLUE"]
                # colors[i] if i < len(self.my_agents) else COLORS["BLUE"]
            )
            
            rbt.draw(self.window_surface)

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i]
            x, y = pos_transform(robot.x, robot.y)
            rbt = SSLRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["YELLOW"],
            )
            rbt.draw(self.window_surface)
        ball.draw(self.window_surface)
  
    def _render(self):

        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        # Modified version of render from SSLBaseEnv to change agent color 
        
        self._render_modified()
        
        for target in self.targets:
            self.draw_target(
                self.window_surface,
                pos_transform,
                target,
                (255, 0, 255),
            )

        if len(self.all_points) > 0:
            my_path = [pos_transform(*p) for p in self.all_points]
            for point in my_path:
                pygame.draw.circle(self.window_surface, (255, 0, 0), point, 3)
        
        for i in range(len(self.robots_paths)):
            if len(self.robots_paths[i]) > 1:
                my_path = [pos_transform(*p) for p in self.robots_paths[i]]
                pygame.draw.lines(self.window_surface, (255, 0, 0), False, my_path, 1)


        # Render planned paths for each agent
        for agent_id, agent in self.my_agents.items():
            if hasattr(agent, 'get_planned_path'):
                planned_path = agent.get_planned_path()
                if planned_path and len(planned_path) > 0:
                    # print()
                    # Convert path points to screen coordinates
                    path_points = [pos_transform(p.x, p.y) for p in planned_path]
                    
                    # Draw points first
                    for point in path_points:
                        pygame.draw.circle(self.window_surface, colors[agent_id], point, 4)
                    
                    # Draw lines only if we have 2 or more points
                    if len(path_points) >= 2:
                        pygame.draw.lines(self.window_surface, colors[agent_id], False, path_points, 2)
                        
                    # Highlight start and end points
                    if path_points:
                        # Start point in green
                        pygame.draw.circle(self.window_surface, (0, 255, 0), path_points[0], 6, 2)
                        # End point in red
                        pygame.draw.circle(self.window_surface, (255, 0, 0), path_points[-1], 6, 2)
            


    def draw_target(self, screen, transformer, point, color):
        x, y = transformer(point.x, point.y)
        size = 0.09 * self.field_renderer.scale
        pygame.draw.circle(screen, color, (x, y), size, 2)