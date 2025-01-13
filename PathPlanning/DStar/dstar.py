from typing import List, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt

# Toggle animation
show_animation = True

class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 35):
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate cell dimensions
        self.cell_length = field_length / grid_size
        self.cell_width = field_width / grid_size
        
    def create_grid(self, obstacles: dict, robot_radius: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # For each cell, check if it intersects with any obstacle
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_x = -self.field_length/2 + i * self.cell_length + self.cell_length/2
                cell_y = -self.field_width/2 + j * self.cell_width + self.cell_width/2
                
                # Check each obstacle
                for robot in obstacles.values():
                    dist = np.hypot(robot['x'] - cell_x, robot['y'] - cell_y)
                    if dist < robot_radius + self.cell_length/2:
                        grid[i, j] = 1
                        break
                        
        return grid
    
    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int((x + self.field_length/2) / self.cell_length)
        grid_y = int((y + self.field_width/2) / self.cell_width)
        
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))
        
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = -self.field_length/2 + grid_x * self.cell_length + self.cell_length/2
        y = -self.field_width/2 + grid_y * self.cell_width + self.cell_width/2
        return x, y

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."  # state type
        self.t = "new"  # tag for state
        self.h = 0  # heuristic
        self.k = 0  # key value for priority
    
    def cost(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Dstar:
    def __init__(self, grid, converter):
        self.grid = grid
        self.converter = converter
        self.open_list = set()
    
    def process_state(self):
        x = self.min_state()
        if x is None:
            return -1
        
        k_old = self.get_kmin()
        self.remove(x)
        
        if k_old < x.h:
            for y in self.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        
        if k_old == x.h:
            for y in self.get_neighbors(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)) or \
                   (y.parent != x and y.h > x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.get_neighbors(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                elif y.parent != x and y.h > x.h + x.cost(y):
                    self.insert(x, x.h)
                elif y.parent != x and x.h > y.h + x.cost(y) and y.t == "close" and y.h > k_old:
                    self.insert(y, y.h)
        return self.get_kmin()
    
    def min_state(self):
        if not self.open_list:
            return None
        return min(self.open_list, key=lambda state: state.k)
    
    def get_kmin(self):
        if not self.open_list:
            return -1
        return min(state.k for state in self.open_list)
    
    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)
    
    def remove(self, state):
        state.t = "close"
        self.open_list.remove(state)
    
    def get_neighbors(self, state):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dx, dy in directions:
            nx, ny = state.x + dx, state.y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                if self.grid[nx, ny] == 0:  # Not an obstacle
                    neighbors.append(State(nx, ny))
        return neighbors
    
    def run(self, start, goal):
        rx, ry = [], []
        start_state = State(*start)
        goal_state = State(*goal)
        self.insert(goal_state, 0)
        
        while True:
            self.process_state()
            if start_state.t == "close":
                break
        
        state = start_state
        while state != goal_state:
            rx.append(state.x)
            ry.append(state.y)
            state = state.parent
            if show_animation:
                px, py = self.converter.grid_to_continuous(state.x, state.y)
                plt.plot(px, py, "xr")
                plt.pause(0.01)
        
        return rx, ry

def main():
    field_length, field_width = 100, 100
    grid_size = 35
    robot_radius = 0.5
    
    obstacles = {1: {'x': 10, 'y': 10}, 2: {'x': 20, 'y': 20}, 3: {'x': 50, 'y': 50}}
    converter = GridConverter(field_length, field_width, grid_size)
    grid = converter.create_grid(obstacles, robot_radius)
    
    start = (5.0, 5.0)  # Continuous start position
    goal = (45.0, 45.0)  # Continuous goal position
    
    start_grid = converter.continuous_to_grid(*start)
    goal_grid = converter.continuous_to_grid(*goal)
    
    dstar = Dstar(grid, converter)
    rx, ry = dstar.run(start_grid, goal_grid)
    
    if show_animation:
        plt.show()

if __name__ == '__main__':
    main()
