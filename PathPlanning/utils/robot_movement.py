from utils.ssl.Navigation import Navigation
from utils.Point import Point
from typing import List, Set, Tuple
import numpy as np
from rsoccer_gym.Entities import Robot







def get_evasive_velocity(opponents: dict[int, Robot], robot: Robot, safety_radius: float, max_speed: float) -> Point:
    """Calculate evasive velocity when robots get too close while standing."""
    evasive_x, evasive_y = 0.0, 0.0
    
    for opponent in opponents.values():
        dx = robot.x - opponent.x
        dy = robot.y - opponent.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < safety_radius:
            force = (safety_radius - distance) / safety_radius
            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance
            else: 
                angle = np.random.random() * 2 * np.pi
                dx = np.cos(angle)
                dy = np.sin(angle)
            
            
            evasive_x += dx * force * max_speed * 0.5
            evasive_y += dy * force * max_speed * 0.5
    
    # If no evasion needed, return zero velocity
    if evasive_x == 0 and evasive_y == 0:
        return Point(0.0, 0.0)
        
    # Normalize and scale evasive velocity
    magnitude = np.sqrt(evasive_x*evasive_x + evasive_y*evasive_y)
    if magnitude > max_speed:
        evasive_x = (evasive_x / magnitude) * max_speed
        evasive_y = (evasive_y / magnitude) * max_speed
        
    return Point(evasive_x, evasive_y)






