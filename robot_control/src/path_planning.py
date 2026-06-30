import numpy as np
import math
import heapq

class Node:
    def __init__(self, point):
        self.point = np.array(point) # Can be [x,y] or [theta1...theta6]
        self.parent = None
        self.cost = 0.0

def distance(p1, p2):
    """Calculates Euclidean distance in N-dimensions."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def check_collision(point1, point2, obstacles):
    """
    For a 6-axis robot, this is where you run Forward Kinematics (FK).
    You convert the 6 joint angles into a 3D bounding box for the robot,
    and check if it intersects with the 3D bounding boxes of your obstacles.
    """
    steps = 10
    for i in range(steps + 1):
        # Interpolate along the path between the two nodes
        t = i / steps
        intermediate_point = point1 + t * (point2 - point1)
        
        # Simplified 2D/3D check: 
        # In reality, check if robot_mesh(intermediate_point) intersects obstacles
        for obs in obstacles:
            if np.linalg.norm(intermediate_point - obs['center']) < obs['radius']:
                return True # Collision detected!
    return False

def generate_rrt_graph(start_coords, goal_coords, obstacles, bounds, step_size=0.5, max_nodes=1000):
    """Builds a random tree avoiding dynamic obstacles."""
    start_node = Node(start_coords)
    goal_node = Node(goal_coords)
    
    nodes = [start_node]
    
    for _ in range(max_nodes):
        # 1. Random Sample (Bias 10% of the time towards the goal to speed it up)
        if np.random.rand() < 0.1:
            rand_point = goal_node.point
        else:
            rand_point = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
        # 2. Find Nearest Node in existing tree
        nearest_node = min(nodes, key=lambda n: distance(n.point, rand_point))
        
        # 3. Steer towards random point by step_size
        direction = rand_point - nearest_node.point
        length = np.linalg.norm(direction)
        if length > step_size:
            direction = (direction / length) * step_size
        new_point = nearest_node.point + direction
        
        # 4. Collision Check
        if not check_collision(nearest_node.point, new_point, obstacles):
            new_node = Node(new_point)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + distance(nearest_node.point, new_point)
            nodes.append(new_node)
            
            # Check if we reached the goal
            if distance(new_node.point, goal_node.point) < step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                break
                
    return nodes, goal_node

def a_star_search(goal_node):
    """
    A* algorithm to trace back the optimal path from the Goal to the Start.
    In standard RRT, there is only one path back to the root, so this acts 
    as a backtracker. If you implement RRT* (which creates a true mesh graph), 
    A* evaluates the lowest cost path.
    """
    path = []
    current = goal_node
    
    # Backtrack through parent pointers
    while current is not None:
        path.append(current.point)
        current = current.parent
        
    return path[::-1] # Reverse to get Start -> Goal order

# --- Example Usage for a dynamic environment ---
# 6D Bounds for a robot (e.g., -180 to 180 degrees for 6 joints)
joint_bounds = np.array([[-np.pi, np.pi]] * 6) 
start_angles = [0, 0, 0, 0, 0, 0]
target_angles = [1.5, -0.5, 1.0, 0, 0, 0]

# Dynamic Obstacles captured by your depth sensor (converted to XYZ spheres/boxes)
dynamic_obstacles = [
    {'center': np.array([0.5, 0.5, 0.5]), 'radius': 0.2},
    {'center': np.array([1.2, -0.3, 0.8]), 'radius': 0.15}
]

print("Building RRT Graph...")
tree_nodes, reached_goal = generate_rrt_graph(
    start_angles, 
    target_angles, 
    dynamic_obstacles, 
    joint_bounds
)

if reached_goal.parent is not None:
    print("Graph connected to goal! Running A* Search...")
    optimal_path = a_star_search(reached_goal)
    print(f"Path found with {len(optimal_path)} waypoints.")
else:
    print("Failed to find a path. Increase max_nodes or check if obstacles block the goal.")