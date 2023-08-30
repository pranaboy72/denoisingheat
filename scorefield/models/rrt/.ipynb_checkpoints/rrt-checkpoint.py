import numpy as np
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        
def distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def get_random_node(x_max, y_max):
    return Node(random.uniform(0, x_max), random.uniform(0, y_max))

def get_nearest_node(node_list, random_node):
    dlist = [distance(node, random_node) for node in node_list]
    return node_list[np.argmin(dlist)]

def steer(from_node, to_node, extend_length=float('inf')):
    angle = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
    length = distance(from_node, to_node)
    
    if extend_length > length:
        extend_length = length
        
    x = from_node.x + extend_length * np.cos(angle)
    y = from_node.y + extend_length * np.sin(angle)
    
    return Node(x,y)

def rrt_planning(start, goal, x_max, y_max, expand_dis=0.1, max_nodes=500):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    node_list = [start_node]
    
    for _ in range(max_nodes):
        rnd_node = get_random_node(x_max, y_max)
        nearest_node = get_nearest_node(node_list, rnd_node)
        new_node = steer(nearest_node, rnd_node, expand_dis)
        new_node.parent = nearest_node
        node_list.append(new_node)
        
        if distance(new_node, goal_node) < expand_dis:
            final_node = steer(new_node, goal_node, expand_dis)
            final_node.parent = new_node
            node_list.append(final_node)
            break
        
    path = [[goal[0], goal[1]]]
    while final_node.parent:
        path.append([final_node.x, final_node.y])
        final_node = final_node.parent
    path.append([start[0], start[1]])
    
    return path[::-1]


    