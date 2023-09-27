import numpy as np
import random
import torch
import math
from scipy.interpolate import interp1d

class Node:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.parent = None
        self.cost = 0

class RRTStar:
    def __init__(self, image_size, time_steps, delta_dist=0.05, radius=0.1, device='cuda', random_seed=None):
        self.image_size = image_size
        self.time_steps = time_steps
        self.delta_dist = delta_dist
        self.radius = radius
        self.device = device
        self.random_seed = random_seed
        self.set_random_seed()
        
    def set_random_seed(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

    def distance(self, n1, n2):
        return math.sqrt((n1.h - n2.h)**2 + (n1.w - n2.w)**2)

    def collision_check(self, node, obstacle_mask):
        if obstacle_mask is None:
            return True
        h_index = int(node.h * 0.5 + 0.5)
        w_index = int(node.w * 0.5 + 0.5)

        if 0 <= h_index < obstacle_mask.shape[0] and 0 <= w_index < obstacle_mask.shape[1]:
            return not obstacle_mask[h_index, w_index].item()

        return False


    def steer(self, from_node, to_node):
        theta = np.arctan2(to_node.h - from_node.h, to_node.w - from_node.w)
        new_node = Node(from_node.h + self.delta_dist * np.sin(theta),
                        from_node.w + self.delta_dist * np.cos(theta))
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.distance(new_node, from_node)
        return new_node


    def find_nearest(self, node, nodes):
        nearest = nodes[0]
        min_dist = self.distance(node, nearest)
        for n in nodes:
            d = self.distance(node, n)
            if d < min_dist:
                nearest = n
                min_dist = d
        return nearest

    def get_random_node(self, obstacle_mask):
        count = 0
        while count < 100:
            rand_node = Node(random.uniform(-1, 1), random.uniform(-1, 1))
            if self.collision_check(rand_node, obstacle_mask):
                return rand_node
            count += 1
        return None


    def get_neighbour_nodes(self, new_node, nodes):
        return [node for node in nodes if self.distance(node, new_node) <= self.radius]


    def optimize_path(self, new_node, neighbours, obstacle_mask):
        if not neighbours:
            return
        min_cost = new_node.cost
        best_parent = new_node.parent
        for neighbour in neighbours:
            if self.collision_check(neighbour, obstacle_mask):
                continue
            temp_cost = neighbour.cost + self.distance(neighbour, new_node)
            if temp_cost < min_cost:
                min_cost = temp_cost
                best_parent = neighbour
        new_node.parent = best_parent
        new_node.cost = min_cost


    def get_path(self, last_node):
        path = []
        while last_node:
            path.append((last_node.h, last_node.w))
            last_node = last_node.parent
        return path[::-1]
    
    def norm_to_pixel(self, coordinate): # [-1~1] -> 64x64
        return int((coordinate + 1) * self.image_size / 2)        
    
    def shortcut_path(self, path, obstacle_mask):
        # Start with the full path
        optimized_path = path.copy()
        i = 0
        # While there's more path left to optimize
        while i < len(optimized_path) - 2:
            nodeA = Node(*optimized_path[i])
            for j in range(len(optimized_path) - 1, i + 1, -1):
                nodeB = Node(*optimized_path[j])
                # If the straight path between nodeA and nodeB is collision-free, make a shortcut
                if self.collision_check(nodeA, obstacle_mask, nodeB):
                    # Remove nodes between nodeA and nodeB
                    del optimized_path[i+1:j]
                    break
            i += 1
        return optimized_path
    
    def extract_nodes_from_tensors(self, starts, goals):
        B, N, _ = starts.shape
        self.starts = [[Node(h.item(), w.item()) for h, w in start_batch] for start_batch in starts]
        self.goals = [[Node(h.item(), w.item()) for _ in range(N)] for h, w in goals.squeeze(1)]  # Duplicate the goal for N times


    def interpolate_path(self, path, num_points=10):
        # Extracting x and y coordinates from path
        x, y = zip(*path)
        # Creating a parameter for our path
        u = np.linspace(0, 1, len(path))
        # Creating the interpolation functions
        fx = interp1d(u, x, kind='linear')
        fy = interp1d(u, y, kind='linear')
        # Creating new parameters for the desired number of points
        new_u = np.linspace(0, 1, num_points)
        # Getting the new interpolated path
        new_x = fx(new_u)
        new_y = fy(new_u)
        return list(zip(new_x, new_y))

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points


    def collision_check_line(self, start, end, obstacle_mask):
        x0, y0 = self.norm_to_pixel(start.h), self.norm_to_pixel(start.w)
        x1, y1 = self.norm_to_pixel(end.h), self.norm_to_pixel(end.w)
        
        # Get all the cells the line would pass through using Bresenham's
        for x, y in self.bresenham_line(x0, y0, x1, y1):
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                if obstacle_mask[x, y].item():
                    return False
        return True
    
    def collision_check(self, new_node, obstacle_mask, previous_node=None):
        """ Collision check using Bresenham's Line Algorithm """
        if previous_node:
            return self.collision_check_line(previous_node, new_node, obstacle_mask)
        else:
            h_pixel = self.norm_to_pixel(new_node.h)
            w_pixel = self.norm_to_pixel(new_node.w)
            if 0 <= h_pixel < self.image_size and 0 <= w_pixel < self.image_size:
                return not obstacle_mask[h_pixel, w_pixel].item()
            return False
        
    def collision_check_with_margin(self, node, obstacle_mask, margin=3):
        h_pixel = self.norm_to_pixel(node.h)
        w_pixel = self.norm_to_pixel(node.w)
        
        for dh in range(-margin, margin+1):
            for dw in range(-margin, margin+1):
                h_index = h_pixel + dh
                w_index = w_pixel + dw

                if 0 <= h_index < self.image_size and 0 <= w_index < self.image_size:
                    if obstacle_mask[h_index, w_index].item():
                        return False
        return True
        
    def plan_for_one(self, start, goal, obstacle_mask, max_iters=2000):
        nodes = [start]

        for _ in range(max_iters):
            rand_node = self.get_random_node(obstacle_mask)
            if rand_node is None:
                continue
            nearest_node = self.find_nearest(rand_node, nodes)
            new_node = self.steer(nearest_node, rand_node)

            # Check if the direct line between nearest_node and new_node collides with any obstacle
            if not self.collision_check_line(nearest_node, new_node, obstacle_mask):
                continue

            if not self.collision_check_with_margin(new_node, obstacle_mask):
                continue

            nodes.append(new_node)
            neighbours = self.get_neighbour_nodes(new_node, nodes)
            self.optimize_path(new_node, neighbours, obstacle_mask)

            if self.distance(new_node, goal) < self.delta_dist:
                return self.get_path(new_node)

        print("Warning: RRT* couldn't find a path in given iterations. Consider increasing max_iters.")
        return [None]  # Default path staying at the start position


    def plan(self, starts, goals, obstacle_masks=None, max_iters=5000):
        self.B, N, _ = starts.shape

        self.starts = [[Node(h.item(), w.item()) for h, w in start_batch] for start_batch in starts]
        self.goals = [[Node(h.item(), w.item()) for h, w in goal_batch] for goal_batch in goals]

        if obstacle_masks is not None:
            obstacle_masks = obstacle_masks.to(self.device)

        all_paths = []
        all_deltas = []

        for b in range(self.B):
            paths_for_batch = []
            deltas_for_batch = []
            
            # Finding path for each start and goal pair
            for n in range(N):
                path = self.plan_for_one(self.starts[b][n], self.goals[b][0], obstacle_masks[b] if obstacle_masks is not None else None, max_iters)
                if path[0] == None:
                    return None, None
                paths_for_batch.append(path)
                
                deltas = [(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1)]
                deltas_for_batch.append(deltas)
                
            all_paths.append(paths_for_batch)
            all_deltas.append(deltas_for_batch)

        return all_paths, all_deltas