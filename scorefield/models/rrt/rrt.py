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
    def __init__(self, image_size, time_steps, delta_dist=0.05, radius=0.1, device='cuda'):
        self.image_size = image_size
        self.time_steps = time_steps
        self.delta_dist = delta_dist
        self.radius = radius
        self.device = device

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


    def collision_check(self, new_node, obstacle_mask, previous_node=None):
        h_pixel = self.norm_to_pixel(new_node.h)
        w_pixel = self.norm_to_pixel(new_node.w)

        if previous_node is None:
            if 0 <= h_pixel < self.image_size and 0 <= w_pixel < self.image_size:
                return not obstacle_mask[h_pixel, w_pixel].item()
            return False

        num_steps = int(self.distance(previous_node, new_node) / self.delta_dist)
        if num_steps == 0:
            num_steps = 1

        for i in range(num_steps):
            alpha = i / num_steps
            h = previous_node.h * (1 - alpha) + new_node.h * alpha
            w = previous_node.w * (1 - alpha) + new_node.w * alpha
            h_pixel = self.norm_to_pixel(h)
            w_pixel = self.norm_to_pixel(w)
            if 0 <= h_pixel < self.image_size and 0 <= w_pixel < self.image_size:
                if obstacle_mask[h_pixel, w_pixel].item():
                    return False
        return True


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

    def plan_for_one(self, start, goal, obstacle_mask, max_iters=2000):
        nodes = [goal]  # start with the goal

        for _ in range(max_iters):
            rand_node = self.get_random_node(obstacle_mask)
            if rand_node is None:
                continue
            nearest_node = self.find_nearest(rand_node, nodes)
            new_node = self.steer(nearest_node, rand_node)

            if not self.collision_check(new_node, obstacle_mask, nearest_node):
                continue

            nodes.append(new_node)
            neighbours = self.get_neighbour_nodes(new_node, nodes)
            self.optimize_path(new_node, neighbours, obstacle_mask)

            if self.distance(new_node, start) < self.delta_dist:
                raw_path = self.get_path(new_node)
                path = self.interpolate_path(raw_path, self.time_steps)
                # Optimize the path
                # path = self.shortcut_path(path, obstacle_mask)

                while len(path) < self.time_steps:
                    path.insert(0, path[0])
                deltas = [(path[i][0]-path[i+1][0], path[i][1]-path[i+1][1]) for i in range(len(path)-1)]
                deltas.insert(0, (0, 0))
                return path, deltas

        print("Warning: RRT* couldn't find a path in given iterations. Consider increasing max_iters.")
        default_path = [(start.h, start.w) for _ in range(self.time_steps)]  # Default path staying at the start position
        return default_path, [(0, 0) for _ in range(self.time_steps)]


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



    def collision_check_line(self, start, end, obstacle_mask):
        num_steps = int(self.distance(start, end) / self.delta_dist)
        if num_steps == 0:
            num_steps = 1

        for i in range(num_steps):
            alpha = i / num_steps
            h = start.h * (1 - alpha) + end.h * alpha
            w = start.w * (1 - alpha) + end.w * alpha
            h_pixel = self.norm_to_pixel(h)
            w_pixel = self.norm_to_pixel(w)
            if 0 <= h_pixel < self.image_size and 0 <= w_pixel < self.image_size:
                if obstacle_mask[h_pixel, w_pixel].item():
                    return False
        return True

    def plan(self, starts, goals, obstacle_masks=None, max_iters=4000):
        self.B, N, _ = starts.shape
        
        self.starts = [[Node(h.item(), w.item()) for h, w in start_batch] for start_batch in starts]
        self.goals = [Node(h.item(), w.item()) for h, w in goals[:, 0]]

        if obstacle_masks is not None:
            obstacle_masks = obstacle_masks.to(self.device)

        all_paths = torch.zeros((self.B, N, self.time_steps, 2), dtype=starts.dtype, device=starts.device)
        all_deltas = torch.zeros((self.B, N, self.time_steps, 2), dtype=starts.dtype, device=starts.device)


        for b in range(self.B):
            for n in range(N):
                path, delta = self.plan_for_one(self.starts[b][n], self.goals[b], obstacle_masks[b] if obstacle_masks is not None else None, max_iters)
                
                all_paths[b, n] = torch.tensor(path, dtype=starts.dtype, device=starts.device)
                all_deltas[b, n] = torch.tensor(delta, dtype=starts.dtype, device=starts.device)

        return torch.flip(all_paths, [2]), torch.flip(all_deltas, [2])


