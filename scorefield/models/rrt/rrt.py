import numpy as np
import random
import torch
import math

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

    def collision_check(self, node, obstacle_mask):
        h_pixel = self.norm_to_pixel(node.h)
        w_pixel = self.norm_to_pixel(node.w)
        
        if 0 <= h_pixel < self.image_size and 0 <= w_pixel < self.image_size:
            return not obstacle_mask[h_pixel, w_pixel].item()
        return False

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
        while True:
            rand_node = Node(random.uniform(-1, 1), random.uniform(-1, 1))
            if self.collision_check(rand_node, obstacle_mask):
                return rand_node

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
        nodes = [goal]  # start with goal

        for _ in range(max_iters):
            rand_node = goal
            nearest_node = self.find_nearest(start, nodes)
            new_node = self.steer(nearest_node, rand_node)

            if not self.collision_check(new_node, obstacle_mask):
                continue

            nodes.append(new_node)
            neighbours = self.get_neighbour_nodes(new_node, nodes)
            self.optimize_path(new_node, neighbours, obstacle_mask)

            if self.distance(new_node, start) < self.delta_dist or len(nodes) == self.time_steps:
                path = self.get_path(new_node)
                while len(path) < self.time_steps:
                    path.insert(0, path[0])
                return path

        return None
    
    def steer(self, from_node, to_node):
        theta = np.arctan2(to_node.h - from_node.h, to_node.w - from_node.w)
        steps_remaining = self.time_steps - len(self.paths)  # paths represent the nodes added so far
        dist = self.distance(from_node, to_node)
        step_dist = dist / steps_remaining
        new_node = Node(from_node.h + step_dist * np.sin(theta),
                        from_node.w + step_dist * np.cos(theta))
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.distance(new_node, from_node)
        return new_node

    def get_path(self, last_node):
        path = []
        while last_node:
            path.append((last_node.h, last_node.w))
            last_node = last_node.parent
        return path[::-1]
    
    def norm_to_pixel(self, coordinate): # [-1~1] -> 64x64
        return int((coordinate + 1) * self.image_size / 2)        

    def plan(self, starts, goals, obstacle_masks=None, max_iters=2000):
        self.B = starts.shape[0]
        self.starts = [Node(h.item(), w.item()) for h, w in starts[:, 0]]
        self.goals = [Node(h.item(), w.item()) for h, w in goals[:, 0]]
        
        if obstacle_masks is not None:
            obstacle_masks = obstacle_masks.to(self.device)

        self.paths = []
        for b in range(self.B):
            self.paths.append(self.plan_for_one(self.starts[b], self.goals[b], obstacle_masks[b] if obstacle_masks is not None else None, max_iters))
            if len(self.paths[b]) < self.time_steps:
                repeats = self.time_steps - len(self.paths[b])
                for _ in range(repeats):
                    self.paths[b].append(self.paths[b][-1])
        
        return self.paths

    def random_sample(self):
        return [random.randint(1, self.time_steps-2) for _ in range(self.B)]
