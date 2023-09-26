import torch

class RRTStar:
    def __init__(self, max_iters=200, epsilon=0.2, gamma=20.0, device='cuda'):
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.device=device

    def get_random_point(self, B, device):
        return (2 * torch.rand(B, 1, 2, device=device) - 1)

    def is_inside_boundary(self, point):
        return (point[..., 0] >= -1) & (point[..., 0] <= 1) & (point[..., 1] >= -1) & (point[..., 1] <= 1)

    def nearest(self, nodes, point):
        # Calculate the squared distances for computational efficiency
        squared_distances = ((nodes - point) ** 2).sum(dim=-1)
        return torch.argmin(squared_distances, dim=-1)

    def steer(self, from_node, to_node):
        # Both from_node and to_node should have shape (1,2)
        from_node = from_node.unsqueeze(0)
        vec = to_node - from_node
        dist = torch.norm(vec, dim=-1, keepdim=True)
        step = torch.clamp(dist, max=self.epsilon) / dist
        new_point = from_node + step * vec
        inside_boundary = self.is_inside_boundary(new_point)
        new_point = torch.where(inside_boundary, new_point, from_node)
        return new_point.squeeze(0)  # Ensure shape is (1,2)


    def collision_free(self, p1, p2, obstacle_mask):
        # Convert the points from [-1, 1] to pixel space [0, 63]
        p1_pixel = ((p1 + 1) * 31.5).long()
        p2_pixel = ((p2 + 1) * 31.5).long()

        # Bresenham's line algorithm to check collision
        dx = (p2_pixel[..., 0] - p1_pixel[..., 0]).abs()
        dy = -(p2_pixel[..., 1] - p1_pixel[..., 1]).abs()
        sx = torch.where(p2_pixel[..., 0] > p1_pixel[..., 0], 1, -1)
        sy = torch.where(p2_pixel[..., 1] > p1_pixel[..., 1], 1, -1)
        err = dx + dy

        collision = torch.zeros(p1.shape[:-1], dtype=torch.bool, device=p1.device)

        for _ in range(64):  # maximum number of iterations for our 64x64 grid
            if torch.all(collision):
                break

            # Accessing obstacle_masks correctly using integer coordinates
            collision |= obstacle_mask[p1_pixel[..., 1], p1_pixel[..., 0]]

            e2 = 2 * err
            change_x = e2 > dy
            change_y = e2 < dx

            err += dy * change_x + dx * change_y
            
            # Update pixel values while ensuring they remain within valid bounds
            p1_pixel[..., 0] = torch.clamp(p1_pixel[..., 0] + sx * change_x, 0, 63)
            p1_pixel[..., 1] = torch.clamp(p1_pixel[..., 1] + sy * change_y, 0, 63)

        return ~collision


    def cost(self, point1, point2):
        point1 = point1.view(-1, 2).squeeze()
        point2 = point2.view(-1, 2).squeeze()
        return torch.norm(point2 - point1)
    
    def find_path(self, starts, goals, obstacle_masks):
        B, N, _ = starts.shape
        paths = torch.zeros(B, N, 2, device=starts.device)

        for i in range(B):
            for j in range(N):
                start = starts[i, j, None, :]
                goal = goals[i]

                nodes = start.clone().unsqueeze(1)  # Start with shape (1,2)
                costs = torch.tensor([0.0], device=starts.device)

                for _ in range(self.max_iters):
                    rnd_point = self.get_random_point(1, starts.device)
                    print(nodes.shape, rnd_point.shape)
                    
                    near_idx = self.nearest(nodes, rnd_point)
                    new_point = self.steer(nodes[near_idx], rnd_point)

                    if self.collision_free(nodes[near_idx], new_point, obstacle_masks[i]):
                        neighbors_idx = (torch.norm(nodes - new_point, dim=-1) < self.gamma).nonzero(as_tuple=True)[0]
                        if neighbors_idx.numel() == 0:
                            continue

                        costs_with_new = costs[neighbors_idx] + self.cost(nodes[neighbors_idx], new_point)
                        min_cost, min_idx = costs_with_new.min(0)

                        nodes = torch.cat([nodes, new_point], dim=1)
                        costs = torch.cat([costs, min_cost.unsqueeze(0)])

                        for idx in neighbors_idx:
                            potential_cost = costs[idx] + self.cost(nodes[idx].unsqueeze(0), new_point_expanded.unsqueeze(0))
                            if potential_cost < costs[-1] and self.collision_free(nodes[idx], new_point_expanded, obstacle_masks[i]):
                                costs[-1] = potential_cost


                last_node = nodes[-1]
                if self.collision_free(last_node, goal, obstacle_masks[i]):
                    paths[i, j] = last_node

        return paths

    def run(self, starts, goals, obstacle_masks):
        return self.find_path(starts, goals, obstacle_masks)
