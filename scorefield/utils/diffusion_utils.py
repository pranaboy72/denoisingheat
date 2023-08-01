import torch
import torch.nn.functional as F


def bilinear_interpolate(fmap, pos):
    """
    Returns the bilinear interpolated feature vector at pos.
    
    Parameters:
        fmap (torch.Tensor): The output feature map of a FCN of shape (N, C, H, W)
        pos (torch.Tensor): Tensor of positions of shape (N, 2) where each row is (x, y) ranging from -1 to 1
        
    Returns:
        torch.Tensor: Interpolated feature vectors of shape (N, C)
    """
    
    # Ensure fmap and pos are compatible
    assert fmap.size(0) == pos.size(0), "Batch size of fmap and pos must be the same"

    # Reshape the pos tensor to make it compatible with grid_sample
    grid = pos.unsqueeze(-2).unsqueeze(-2)  # Shape will be (N, 1, 1, 2)

    # Ensure grid and fmap are on the same device
    grid = grid.to(fmap.device)

    # Perform bilinear interpolation
    interpolated_value = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # Remove the spatial dimensions and return the interpolated feature vector
    return interpolated_value.squeeze(-1).squeeze(-1)

def compute_positions_based_on_time(t, H=256, W=256):
        # Modify this function to calculate the position based on time
        x = ((t % W) / W) * 2 - 1
        y = ((t // H) / H) * 2 - 1
        pos = torch.stack([x, y], dim=-1)  # Combine x and y to form position tensor
        return pos