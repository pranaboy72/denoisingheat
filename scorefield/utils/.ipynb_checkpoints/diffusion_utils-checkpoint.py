import torch
import torch.nn.functional as F


def bilinear_interpolate(fmap, pos):
    """
    Returns the bilinear interpolated feature vector at pos.
    
    Parameters:
        fmap (torch.Tensor): The output feature map of the model of shape (N, C, H, W)
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
   
def bilinear_interpolate_batch(fmap, pos):
    """
    Returns the bilinear interpolated feature vector at pos in each fmap.
    
    Parameters:
        fmap (torch.Tensor): The output feature map of the model of shape (N, C, H, W)
        pos (torch.Tensor): Tensor of positions of shape (N, P, 2) where each row is (x, y) ranging from -1 to 1
        
    Returns:
        torch.Tensor: Interpolated feature vectors of shape (N, P, C)
    """
    # Ensure fmap and pos are compatible
    assert fmap.size(0) == pos.size(0), 'Batch size of fmap and pos must be the same'
    
    # Repeat the feature map for each position
    grid = pos.unsqueeze(2)

    # Ensure they are on the same device
    grid = grid.to(fmap.device)

    # Perform bilinear interpolation
    interpolated_value = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    
    return interpolated_value.squeeze(-1).permute(0,2,1)


def bilinear_interpolate_version(fmap, pos):
    """
    Returns the bilinear interpolated feature vector at pos in each fmap.
    
    Parameters:
        fmap (torch.Tensor): The output feature map of the model of shape (N, C, H, W)
        pos (torch.Tensor): Tensor of positions of shape (N, P, 2) where each row is (x, y) ranging from -1 to 1
        
    Returns:
        torch.Tensor: Interpolated feature vectors of shape (N*P, C)
    """
    # Ensure fmap and pos are compatible
    assert fmap.size(0) == pos.size(0), 'Batch size of fmap and pos must be the same'
    
    # Ensure they are on the same device
    pos = pos.to(fmap.device)

    # Convert the pos tensor from shape (N, P, 2) to (N, P, 1, 2)
    grid = pos.unsqueeze(2)

    # Perform bilinear interpolation
    interpolated_value = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    
    # Reshape interpolated_value to (N*P, C)
    interpolated_value = interpolated_value.squeeze(-1).permute(0, 2, 1).reshape(-1, fmap.size(1))
    
    return interpolated_value
    