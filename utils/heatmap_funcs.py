import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom 

def generate_gaussian_heatmap(list_of_joint_list, resolution_size, sigma=1.5):
    """
    Generate a Gaussian heatmap centered at a given coordinate.

    Parameters:
    - list_of_joint_lists: 4D tensor of (batch_size, # joints, 3).
    - resolution_size: Tuple (height, width) representing the resolution of the heatmap.
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - gaussian_heatmap: NumPy array representing the generated Gaussian heatmap.
    """
    assert list_of_joint_list.shape[1] == 14
    assert list_of_joint_list.shape[2] == 3
    assert len(resolution_size) == 2

    output = np.zeros((len(list_of_joint_list), 14, resolution_size[0], resolution_size[1]))
    for i in range(len(list_of_joint_list)):
        heatmaps_for_n_joints = np.zeros((14, resolution_size[0], resolution_size[1]))
        for j in range(14):
            
            if list_of_joint_list[i][j][2] == 1: # joint is visible 
                
                coordinate = list_of_joint_list[i][j]
                gaussian_heatmap = np.zeros(resolution_size)
                # Create meshgrids for the entire resolution_size
                y, x = np.meshgrid(np.arange(resolution_size[0]), np.arange(resolution_size[1]))

                # Use NumPy broadcasting for vectorized operations
                gaussian_heatmap = np.exp(-((x - coordinate[0])**2 + (y - coordinate[1])**2) / (2.0 * sigma**2))

                # Normalize the heatmap to have values between 0 and 1
                gaussian_heatmap /= np.max(gaussian_heatmap)

                heatmaps_for_n_joints[j] = gaussian_heatmap 
            else:
                heatmaps_for_n_joints[j] = np.ones(resolution_size) / 10.0 # set to low confidence value for non-visible joints
        output[i] = heatmaps_for_n_joints
        
    return output

def upsample_heatmap(heatmaps, target_resolution):
    """
    Upsample a heatmap to a target resolution using bilinear interpolation.

    Parameters:
    - heatmap: 4D NumPy array representing batches of the 14 heatmaps per image the lower-resolution heatmap.
    - target_resolution: Tuple (height, width) representing the target resolution.

    Returns:
    - upsampled_heatmap: array of 2D NumPy array representing the upsampled heatmap.
    """
    # Function does not produce smooth gaussian yet 
    output = np.zeros((len(heatmaps), 14, target_resolution[0], target_resolution[1]))
    for b in range(len(heatmaps)):
        for j in range(len(heatmaps[b])):
            # Get the current resolution of the heatmap
            current_resolution = heatmaps[b][j].shape

            # Calculate the scaling factors for height and width
            scale_factor_row = target_resolution[0] / current_resolution[0]
            scale_factor_col = target_resolution[1] / current_resolution[1]

            # Use scipy's zoom function for bilinear interpolation
            upsampled_heatmap = zoom(heatmaps[b][j], (scale_factor_row, scale_factor_col), order=1)
            output[b][j] = upsampled_heatmap
    return output
