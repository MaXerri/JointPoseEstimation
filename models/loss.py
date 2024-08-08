import torch.nn as nn

class JointsMSELoss(nn.Module):

  """
  Weighted Heatmap Pixelwise MSE loss.  

  Params: 
     - use_visibility_weight: use weight based of joint visibility
  """

  def __init__(self, use_visibility_weight = True):
    super(JointsMSELoss, self).__init__()
    self.mse = nn.MSELoss(reduction="mean")
    self.use_visibility_weight = use_visibility_weight
    
  def forward(self, output, target, target_weight):
    """
    Params: 
      - output: model output (batch_size, num_joints, height?, width)
      - target: groundtrutch heatmaps
      - target weight: (batch size, num_joints, 1): based on the visibility of the joint
            
    """

    batch_size = output.shape[0]
    num_joints = output.shape[1] # get #joints for this image of (batch_size, num_joints, height, width)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1) # reshape heatmap predictions
    heatmaps_target = target.reshape((batch_size, num_joints, -1)).split(1, 1) # reshape target values

    loss = 0

    for joint_idx in range(num_joints):
      heatmap_pred = heatmaps_pred[joint_idx].squeeze(1)
      heatmap_target = heatmaps_target[joint_idx].squeeze(1)

      if self.use_visibility_weight:
        loss +=self.mse(target_weight[:,joint_idx] * heatmap_pred, target_weight[:,joint_idx]* heatmap_target)
      else:
        loss +=self.mse(heatmap_pred, heatmap_target)
    # average loss per joint because number of joints in images are variable
    return loss / num_joints 