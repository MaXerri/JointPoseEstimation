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
    
  def forward(self, output, target, target_weights):
    """
    Params: 
      - output: model output (batch_size, num_joints, height?, width)
      - target: groundtrutch heatmaps
      - target weight: (batch size, num_joints, 1): based on the visibility of the joint
            
    """

    batch_size = output.shape[0]
    num_joints = output.shape[1] # get #joints for this image of (batch_size, num_joints, height, width)

    loss = 0

    for joint_idx in range(num_joints):
      
      heatmap_pred = output[:,joint_idx,:,:]
      heatmap_target = target[:,joint_idx,:,:]

      target_weight = target_weights[:,joint_idx].unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1)

      if self.use_visibility_weight:
        loss +=self.mse(target_weight * heatmap_pred, target_weight* heatmap_target) # multiplication will broadcast 
      else:
        loss +=self.mse(heatmap_pred, heatmap_target)
    # average loss per joint because number of joints in images are variable
    return (loss / num_joints) / batch_size
  

class PenalizedJointsMSELoss(nn.Module):

  """
  Weighted Heatmap Pixelwise MSE loss wit penalty for predicting a zero heatmap for a visible joint

  Params: 
     - use_visibility_weight: use weight based of joint visibility
  """

  def __init__(self, use_visibility_weight = True):
    super(JointsMSELoss, self).__init__()
    self.mse = nn.MSELoss(reduction="mean")
    self.use_visibility_weight = use_visibility_weight
    
  def forward(self, output, target, target_weights):
    """
    Params: 
      - output: model output (batch_size, num_joints, height?, width)
      - target: groundtrutch heatmaps
      - target weight: (batch size, num_joints, 1): based on the visibility of the joint
            
    """

    batch_size = output.shape[0]
    num_joints = output.shape[1] # get #joints for this image of (batch_size, num_joints, height, width)

    loss = 0

    for joint_idx in range(num_joints):
      
      heatmap_pred = output[:,joint_idx,:,:]
      heatmap_target = target[:,joint_idx,:,:]

      target_weight = target_weights[:,joint_idx].unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1)

      if self.use_visibility_weight:
        loss +=self.mse(target_weight * heatmap_pred, target_weight* heatmap_target) # multiplication will broadcast 
      else:
        loss +=self.mse(heatmap_pred, heatmap_target)

      # Penalize for predicting zero heatmap for a visible joint


    # average loss per joint because number of joints in images are variable
    return loss / num_joints 
  
