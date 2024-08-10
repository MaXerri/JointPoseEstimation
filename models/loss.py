import torch.nn as nn
import torch

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
    return (loss / num_joints)
  

class HeatmapWingLoss(nn.Module):

  """
  Heatmap Wingloss Function adapted from VITPose and 
   Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression papers

  Params: 
     - w = 14
     - eps = 1
     - a = 2.1
     - theta = 0.5
     - use_visibility_weight: use weight based of joint visibility

  """

  def __init__(self, w = 14, eps = 1, a = 2.1, theta = 0.5, use_visibility_weight = True):
    super(HeatmapWingLoss, self).__init__()
    
    self.use_visibility_weight = use_visibility_weight
    self.w = w
    self.eps = eps
    self.a = a
    self.theta = theta
    
  def WingLoss(self, pred, target):
    """
    Wingloss function for heatmaps
    Hyperpararameters chosen from Adaptive Wingloss Paper: 
     

    Params:
      - pred: predicted heatmap
      - target: ground truth heatmap
    """
    dif = (pred - target).abs()
    h = dif[2]
    w = dif[3]

    A = self.w * (1 / (1 + torch.pow(self.theta / self.eps, self.a - target))) \
        * (self.a - target) * torch.pow(self.theta / self.eps, self.a - target - 1) \
        * (1 / self.eps)
    
    C = self.theta * A - (self.w * torch.log(1 + \
        torch.pow(self.theta / self.eps, self.a - target)))

    loss = torch.where(dif < self.theta, self.w * torch.log(1 + torch.pow(dif / self.eps, self.a - target)), A * dif - C)

    return loss.mean()


  def forward(self, output, target, target_weights):
    """
    Params: 
      - output: model output (batch_size, num_joints, height?, width)
      - target: groundtrutch heatmaps
      - target weight: (batch size, num_joints, 1): based on the visibility of the joint
            
    """
   
    loss = 0
    if self.use_visibility_weight:
      loss +=self.WingLoss(target_weights.unsqueeze(-1).unsqueeze(-1) * \
                           output, target_weights.unsqueeze(-1).unsqueeze(-1) * target) # multiplication will broadcast 
    else:
      loss +=self.WingLoss(output, target)

    return loss
  
