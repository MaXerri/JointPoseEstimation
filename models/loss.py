import torch.nn as nn

class JointsMSELoss(nn.Module):
  def __init__(self):
    super(JointsMSELoss, self).__init__()
    self.mse = nn.MSELoss(size_average=True)
    
  def forward(self, output, target):
    batch_size = output.shape[0]
    num_joints = output.shape[1] # get #joints for this image of (batch_size, num_joints, height, width)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1) # reshape heatmap predictions
    heatmaps_target = target.reshape((batch_size, num_joints, -1)).split(1, 1) # reshape target values

    loss = 0

    for joint_idx in range(num_joints):
      heatmap_pred = heatmaps_pred[joint_idx].squeeze(1)
      heatmap_target = heatmaps_target[joint_idx].squeeze(1)

      loss += 1 * self.mse(heatmap_pred, heatmap_target)
    
    # average loss per joint because number of joints in images are variable
    return loss / num_joints 