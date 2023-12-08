import matplotlib.pyplot as plt
import torch
import torchvision.io
from utils.training_helpers import plot_with_joints_r

def displayHeatmaps(heatmaps):
  print('show image!!')
  for i in range(14):
    plt.imshow(heatmaps[0][i].detach().cpu().numpy())
    plt.show()

def test_plot(model, device):
  # im = torchvision.io.read_image('/Users/richardpignatiello/repos/4701/JointPoseEstimation/data/lsp/images224/resized_im00012.jpg')
  im = torchvision.io.read_image('/Users/richardpignatiello/Downloads/jpe_test/thomas.jpg')
  im = torch.unsqueeze(im, 0)
  im = im.to(device)

  joints = model(im, False)
  # displayHeatmap(joints)
  joints = joints.squeeze()
  im = im.squeeze()
  x = []
  y = []
  for joint in joints:
    # print(f"max: {(joint==torch.max(joint)).nonzero()}")
    coor = (joint==torch.max(joint)).nonzero()
    print(torch.max(joint))
    if torch.max(joint) > .3:
      x.append(int(coor[0][0] * 4))
      y.append(int(coor[0][1] * 4))
      print((int(coor[0][0] * 4), int(coor[0][1] * 4)))
  im = im.cpu()
  plot_with_joints_r(im, x, y)

def plot_many(model, device, data_loader):
  for batch_idx, (imgs, labels, path) in enumerate(data_loader):
    path = path[0]
    if batch_idx == 50:
      break
    im = torchvision.io.read_image(path)
    im = torch.unsqueeze(im, 0)
    im = im.to(mps_device)

    joints = model(im, 'pred')
    joints = joints.squeeze()
    im = im.squeeze()
    x = []
    y = []
    for joint in joints:
      # print(f"max: {(joint==torch.max(joint)).nonzero()}")
      coor = (joint==torch.max(joint)).nonzero()
      x.append(int(coor[0][0] * 4))
      y.append(int(coor[0][1] * 4))
    im = im.cpu()
    plot_with_joints_r(im, x, y)