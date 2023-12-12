import datetime
import torch
import math

def validate(model, optimizer, loss_func, data_loader, device):
  start_time = datetime.timedelta()
  total_loss = 0
  num_correct_joints = 0
  num_predicted_joints = 0
  num_true_joints = 0
  prev_percent = -1
  for batch_idx, (imgs, labels, _) in enumerate(data_loader):
    imgs = imgs.to(device)
    labels = labels.float()
    labels = labels.to(device)

    optimizer.zero_grad()

    output = model(imgs, 'val') # -> (5, H/4, W/4, #joints) 

    loss = loss_func(output, labels.float())
    loss.backward()
    total_loss += loss

    for i in range(labels.shape[0]): # for each image in batch
      for j in range(labels.shape[1]): # for each joint in predictions
        true_joint = labels[i][j]
        pred_joint = output[i][j]
        if torch.max(true_joint) > .1:
          num_true_joints += 1
          if torch.max(pred_joint) < .1:
            break
          num_predicted_joints += 1
          true_coor = (true_joint==torch.max(true_joint)).nonzero()[0]
          pred_coor = (pred_joint==torch.max(pred_joint)).nonzero()[0]
          dist = (true_coor - pred_coor).pow(2).sum(dim=0).sqrt()
          if dist <= 8:
            num_correct_joints += 1
    percent = int(batch_idx / len(data_loader) * 100)

    if percent % 10 == 0 and percent != prev_percent:
      print(f'{percent}%')
      prev_percent = percent


  elapsed_time = datetime.datetime.now() - start_time

  precision = num_correct_joints / num_predicted_joints
  recall = num_correct_joints / num_true_joints
  f1 = 2 * precision * recall / (precision + recall)

  print(f'finished validation in {elapsed_time}')
  print(f'total loss: {total_loss}')
  print(f'precision: {precision}')
  print(f'recall: {recall}')
  print(f'F1: {f1}')
  print(f'num_correct_joints: {num_correct_joints}')
  print(f'num_predicted_joints: {num_predicted_joints}')
  print(f'num_true_joints: {num_true_joints}')