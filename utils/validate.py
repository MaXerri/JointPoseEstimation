import datetime

def validate(model, optimizer, loss_func, data_loader, device):
  start_time = datetime.timedelta()
  total_loss = 0
  for batch_idx, (imgs, labels, _) in enumerate(data_loader):
    imgs = imgs.to(device)
    labels = labels.float()
    labels = labels.to(device)

    optimizer.zero_grad()

    output = model(imgs, 'val') # -> (5, H/4, W/4, #joints) 

    loss = loss_func(output, labels.float())
    
    loss.backward()

    total_loss += loss

  elapsed_time = datetime.datetime.now() - start_time

  print(f"finished validation in {elapsed_time}")
  print(f"loss: {total_loss}")