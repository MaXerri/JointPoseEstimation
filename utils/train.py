import datetime
import torch

def train(epochs, model, optimizer, loss_func, data_loader, device, verbose=True, save_name=None, save_freq=None):
    num_batches = len(data_loader)
    for epoch in range(epochs):
        start_epoch = datetime.datetime.now()
        total_loss = 0
        count = 0
        load_tensor_total = datetime.timedelta()
        model_computation_total = datetime.timedelta()
        calc_loss_total = datetime.timedelta()
        loss_backward_total = datetime.timedelta()
        optimizer_time_total = datetime.timedelta()
        for (imgs, labels, _) in data_loader:
            load_tensors_start = datetime.datetime.now()
            imgs = imgs.to(device)
            labels = labels.float()
            labels = labels.to(device)
            load_tensor_total += datetime.datetime.now() - load_tensors_start

            optimizer.zero_grad()

            computation_start = datetime.datetime.now()
            output = model(imgs, 'train') # -> (5, H/4, W/4, #joints) 
            computation_time_total = datetime.datetime.now() - computation_start

            calc_loss_start = datetime.datetime.now()
            loss = loss_func(output, labels.float())
            calc_loss_total += datetime.datetime.now() - calc_loss_start
            
            loss_backward_start = datetime.datetime.now()
            loss.backward()
            loss_backward_total += datetime.datetime.now() - loss_backward_start

            optimizer_start_time = datetime.datetime.now()
            optimizer.step()
            optimizer_time_total = datetime.datetime.now() - optimizer_start_time
            total_loss += loss

            break
            
        if verbose:
            print("epoch:", epoch, "loss: ", total_loss)
            elapsed_epoch = datetime.datetime.now() - start_epoch
            print(f"epoch {epoch} trained in: {elapsed_epoch}")
            print(f"AVG time per batch {elapsed_epoch / num_batches}")
            print(f"AVG tensor load time: {load_tensor_total / num_batches}")
            print(f"AVG prediction time: {computation_time_total / num_batches}")
            print(f"AVG loss calc time: {calc_loss_total / num_batches}")
            print(f"AVG loss backward time: {loss_backward_total / num_batches}")
            print(f"AVG optimizer step time: {optimizer_time_total / num_batches}")
            # displayHeatmap(output)
            # test_plot()
        print(epoch)
        print((epoch + 1) % save_freq == 0)
        if save_name is not None and save_freq is not None and (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f'./trained_models/{save_name}_{epoch+1}.pth')
    if save_name is not None:
        torch.save(model.state_dict(), f'./trained_models/{save_name}.pth')