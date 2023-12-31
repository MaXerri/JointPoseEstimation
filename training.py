import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from datasets.custom_datasets import LSPDataset, MPIIDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.pose_estimation_model import TransformerPoseModel
from utils.training_helpers import resize_single_joint
from utils.training_helpers import plot_with_joints
from utils.preprocessing_helpers import get_image_sizes
from models.loss import JointsMSELoss
from torch.optim import Adam
from utils.heatmap_funcs import generate_gaussian_heatmap, upsample_heatmap

def main():
    
# dataset loading
#swap axis -> (n,#joints,cooridantes)
    annot = np.load('/home/mxerri/JointPoseEstimation/Data/lsp/leeds_sports_extended.npy')
    annot_s = np.swapaxes(annot, 0,2)
    annot_s = np.swapaxes(annot_s,1,2)

    # retrieve image sizes
    image_sizes = get_image_sizes('/home/mxerri/JointPoseEstimation/Data/lsp/images/')
    image_sizes_resized = get_image_sizes('/home/mxerri/JointPoseEstimation/Data/lsp/images224/')
    annot_resize = np.zeros_like(annot_s)

    # resize annotations
    for i in range(10000):
        annot_resize[i] = resize_single_joint(annot_s[i],image_sizes_resized[i],image_sizes[i] ) 
        # make annortations into 56 x 56 for loss function 
        annot_resize[i] = resize_single_joint(annot_resize[i],(56,56),(224,224))

    print(annot_resize.shape)

    # heatmap annotations are converted in the dataloader, otherwise we can change pagetable size 
    # to accomodate a larger array to pre-load 

    # create officia datasets and dataloaders for training
    dataset = LSPDataset(annot_resize,"/home/mxerri/JointPoseEstimation/Data/lsp/images224/")
    dataset_mini = torch.utils.data.Subset(dataset,list(range(0,2000)))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    train_loader_mini = DataLoader(dataset_mini, batch_size=16, shuffle=False)


    model = TransformerPoseModel(2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = JointsMSELoss()

    print("begin training")
    for epoch in range(10):
        total_loss = 0
        count = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader_mini):

            #print(batch_idx)
            #print(imgs.shape)
            #print(labels.shape)

            optimizer.zero_grad()

            output = model(imgs) # -> (5, H/4, W/4, #joints) 

            # Heatmap dimensions are 56x56, so we need to resize at the end

            #print("model output shape")
            #print(output.shape)

            # upsample heatmap to 224 
            #output = upsample_heatmap(output, (224,224)) # check this doesnt mess w back prop

            loss = loss_func(output, labels.float())
            
            loss.backward()
            optimizer.step()
            total_loss += loss

        if epoch % 1 == 0:
            print("epoch: ", epoch, "loss: ", total_loss)

    torch.save(model.state_dict(), "/home/mxerri/JointPoseEstimation/Models/model.pt")
                

if __name__ == '__main__':
    main()