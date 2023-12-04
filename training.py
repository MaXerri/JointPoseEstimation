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
    print(annot_resize.shape)

    # heatmap annotations are converted in the dataloader, otherwise we can change pagetable size 
    # to accomodate a larger array to pre-load 

    # create officia datasets and dataloaders for training
    dataset = LSPDataset(annot_resize,"/home/mxerri/JointPoseEstimation/Data/lsp/images224/")
    dataset_mini = torch.utils.data.Subset(dataset,list(range(0,20)))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_loader_mini = DataLoader(dataset_mini, batch_size=5, shuffle=False)
        
    model = TransformerPoseModel(2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = JointsMSELoss()


    for epoch in range(1):
        total_loss = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader_mini):

            print(batch_idx)
            print(imgs.shape)
            print(labels.shape)

            optimizer.zero_grad()

            output = model(imgs) # -> (5, H/4, W/4, #joints) 

            # Heatmap dimensions are 56x56, so we need to resize at the end

            print("model output shape")
            print(output.shape)

            loss = loss_func(output, labels)
            loss = loss.item()
            
            loss.backward()
            optimizer.step()
            total_loss += loss

            
            print("breaking")
            break 

                

if __name__ == '__main__':
    main()