import torch
from torch import optim
import numpy as np
from datasets.custom_datasets import LSPDataset, MPIIDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.visual_trasnformer import ViT

def main():


    # dataloader stuff
    annot = np.load('/JointPoseEstimation/Data/lsp/leeds_sports_extended.npy')
    annot_s = np.swapaxes(annot, 0,2)
    annot_s = np.swapaxes(annot_s,1,2)

    dataset = LSPDataset(annot_s,"/JointPoseEstimation/Data/lsp/images224/")
    print(annot_s.shape)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    
    # define model, optimizer and loss

    model = ViT()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

    for epoch in range(1):
        acc_loss = 0
        for img, label in (train_loader):
            # Move the batch to the device
            img, label = img.to(device) , label.to(device)
            output = model()

            

if __name__ == '__main__':
    main()