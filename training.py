import torch
from torch import optim
import numpy as np
from datasets.custom_datasets import LSPDataset, MPIIDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.transformer_block import ViT

def main():
    
    # dataloader stuff
    annot = np.load('/home/mxerri/JointPoseEstimation/Data/lsp/leeds_sports_extended.npy')
    annot_s = np.swapaxes(annot, 0,2)
    annot_s = np.swapaxes(annot_s,1,2)

    dataset = LSPDataset(annot_s,"/home/mxerri/JointPoseEstimation/Data/lsp/images/")
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    
    # define model, optimizer and loss

    model = ViT()
    # prints out model parameters
    #for name, param in model.state_dict().items():
    #    print(name, param.size())

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

    for epoch in range(1):
        acc_loss = 0
        for img, label in (train_loader):
            # Move the batch to the device
            img, label = img.to(device) , label.to(device)
            print("hello")
            output = model()

            

if __name__ == '__main__':
    main()