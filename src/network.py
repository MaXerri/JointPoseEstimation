import torch
import numpy as np
import sys
import os


sys.path.append('../JointPoseEstimation')
from models.pose_estimation_model import TransformerPoseModel
from PIL import Image 
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision import transforms
from matplotlib import pyplot as plt

def instantiate():
    model = TransformerPoseModel(12, pretrained_model="GoogleViT")

    lora_config  = LoraConfig(
        r = 8,
        lora_alpha = 8,
        target_modules = ['query', 'value'],
        lora_dropout = 0.1,
        bias = "none"
    )

    base_model = get_peft_model(model, lora_config)
    base_model.load_state_dict(torch.load("base_model_10_20_24.pth"))
    lora_model = PeftModel(base_model, "lora_weights_10_20_24")

    lora_model.eval()
    return lora_model

def run_model(image_path):

    model = instantiate()

    #img preprocessing, 
    image = Image.open(image_path)
    image224 = image.resize((224, 224)).unsqueeze(0) # resize to 224x224 and add batch dimension of 1
    preprocess = transforms.Compose([
        transforms.ToTensor(), # convert pixels to range [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalization
    ])
    
    img_transformed = preprocess(image224)

    output = model(img_transformed, True)

    image_print = image.swapaxes(0, 1).swapaxes(1, 2)

    fig, ax = plt.subplots()

    ax.imshow(image_print)
    target_res = (image.shape[0], image.shape[1])

    for i in range(14):
        z = np.unravel_index(output[0,i,:,:].argmax(), output[0,i,:,:].numpy().shape)
        ax.scatter(z[1] * (target_res[0] / 56), z[0] * (target_res[1]/ 56) ) # rescale coordinates to original image size

    ax.axis('off') # remove saxes, ticks, and labels

    return fig, ax


    






