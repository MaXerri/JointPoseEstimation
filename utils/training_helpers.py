import numpy as np
import matplotlib.pyplot as plt


def batched_resize_joint(joints, resize_dim, original_dim):
    """
    returns the new batched joint coordinate in the batch of images
    """
    vfunc = np.vectorize(resize_single_joint)
    final = vfunc(joints, resize_dim, original_dim)
    return final

def resize_single_joint(joints, resize_dim, original_dim):
    """
    returns the new joint coordinate in the reshaped image
    """
    new = np.zeros((14,3))
    for idx, point in enumerate(joints):
        x1 = int((point[0] * (resize_dim[0] / original_dim[0])))
        y1 = int((point[1] * (resize_dim[1] / original_dim[1])))
        new[idx,:] = [x1,y1,point[2]] # append resized coordinate
    return new

# plotting functions:
def plot_with_joints(img, joints):
    """
    Plots an image overlayed with the joints
    """

    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,0,1)
    plt.scatter(joints[:,0],joints[:,1])
    plt.imshow(img)
    plt.show()
