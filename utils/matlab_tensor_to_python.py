from scipy.io import loadmat
import numpy as np


def lsp_mtx_to_nparray(path):
    """
    lsp matlab joints file is just a 14x3x10000 double and is being converted to 
    the same size array in numpy
    """

    #load .mat
    # this is my current path --->  /home/mxerri/JointPoseEstimation/Data/lsp/joints.mat
    data = loadmat(path)

    matlab_array = data['joints']

    np_array = np.array(matlab_array)

    # saves the array as a .npy file
    np.save('leeds_sports_extended',np_array)
