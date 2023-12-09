import os 
import PIL 
from PIL import Image


def get_image_sizes(folder_path):
    """
    retreive the sizes of the original images
    """
    sizes = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Iterate over files in the folder
    for filename in image_files:

      file_path = os.path.join(folder_path, filename)

      # Open the image using PIL
      with Image.open(file_path) as img:
          # Get the size of the image (width, height)
          img_size = img.size
          sizes.append(img_size)

    return sizes

def resize_images(input_folder, output_folder, new_size):
    """
    Resize all images in a folder to a given size and store in another folder that already exists.

    input folder: path to the folder containing the original images
    output folder: path to the folder where the resized images will be stored
    Size:  a size 2 tuple (width, height)
    """

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (you can add more file type checks if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                resized_img = img.resize(new_size)
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, f"re_{filename}")
                resized_img.save(output_path)


def get_list_of_image_names(path):
    """
    Returns the names of all files within a directory in alphabetical order.  This
    should only be used on folders with only images in them
    """
    # path = "/home/mxerri/JointPoseEstimation/Data/lsp/images/"
    dir_list = sorted(os.listdir(path))
    return dir_list


