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

def resize(dirs, path):
    """
    Searches the the directory located at path dirs and resizes iamges in this path
    """
    for item in dirs:
        if os.path.isfile(path+item):
            img = Image.open(path+item)
            print(img.size)
            f, e = os.path.splitext(path+item)
            img = img.resize((224,224), Image.ANTIALIAS)
            img.save(f + '.jpg')
            print(img.size)


