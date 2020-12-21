
import os
from utils import *
from model import *
from PIL import Image
import sys
import numpy as np
from config import *

# Test

#mask owner mask dataset
def getMask(path):
    mask = Image.open(path)
    return np.array(mask)

def test(image_path,mask_path):
    path = os.path.abspath(".") + "/result/"
    generator = getNetwork()
    all_image_paths = getAllImagePath(image_path)
    mask_paths = getAllImagePath(mask_path)
    generator.load_weights(model_path)
    if(len(all_image_paths)!= len(mask_paths)):
        print("The number of images and masks is unequal !")
        return
    i = 0
    for image_path,mask_path in zip(all_image_paths,mask_paths):
        i += 1
        image = getDataset([image_path]) / 255.
        mask = getMask(mask_path)/255.

        result = generator([tf.cast(image[np.newaxis,...],tf.float32),tf.cast(mask[np.newaxis,...],tf.float32)]).numpy()
        result_image = Image.fromarray(np.uint8(result*255))
        result_image.save(path+str(i)+".png")


if __name__ =="__main__":
    args = sys.argv
    if(len(args)<2):
        print("Input Invalid")
    else:
        test(args[0],args[1])
