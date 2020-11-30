
import numpy as np
import pathlib
from PIL import Image
from config import *


def getAllImagePath(name):
    path_root = pathlib.Path(name)
    result = list(path_root.glob("*"))
    return np.array(result)

def getMaskListPaths(name):
  path_root = pathlib.Path(name)
  mask_paths = list(path_root.glob("*"))
  return np.array(mask_paths)


def getDataset(all_image_paths):
  train = []
  for path in all_image_paths:
    path = str(path)
    image = Image.open(path)
    image = image.resize((image_size,image_size),Image.BILINEAR)
    train.append(np.array(image))
  return np.array(train)

def getMaskList(mask_paths):
  mask_list = []
  for path in mask_paths:
    path = str(path)
    image = Image.open(path)
    image = image.resize((image_size,image_size),Image.BILINEAR)
    image  =np.array(image)/255.
    for i in range(image_size):
    	for j in range(image_size):
    		if(image[i,j]<0.5):
    			image[i,j]=0
    		else:
    			image[i,j]=1
    mask = (1-np.array(image))[...,np.newaxis]
    if(mask_channel != 1): # 1 or 3
        mask = np.concatenate([mask,mask,mask],axis=-1)
    mask_list.append(mask)
  return np.array(mask_list)