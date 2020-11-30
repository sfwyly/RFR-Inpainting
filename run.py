
from model import *
from utils import *
from config import *
import math
import sys
from loss import *

generator  = getNetwork()

# image_path : image root
# mask_path  : mask root
def train(image_path,mask_path):

    generator = getNetwork()
    all_image_paths = getAllImagePath(image_path)
    mask_paths = getAllImagePath(mask_path)
    holesNum = len(mask_paths)
    holeidx = [_ for _ in range(holesNum)]

    for epoch in range(epochs):
        np.random.shuffle(all_image_paths)
        np.random.shuffle(mask_paths)
        np.random.shuffle(holeidx)
        for load_train in range(
                int(np.ceil(len(all_image_paths) / (batch * batch_size)))):

            X_train = getDataset(
                all_image_paths[load_train * (batch * batch_size):(load_train + 1) * (batch * batch_size)]) / 255.

            X_labels = X_train
            id_list = [j for j in range(X_train.shape[0])]
            np.random.shuffle(id_list)
            for t in range(math.ceil(X_train.shape[0] / batch_size)):
                idx = id_list[t * batch_size:(t + 1) * batch_size]
                X_ = X_train[idx]
                #         mask_list = hole_list[np.random.randint(holesNum)][np.newaxis,...]
                mask_list = getMaskList(mask_paths[np.random.choice(holeidx, 1)])
                mask_list = np.concatenate([mask_list for _ in range(len(X_))], axis=0)
                #         print(X_.shape,mask_list.shape)
                X_ = X_ * mask_list
                loss, style_loss, L1_loss, tvl_loss, perceptual_loss = train_step(generator,X_labels[idx], X_, mask_list)  # 训练生成器

            print(load_train+1," / ",epoch+1," loss: ", loss.numpy(), style_loss.numpy(), L1_loss.numpy(), tvl_loss.numpy(),
                  perceptual_loss.numpy())
            if ((load_train + 1) % save_bacth == 0):
                s = str(epoch+1)+"_"+str(load_train+1)
                generator.save_weights("/RFR_Inpainting_"+s+".h5")

        generator.save_weights("/RFR_Inpainting_" + str(epoch) + ".h5")


if __name__ =="__main__":
    args = sys.argv
    if(len(args)<2):
        print("Input Invalid")
    else:
        train(args[0],args[1])
