# RFR-Inpainting
Reproducting "Recurrent Feature Reasoning For Image Inpainting" of CVPR 2020 by tensorflow

## Environmental requirements
* tensorflow 2.0 

## Code Advantages

* Lesser useless code
* Simpler core network architecture
* Faster configuration and running

## The directory structures

**network.py** : Core network structure  
**utils.py** : Other core utils  
**config.py** : Parameter Configuration  
**run.py** : Run the network  
**test.py** : Test the network  

## Using

> python run.py /root/image_root_path/ /root/mask_root_path    
All parameters are set in config.py.

## Pretrained Models

**CeleA** : At Once
**Place2** : TODO  
**Paris Street View** ：TODO  

## Details  

Unlike the [original version](https://github.com/jingyuanli001/RFR-Inpainting), I normalize all inputs to between 0 and 1, and use the sigmoid function for the output. Because I find the author's original code hard to converge.

