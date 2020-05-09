# -*- coding:utf-8 -*-
import os,sys
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.models import load_model
import keras.backend as K
import pickle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
K.set_image_dim_ordering('tf')

class CFOCFeatureShow:
    def __init__(self):
        self.m_image_list = []
    
    def Add(self,image_list):
        self.m_image_list.append(image_list)
    
    def ShowImage(self,raw_image = []):
        img_count = 0
        for img_list in self.m_image_list:
            img_count += len(img_list)
        col_count = 10
        row_count = img_count//col_count + 2
        id = 1
        if len(raw_image) > 0:
            plt.subplot(row_count, col_count, id)
            plt.imshow(raw_image, cmap='gray')
            plt.axis('on')
            id += col_count
            
        for img_list in self.m_image_list:
            for img in img_list:
                plt.subplot(row_count, col_count, id)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                id += 1
            
        plt.show()
        
    @staticmethod
    def Show(feature_img_list,layer_id = None,raw_image=[]):
        show = CFOCFeatureShow()
        for id in range(0,len(feature_img_list)):
            if layer_id == None or layer_id == id:
                img_list = feature_img_list[id]
                show.Add(img_list)
                
        show.ShowImage(raw_image)
        
class CFOCModelParse:
    def __init__(self,model):
        self.m_model = model
        
    def Get_Layers(self):
        return self.m_model.layers
        
    def Get_Functions(self,layer_id):
        layers = self.Get_Layers()
        function = K.function([layers[0].input],[layers[layer_id].output])
        layer_name = layers[layer_id].name
        return layer_name,function
        
    def Run_Layer(self,layer_id,image):
        layer_name,function = self.Get_Functions(layer_id)
        f1 = function([image])[0]
        return layer_name,f1
        
    def Get_Features(self,layer_id,image):
        layer_name,function = self.Get_Functions(layer_id)
        if layer_name.find("conv2d_") < 0:
            return np.array([])
        f1 = function([image])[0]
        channel,h,w,count = f1.shape
        ret = []
        for _ in range(count):
            show_img = f1[:, :, :, _]
            shape = show_img.shape
            show_img.shape = [h, w]
            ret.append(show_img)
        return ret
        
#same as image preprocess before train
def pre_process_image(image):
    input_shape = (28,28,1)
    x = np.array(image)
    if input_shape[2] == 1:
        return x.reshape(x.shape[0], input_shape[0], input_shape[1], 1) 
    else:
        return x
        
def main():
    model = load_model("./data/model.h5")
    image = cv2.imread("./data/test.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = pre_process_image([image])
    
    test = CFOCModelParse(model)
    layers = test.Get_Layers()
    for layer in layers:
        name = layer.name
        input = layer.input
        output = layer.output
        print(name,input,output)
        
    feature_img_list = []
    for i in range(len(layers)):
        layer_name,out = test.Run_Layer(i,image)
        print(i,layer_name,out.shape)
        if layer_name.find("conv2d_") >= 0:
            features = test.Get_Features(i,image)
            feature_img_list.append(features)
    
    CFOCFeatureShow.Show(feature_img_list)
    
if __name__ == '__main__':
    main()
