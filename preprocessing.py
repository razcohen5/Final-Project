# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:45:57 2019

@author: Raz
"""
import pandas as pd
from scipy.misc import imread,imsave
import random
import numpy as np
import os
import cv2

def formatImg(image_id,format,current_folder):  ###format 0=jpg ,1=tiff ,2=tif
#    py_dir = 'C:/Users/Ofek/PycharmProjects/untitled2/Final project/data'
    if (format == 0):
        img_path = current_folder+'/' + str(image_id) + '.jpg'

    elif (format == 1):
        img_path = current_folder+'/' + str(image_id) + '.tiff'

    else:
        img_path = current_folder+'/' + str(image_id) + '.tif'

    img = cv2.imread(img_path)
    return img


def getImg(image_id,current_folder):
    
    img = formatImg(image_id,0,current_folder)
        # plt.imshow(img)
    if img is None:
            img = formatImg(image_id,1,current_folder)
            if img is None:
                img = formatImg(image_id,2,current_folder)
    return img


def imagesCrop(current_folder,newfolder,matrix,tag_idC,image_idC,p1_xC,p1_yC,p2_xC,p2_yC,p3_xC,p3_yC,p4_xC,p4_yC):
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
        
    for row in range(matrix.shape[0]):
        img = getImg(matrix.iloc[row][image_idC],current_folder)
        singleImageCrop(newfolder,img,matrix.iloc[row][tag_idC],matrix.iloc[row][p1_xC],matrix.iloc[row][p1_yC],matrix.iloc[row][p2_xC],matrix.iloc[row][p2_yC],matrix.iloc[row][p3_xC],matrix.iloc[row][p3_yC],matrix.iloc[row][p4_xC],matrix.iloc[row][p4_yC])
    
    
def singleImageCrop(newfolder,img,tag_id,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y):
    
    p1_x = min(max(0,p1_x),img.shape[1])
    p1_y = min(max(0,p1_y),img.shape[0])
    p2_x = min(max(0,p2_x),img.shape[1])
    p2_y = min(max(0,p2_y),img.shape[0])
    p3_x = min(max(0,p3_x),img.shape[1])
    p3_y = min(max(0,p3_y),img.shape[0])
    p4_x = min(max(0,p4_x),img.shape[1])
    p4_y = min(max(0,p4_y),img.shape[0])
    
    pts = np.array([[int(p1_x),int(p1_y)],
                    [int(p2_x),int(p2_y)],
                    [int(p3_x),int(p3_y)],
                    [int(p4_x),int(p4_y)]])

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    
    ## (2) make mask
    pts = pts - pts.min(axis=0)
    
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    
#resize to square    
# =============================================================================
#     dimention = max(dst2.shape[0],dst2.shape[1])
#     res = cv2.resize(dst2, dsize=(dimention, dimention), interpolation=cv2.INTER_CUBIC)
#     
#     cv2.imshow('image',res)
#     cv2.waitKey(0)
#     print(res.shape)
# =============================================================================
    cv2.imwrite(newfolder + '/' + str(tag_id) + '.jpg',dst2)
    
# =============================================================================
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.imshow('image',croped)
#     cv2.waitKey(0)
#     cv2.imshow('image',mask)
#     cv2.waitKey(0)
#     cv2.imshow('image',dst)
#     cv2.waitKey(0)
#     cv2.imshow('image',dst2)
#     cv2.waitKey(0)
# =============================================================================
    
#@@@@@@@@@@@@@@@@@@@@@@@FIRST STEP@@@@@@@@@@@@@@@@@@@@@@@
train_csv="train.csv"
dataset = pd.read_csv(train_csv)
dataset = dataset.drop(
        dataset.columns[[12,13,14,15,16,17,18,19,20,21,22,23]],axis = 1)
imagesCrop("training imagery","cropped imagery",dataset,0,1,2,3,4,5,6,7,8,9)
dataset = dataset.drop(
        dataset.columns[[1,2,3,4,5,6,7,8,9]],axis = 1)
dataset.to_csv('aftercrop.csv',index=False)
test = pd.read_csv('aftercrop.csv')

import os
cwd = os.getcwd()