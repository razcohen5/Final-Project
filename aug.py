from keras_preprocessing.image import ImageDataGenerator,img_to_array
from load_and_train import *
import numpy as np
import matplotlib.pyplot as plt
x_train,size_train_label,sub_train_label,color_train_label,x_test,size_test_label,sub_test_label,color_test_label=load_data().load()

inedx_size=np.where(size_train_label==0)[0]
large_x=[]
for i in inedx_size:
    large_x.append(x_train[i])
large_x=np.array(large_x)

image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=.20,
    height_shift_range=.20,
    horizontal_flip=True)

for img in large_x:
    i = 0
    repeat=21
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    for batch in image_gen.flow(x, batch_size=1,
                              save_to_dir='preview', save_format='jpg'):
        i += 1
        if repeat < i:
            break

