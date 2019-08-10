# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from multiinputmodel import datasets
from multiinputmodel import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import imutils
import random
import pickle
import cv2
from keras.utils import plot_model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset3")))
sumheight=0
sumwidth=0
iterations=0
for imagePath in imagePaths:
    iterations+=1
    image = cv2.imread(imagePath)
    sumheight=sumheight+image.shape[0]
    sumwidth=sumwidth+image.shape[1]
avgheight=sumheight/iterations#62
avgwidth=sumwidth/iterations#60

imagePaths = sorted(list(paths.list_images("dataset3")))
heightarr=[]
widtharr=[]
iterations=0
for imagePath in imagePaths:
    iterations+=1
    image = cv2.imread(imagePath)
    heightarr.append(image.shape[0])
    widtharr.append(image.shape[1])
heightarr.sort()
widtharr.sort()
midheight=heightarr[int(iterations/2)]#62
midwidth=widtharr[int(iterations/2)]#60

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset3")))
sumheight=0
sumwidth=0
iterations=0
for imagePath in imagePaths:
    iterations+=1
    image = cv2.imread(imagePath)
    sumheight=sumheight+image.shape[0]/image.shape[1]
    sumwidth=sumwidth+image.shape[1]
avgheight=sumheight/iterations#62
avgwidth=sumwidth/iterations#60

image = cv2.imread("testresize/33h72w.jpg")
#cv2.imshow('before',image)
#cv2.imwrite("testresize/before.jpg",image)
image = cv2.resize(image, (30,30))
cv2.imwrite("testresize/after.jpg",image)
#cv2.imshow('after',image)




# =============================================================================
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, required=True,
# 	help="path to input dataset of house images")
# args = vars(ap.parse_args())
# 
# # construct the path to the input .txt file that contains information
# # on each house in the dataset and then load the dataset
# print("[INFO] loading house attributes...")
# inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
# df = datasets.load_house_attributes(inputPath)
# 
# # load the house images and then scale the pixel intensities to the
# # range [0, 1]
# print("[INFO] loading house images...")
# images = datasets.load_house_images(df, args["dataset"])
# images = images / 255.0
# 
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# print("[INFO] processing data...")
# split = train_test_split(df, images, test_size=0.25, random_state=42)
# (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
# 
# 
# # find the largest house price in the training set and use it to
# # scale our house prices to the range [0, 1] (will lead to better
# # training and convergence)
# maxPrice = trainAttrX["price"].max()
# trainY = trainAttrX["price"] / maxPrice
# testY = testAttrX["price"] / maxPrice
# 
# # process the house attributes data by performing min-max scaling
# # on continuous features, one-hot encoding on categorical features,
# # and then finally concatenating them together
# (trainAttrX, testAttrX) = datasets.process_house_attributes(df,
# 	trainAttrX, testAttrX)
# =============================================================================

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@RAZ
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@RAZ
# grab the image paths and randomly shuffle them
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS1 = (53, 53, 3)
IMAGE_DIMS2 = (53, 53, 3)
IMAGE_DIMS3 = ()

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset3")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data1 = []
data2 = []
labels = []
counter = 0
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image1 = cv2.resize(image, (53,53))
    image2 = cv2.resize(image, (70,70))
    image1 = img_to_array(image1)
    image2 = img_to_array(image2)
    data1.append(image1)
    data2.append(image2)
    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)
    counter+=1
    print(counter)


# scale the raw pixel intensities to the range [0, 1]
data1 = np.array(data1, dtype="float") / 255.0
data2 = np.array(data2, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data1.nbytes / (1024 * 1000.0)))
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data2.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
labelsdic={}
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
    labelsdic[i]=str(label)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
# =============================================================================
# (trainX, testX, trainY, testY) = train_test_split(data,
# 	labels, test_size=0.2, random_state=42)
# =============================================================================
(trainX1, testX1, trainX2, testX2, trainY, testY) = train_test_split(data1,data2,
	labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# create the MLP and CNN models
# =============================================================================
# mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
# cnn = models.create_cnn(64, 64, 3, regress=False)
# =============================================================================
cnn1 = models.SmallerVGGNet.build(
	width=53, height=53,
	depth=3, classes=len(mlb.classes_),
	finalAct="sigmoid")
cnn2 = models.SmallerVGGNet.build(
	width=70, height=70,
	depth=3, classes=len(mlb.classes_),
	finalAct="sigmoid")

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([cnn1.output, cnn2.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head

# =============================================================================
# maybe add another layer of 1000
# =============================================================================

x = Dense(1024)(combinedInput)		
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(len(mlb.classes_))(x)
x = Activation("sigmoid")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[cnn1.input, cnn2.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# =============================================================================
# opt = Adam(lr=1e-3, decay=1e-3 / 200)
# =============================================================================
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
# =============================================================================
# model.fit(
# 	[trainAttrX, trainImagesX], trainY,
# 	validation_data=([testAttrX, testImagesX], testY),
# 	epochs=200, batch_size=8)
# =============================================================================

#Save the checkpoint in the /output folder
filepath = "multiinputoutput/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',#change from val acc
                            verbose=1,
                            save_best_only=True,
                            mode='min')

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow([trainX1,trainX2], trainY, batch_size=BS),
	validation_data=([testX1,testX2], testY),#
	steps_per_epoch=(len(trainX1)+len(trainX2)) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("multiinputoutput/mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()

plot_model(model, to_file='multiinputoutput/model2.png')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("multiinputoutput/training2.png")

loadedmodel = load_model('multiinputoutput/weights.09-0.21.hdf5')
mlb = pickle.loads(open("multiinputoutput/mlb.pickle", "rb").read())
for i in range(200):
#    pred = loadedmodel.predict(testX[i:i+1])
#    print(pred_to_name(pred),label_to_name(testY[i]),eval_pred(pred,testY[i]))
    print("[INFO] classifying image...")
    proba = loadedmodel.predict([testX1[i:i+1],testX2[i:i+1]])[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (label, p) in zip(mlb.classes_, proba):
    	print("{}: {:.2f}%".format(label, p * 100))
    print("actual label: ",mlb.classes_[np.flip(np.argsort(testY[i]))[:3]])
    
# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))