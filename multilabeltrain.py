#@@@@@@@@@@@@@@@@@THIRD STEP@@@@@@@@@@@@@@@@@@@

# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from multilabelmodel.smallervggnet import SmallerVGGNet
from imutils import paths
import matplotlib.pyplot as plt
import imutils
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.utils import plot_model
from keras.models import load_model

# =============================================================================
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset (i.e., directory of images)")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to output label binarizer")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output accuracy/loss plot")
# args = vars(ap.parse_args())
# =============================================================================

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (53, 53, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset3")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []
counter = 0
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)
    counter+=1
    print(counter)


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

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
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint
#Save the checkpoint in the /output folder
filepath = "multilabeloutput/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',#change from val acc
                            verbose=1,
                            save_best_only=True,
                            mode='min')

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),#
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("multilabeloutput/mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()

# save model architecture
plot_model(model, to_file='multilabeloutput/model.png')

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
plt.savefig("multilabeloutput/training.png")

#test model
loadedmodel = load_model('multilabeloutput/weights.93-0.14.hdf5')
mlb = pickle.loads(open("multilabeloutput/mlb.pickle", "rb").read())
for i in range(200):
#    pred = loadedmodel.predict(testX[i:i+1])
#    print(pred_to_name(pred),label_to_name(testY[i]),eval_pred(pred,testY[i]))
    print("[INFO] classifying image...")
    proba = loadedmodel.predict(testX[i:i+1])[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (label, p) in zip(mlb.classes_, proba):
    	print("{}: {:.2f}%".format(label, p * 100))
    print("actual label: ",mlb.classes_[np.flip(np.argsort(testY[i]))[:3]])
    
#NO AUG
# =============================================================================
# H = model.fit(
#  	x=trainX, y=trainY, batch_size=BS,
#  	validation_data=(testX, testY),#
#  	epochs=EPOCHS, verbose=1, callbacks=[checkpoint])
# =============================================================================
#@@@@@@@@@@@@@@@@DARIHH@@@@@@@@@@@@@@@@@@@@
name_labels=["black",
    "blue",
    "green",
    "hatchback",
    "jeep",
    "large",
    "lighttruck",
    "minivan",
    "other",
    "pickup",
    "red",
    "sedan",
    "silver",
    "small",
    "truck",
    "van",
    "white",
    "yellow"]
    
def pred_to_name(pred):
    sized = [5,13]
    subd=[3,4,6,7,8,9,11,14,15]
    colord=[0,1,2,10,12,16,17]
    sorted_by_args=np.flip(pred.argsort()[0],0)
    to_return=[]
    size_check=False
    subd_check=False
    colord_check=False
    for prob in sorted_by_args:
        if not size_check:
            if prob in sized:
                to_return.append(name_labels[prob])
                size_check=True
        if not subd_check:
            if prob in subd:
                to_return.append(name_labels[prob])
                subd_check=True
        if not colord_check:
            if prob in colord:
                to_return.append(name_labels[prob])
                colord_check=True
    return to_return

def eval_pred(pred,trues):
    
    sized = [5, 13]
    subd = [3, 4, 6, 7, 8, 9, 11, 14, 15]
    colord = [0, 1, 2, 10, 12, 16, 17]
    
    sorted_by_args=np.flip(pred.argsort()[0],0)
    size_check=False
    subd_check=False
    colord_check=False
    to_return=np.zeros(18).astype(np.uint8)
    for i, prob in enumerate(sorted_by_args):
        if not size_check:
            if prob in sized:
                to_return[prob]=1
                size_check=True
        if not subd_check:
            if prob in subd:
                to_return[prob]=1
                subd_check=True
        if not colord_check:
            if prob in colord:
                to_return[prob]=1
                colord_check=True
    return all(trues==to_return)

def label_to_name(lab):
   args=np.flip(lab.argsort(), 0)[:3].astype(np.uint8)    
   return (name_labels[args[0]],name_labels[args[1]],name_labels[args[2]])
a= [[1,3,2]]
b=np.flip(np.argsort(a[0]))[:3]
print(mlb.classes_)
from keras.models import load_model
#loadedmodel = load_model('output/1size3hotmodel.hdf5')
#loadedmodel = load_model('output/againmultilabel.hdf5')
loadedmodel = load_model('multilabeloutput/weights.93-0.14.hdf5')
mlb = pickle.loads(open("multilabeloutput/mlb.pickle", "rb").read())
for i in range(200):
#    pred = loadedmodel.predict(testX[i:i+1])
#    print(pred_to_name(pred),label_to_name(testY[i]),eval_pred(pred,testY[i]))
    print("[INFO] classifying image...")
    proba = loadedmodel.predict(testX[i:i+1])[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (label, p) in zip(mlb.classes_, proba):
    	print("{}: {:.2f}%".format(label, p * 100))
    print("actual label: ",mlb.classes_[np.flip(np.argsort(testY[i]))[:3]])
#    pred = loadedmodel.predict(trainX[i:i+1])
#    print(pred_to_name(pred),label_to_name(trainY[i]),eval_pred(pred,trainY[i]))
    
#for i in range(10000): 
#    prediction = loadedmodel.predict(trainX[i:i+1]).argsort()[0][-3:]
#    labelsprediction = []
#    for value in prediction:
#        labelsprediction.append(labelsdic.get(value))
#    print(labelsprediction,trainY[i].argsort()[-3:])
    
# =============================================================================
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(args["model"])
# 
# # save the multi-label binarizer to disk
# print("[INFO] serializing label binarizer...")
# f = open(args["labelbin"], "wb")
# f.write(pickle.dumps(mlb))
# f.close()
# =============================================================================



#@@@@@@@@@@@@@@@@@@@@NEW IMAGESSSSSSSSSSSSSS@@@@@@@@@@@@@@@@@@@@@@@@@
# load the image
image = cv2.imread("dataset/large_truck_blue/13001.jpg")
output = imutils.resize(image, width=400)
#output = imutils.resize(image, width=400)
cv2.imshow("image",image)
cv2.waitKey(0)
# pre-process the image for classification
image = cv2.resize(image, (53, 53))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:3]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
    
for i in range(50): 
    print(model.predict(trainX[i:i+1]).argsort()[0][-3:],trainY[i].argsort()[-3:])





