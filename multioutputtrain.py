# USAGE
# python train.py --dataset dataset --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from multioutputmodel.fashionnet import FashionNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import tensorflow as tf

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset (i.e., directory of images)")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
#ap.add_argument("-l", "--categorybin", required=True,
#	help="path to output category label binarizer")
#ap.add_argument("-c", "--colorbin", required=True,
#	help="path to output color label binarizer")
#ap.add_argument("-p", "--plot", type=str, default="output",
#	help="base filename for generated plots")
#args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (53, 53, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
data = []
sizeLabels = []
categoryLabels = []
colorLabels = []
counter = 0
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    data.append(image)
    # extract the clothing color and category from the path and
    # update the respective lists
    (size, cat, color) = imagePath.split(os.path.sep)[-2].split("_")
    sizeLabels.append(size)
    categoryLabels.append(cat)
    colorLabels.append(color)
    counter+=1
    print(counter)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
sizeLabels = np.array(sizeLabels)
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
sizeLB = LabelBinarizer()
categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
sizeLabels = sizeLB.fit_transform(sizeLabels)
sizeLabels = tf.keras.utils.to_categorical( #labelbinarizer wont vectorize 2 class problem
    sizeLabels,
    num_classes=2,
    dtype='float32'
)
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, sizeLabels, categoryLabels, colorLabels,
	test_size=0.2, random_state=42)
(trainX, testX, trainSizeY, testSizeY, trainCategoryY, testCategoryY,
	trainColorY, testColorY) = split
 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize our FashionNet multi-output network
model = FashionNet.build(53, 53,
    numSizes=1,
	numCategories=len(categoryLB.classes_),
	numColors=len(colorLB.classes_),
	finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "size_output": "binary_crossentropy",
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"size_output": 1.0, "category_output": 1.0, "color_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

#ani hosafti
from keras.callbacks import ModelCheckpoint
#Save the checkpoint in the /output folder
filepath = "multioutputoutput/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

# train the network to perform multi-output classification
H = model.fit(trainX,
	{"size_output": trainSizeY, "category_output": trainCategoryY, "color_output": trainColorY},
	validation_data=(testX,
		{"size_output": testSizeY, "category_output": testCategoryY, "color_output": testColorY}),
	epochs=EPOCHS,
	verbose=1,
    callbacks=[checkpoint])
 
# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open("multioutputoutput/size_lb.pickle", "wb")
f.write(pickle.dumps(sizeLB))
f.close()

# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open("multioutputoutput/category_lb.pickle", "wb")
f.write(pickle.dumps(categoryLB))
f.close()
 
# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open("multioutputoutput/color_lb.pickle", "wb")
f.write(pickle.dumps(colorLB))
f.close()
    
#test model
from keras.models import load_model
loadedmodel = load_model('multioutputoutput/multioutput.hdf5',custom_objects={"tf": tf})
sizeLB = pickle.loads(open("multioutputoutput/size_lb.pickle", "rb").read())
categoryLB = pickle.loads(open("multioutputoutput/category_lb.pickle", "rb").read())
colorLB = pickle.loads(open("multioutputoutput/color_lb.pickle", "rb").read())

for i in range(testX.shape[0]):
    (sizeProba,categoryProba, colorProba) = model.predict(testX[i:i+1])
    #print(pred_to_name(pred)) #only if multilabel
    if sizeProba[0]>0.5:
        sizeIdx = 1
    else:
        sizeIdx = 0
    categoryIdx = categoryProba[0].argmax()
    colorIdx = colorProba[0].argmax()
    sizeLabel = sizeLB.classes_[sizeIdx]
    categoryLabel = categoryLB.classes_[categoryIdx]
    colorLabel = colorLB.classes_[colorIdx]
    print("prediction: ",sizeLabel,categoryLabel,colorLabel)
    print("actual label: ",sizeLB.classes_[testSizeY[i].argmax()],categoryLB.classes_[testCategoryY[i].argmax()],colorLB.classes_[testColorY[i].argmax()])
    
print(sizeLB.classes_)  



    
plot_model(model, to_file='multioutputoutput/model.png')
    
# =============================================================================
# H = model.fit_generator(
# 	aug.flow(trainX, (trainSizeY,trainCategoryY,trainColorY), batch_size=BS),
# 	validation_data=(testX, {"size_output": testSizeY,
#                           "category_output": testCategoryY,
#                           "color_output": testColorY}),
# 	steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)
# =============================================================================
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("multioutputoutput/multipleoutputnoaug2.png")   
# save the model to disk
#print("[INFO] serializing network...")
#model.save("multioutputmodel")

# save the category binarizer to disk
#print("[INFO] serializing category label binarizer...")
#f = open(args["categorybin"], "wb")
#f.write(pickle.dumps(categoryLB))
#f.close()

# save the color binarizer to disk
#print("[INFO] serializing color label binarizer...")
#f = open(args["colorbin"], "wb")
#f.write(pickle.dumps(colorLB))
#f.close()

# plot the total loss, category loss, and color loss
#lossNames = ["loss", "size_output_loss", "category_output_loss", "color_output_loss"]
#plt.style.use("ggplot")
#(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
#
## loop over the loss names
#for (i, l) in enumerate(lossNames):
#	# plot the loss for both the training and validation data
#	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
#	ax[i].set_title(title)
#	ax[i].set_xlabel("Epoch #")
#	ax[i].set_ylabel("Loss")
#	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
#	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
#		label="val_" + l)
#	ax[i].legend()
#
## save the losses figure
#plt.tight_layout()
#plt.savefig("{}_losses.png".format(args["plot"]))
#plt.close()
#
## create a new figure for the accuracies
#accuracyNames = ["size_output_acc", "category_output_acc", "color_output_acc"]
#plt.style.use("ggplot")
#(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
#
## loop over the accuracy names
#for (i, l) in enumerate(accuracyNames):
#	# plot the loss for both the training and validation data
#	ax[i].set_title("Accuracy for {}".format(l))
#	ax[i].set_xlabel("Epoch #")
#	ax[i].set_ylabel("Accuracy")
#	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
#	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
#		label="val_" + l)
#	ax[i].legend()
#
## save the accuracies figure
#plt.tight_layout()
#plt.savefig("{}_accs.png".format(args["plot"]))
#plt.close()