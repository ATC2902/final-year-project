import os                                    #set tf compiler flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     #set tf compiler flags
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import LSTM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
#from helpers import resize_to_fit
import imutils
import cv2


def resize_to_fit(image, width, height):
    
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)

    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image


input_folder = "Letter extracted images/MA"
model_filename = "captcha_model.hdf5"
model_labels_filename = "model_labels.dat"

model = load_model(model_filename)

with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)

data = []
labels = []


for image_file in paths.list_images(input_folder):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    image = np.expand_dims(image, axis=2)
    label = image_file.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.98, random_state=0)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Predict with X_test features
y_pred = model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
Y_test=np.argmax(Y_test, axis=1)

# Compare predictions to y_test labels
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, average = "macro")
recall=recall_score(Y_test, y_pred, average="macro")
f1=f1_score(Y_test, y_pred, average="macro")

print("F1 Score : ",f1*100)
print("Recall Score : ",recall*100)
print('Accuracy :',accuracy*100)
print('Precision :',precision*100)


#plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
sn.heatmap(cm, annot=True)
plt.show()
