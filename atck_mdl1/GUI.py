from tkinter import *
from PIL import ImageTk,Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tkinter import filedialog
#from solve_captcha import solve
from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from PIL import Image
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




model_filename = "/home/rmb/documents/FYP/final-year-project/atck_mdl1/captcha_model.hdf5"
model_labels_filename = "/home/rmb/documents/FYP/final-year-project/atck_mdl1/model_labels.dat"
input_folder = "/home/rmb/documents/FYP/final-year-project/atck_mdl1/content/Samples"

with open(model_labels_filename, "rb") as f:
    lb = pickle.load(f)

model = load_model(model_filename)

def solve(image_file):
    image = cv2.imread(image_file)
    kernel = np.ones((5,5), np.uint8)
    cv2.imshow("Input", image)
    erosion_image = cv2.erode(image, kernel, iterations=1) 
    img = cv2.dilate(image, kernel, iterations=1)
    img = Image.fromarray(img)

    predictions = []

    width, height = img.size
    left = 0
    top = 0
    right = (width-1)/4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im1 = resize_to_fit(im1, 20, 20)
   
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ###############################################################

    left = (width-1)/4
    top = 0
    right = (width-1)/6*2
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ##################################################################

    left = (width-1)/6*2
    top = 0
    right = (width-1)/6*2.7
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))

    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ######################################################################

    left = (width-1)/6*2.7
    top = 0
    right = (width-1)/6*3.3
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    #######################################################################

    left = (width-1)/6*3.3
    top = 0
    right = (width-1)/6*4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)

    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)
    print(prediction)

    captcha_text = "".join(predictions)
    return captcha_text
    cv2.waitKey()

##################
root = Tk()
root.title("CAPTCHA_SOLVER")
root.geometry("500x500")
frame = Frame(root)
frame.pack()
path = ""
def details():

    root.filename = filedialog.askopenfilename(initialdir="/content/Samples", title="Select a file")
    #my_label = Label(root, text=root.filename).pack()
    path = root.filename
    
    img = ImageTk.PhotoImage(Image.open(root.filename))
    panel = Label(root, image = img)
    panel.image = img
    panel.configure(image=img)
    panel.pack(side = "top", fill = "both", expand = "no")
    L1 = Label(frame, text = "Input image", font = 14)
    L1.pack( side = BOTTOM)
    
    solvec(path)

def solvec(path):
    text1 = solve(path)
    L4 = Label(frame, text = "Output: {}".format(text1), font = 14)
    L4.pack( side = BOTTOM)

bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM)

L2 = Label(frame, text="CAPTCHA SOLVER", font = 18)
L2.pack(side = TOP)
L5 = Label(frame, text=" ", font = 18)
L5.pack(side = TOP)

L3 = Label(frame, text="Select a CAPTCHA image", font = 14)
L3.pack (side = TOP)
L6 = Label(frame, text=" ", font = 18)
L6.pack(side = TOP)
L8 = Label(frame, text=" ", font = 18)
L8.pack(side = TOP)
B1 = Button(bottomframe, text ="Select", command = details)

B1.pack()
