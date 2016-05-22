__doc__="""cat preprocessing function"""

import sys

import os
import csv
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def load_cats_dogs():
    """
    downloads dataset from A.Tvorozhkov's dropbox
    """
    urls = ['https://www.dropbox.com/s/amn4gm67ptl69i8/train.zip?dl=1']
    filenames = ['dogs_vs_cats.train.zip']
    for u, f in zip(urls,filenames):
        os.system("wget "+u+" -O "+f)
    return True


# Convert all the image files in the given path into np arrays with dimensions suitable for DL with Theano
def jpg_to_nparray(path,img_names, img_size, grayscale = False):
    X = []
    Y = []
    img_colors = 3

    for counter,img_dir in enumerate(img_names):

        # X
        img = Image.open(path+img_dir)
        img = ImageOps.fit(img, img_size, Image.ANTIALIAS)
        
        if grayscale:
            img = ImageOps.grayscale(img)
            img_colors = 1


        img = np.asarray(img, dtype = 'float32') / 255.
        img = img.reshape([img_colors]+list(img_size))
        X.append(img)

        # Y: 0 for cat, 1 for dog
        if "cat" in img_dir:
            Y.append(0)
        else:
            Y.append(1)


        # Printing
        counter+=1
        if counter%1000 == 0:
            print'processed images: ', counter

    X = np.asarray(X)
    Y = np.asarray(Y,dtype='int32')

    return (X,Y)



# Get ids of the images: we'll need them for generating the submission file for Kaggle
def get_ids(path):
    ids = np.array([],dtype = int)
    for str in os.listdir(path):
        ids = np.append(ids, int(str.partition(".")[0]))
    
    ids = np.array(ids, dtype = int)[...,None]
    return ids

def prepar_img(height,width,grayscale):
    if not os.path.exists("./dogs_vs_cats.train.zip"):
        load_cats_dogs()
    if not os.path.exists("./train"):
        os.system("unzip dogs_vs_cats.train.zip > unzip.log")
    path = ('./train/')
    img_wh = (height,width)
    image_names = os.listdir(path)
    X,y = jpg_to_nparray(path,image_names,
                         img_wh, 
                         grayscale=grayscale
                  )
    np.save("catdog_x.npy", X)
    np.save("catdog_y.npy", y)
    return X,y

def load_data(IMG_GEN_IMG, IMG_HEIGHT=60, IMG_WIDTH=60, IMG_GRAY=True):
    if IMG_GEN_IMG:
        return prepar_img(IMG_HEIGHT, IMG_WIDTH, IMG_GRAY)
    else:  
        if not os.path.exists("./dogs_vs_cats.train.zip"):
            load_cats_dogs()
        if not os.path.exists("./train"):
            os.system("unzip dogs_vs_cats.train.zip > unzip.log")
        X = np.load("catdog_x.npy")
        y = np.load("catdog_y.npy")
        return X,y
    
def prepar_data(X, y, train_size,val_size,test_size):
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size: train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]