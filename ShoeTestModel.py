import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

# path to the model weights file.
weights_path = './shoes1.h5'

train_data_dir = 'classes/traino'
validation_data_dir = 'classes/validationo'
nb_train_samples = 1796
nb_validation_samples = 155
nb_epoch = 50

# dimensions of our images.
img_width, img_height = 150, 150

datagen = ImageDataGenerator(rescale=1./255)

## Build the Network
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights(weights_path)    

model.compile(loss='binary_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

print "Model Loaded!"

# Predict on an Image
import os
from PIL import Image
from sklearn.decomposition import pca
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

jdir ='./classes/NEWDATA/jordans'
ndir ='./classes/NEWDATA/nikes'

test_images = []
print os.listdir(jdir)


## Cycle through Jordans new data to test the model
for i in range(0,len(list(os.listdir(jdir)))):
  img = load_img(jdir + '/' + os.listdir(jdir)[i]) 
  img = img.resize((150,150)) 
  x = img_to_array(img)
  x = x.reshape(1,3,150,150)
  test_images.append((os.listdir(jdir)[i], x))

## Cycle through Nikes new data to test the model
for i in range(0,len(list(os.listdir(ndir)))):
  img = load_img(ndir + '/' + os.listdir(ndir)[i]) 
  img = img.resize((150,150)) 
  x = img_to_array(img)
  x = x.reshape(1,3,150,150)
  test_images.append((os.listdir(ndir)[i], x))

## For each image in the array of new data images, make a prediction and print!
for image in test_images:

  print "Prediction for " + image[0] + str(model.predict(image[1]))