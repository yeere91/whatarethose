import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

# path to the model weights file.
weights_path = './shoes_more_data.h5'

train_data_dir = 'classes/train'
validation_data_dir = 'classes/validation'
nb_train_samples = 2044
nb_validation_samples = 1000
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

## Cycle through Jordans new data to test the model
for i in range(0,len(list(os.listdir(jdir)))):
  img = load_img(jdir + '/' + os.listdir(jdir)[i]) 
  img = img.resize((150,150)) 
  x = img_to_array(img)
  x = x.reshape(1,3,150,150)
  test_images.append((os.listdir(jdir)[i], x))
test_images.sort()

predictions=[]
## For image each in the array of new data images, make a prediction and print!
print "-" * 50
print "PREDICTING JORDANS:"
print "-" * 50
for image in test_images:
  predictions.append(model.predict(image[1]))
  print "Prediction for " + image[0] + str(model.predict(image[1]))

print "Predicted 0:" + str(len(predictions) - sum(predictions))
print "Predicted 1:" + str(sum(predictions))
print "Total Predictions:" + str(len(predictions))
print "Accuracy:" + str((len(predictions) - sum(predictions))/len(predictions))



test_images = []
## Cycle through Nikes new data to test the model
for i in range(0,len(list(os.listdir(ndir)))):
  img = load_img(ndir + '/' + os.listdir(ndir)[i]) 
  img = img.resize((150,150)) 
  x = img_to_array(img)
  x = x.reshape(1,3,150,150)
  test_images.append((os.listdir(ndir)[i], x))
test_images.sort()

print "-" * 50
print "PREDICTING NIKES:"
print "-" * 50

predictions = []
## For each image in the array of new data images, make a prediction and print!
for image in test_images:
  predictions.append(model.predict(image[1]))
  print "Prediction for " + image[0] + str(model.predict(image[1]))

print "Predicted 0:" + str(len(predictions) - sum(predictions))
print "Predicted 1:" + str(sum(predictions))
print "Total Predictions:" + str(len(predictions))
print "Accuracy:" + str(sum(predictions)/len(predictions))


