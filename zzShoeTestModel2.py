import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

# path to the model weights file.
weights_path = './vgg16_weights.h5'
my_weights_path = './transfer_train_model_shoes.h5'

train_data_dir = 'classes/train'
validation_data_dir = 'classes/validation'

# dimensions of our images.
img_width, img_height = 150, 150

datagen = ImageDataGenerator(rescale=1./255)

## Build the Network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')


train_data = np.load(open('bottleneck_features_train_1.npy'))
# train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

# validation_data = np.load(open('bottleneck_features_validation_1.npy'))
# validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

my_model = Sequential()
my_model.add(Flatten(input_shape=train_data.shape[1:]))
my_model.add(Dense(256, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(1, activation='sigmoid'))

my_model.load_weights(my_weights_path)
model.add(my_model)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print "Transfer Training weights loaded!"

# Predict on an Image
import os
from PIL import Image
from sklearn.decomposition import pca
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
import scipy.misc

jdir ='classes/NEWDATA/jordans'
ndir ='./classes/NEWDATA/nikes'

test_images = []

# generator = datagen.flow_from_directory(
#             jdir,
#             target_size=(img_width, img_height),
#             batch_size=32,
#             class_mode=None,
#             shuffle=False)

## Cycle through Jordans new data to test the model
for i in range(0,len(list(os.listdir(jdir)))):
  img = load_img(jdir + '/' + os.listdir(jdir)[i]) 
  img = img.resize((150,150)) 
  x = img_to_array(img)
  x = x.reshape(1,3,150,150)
  test_images.append((os.listdir(jdir)[i], x))

test_images.sort()
# print "generated!"
# print generator.next()[0]

# im = Image.fromarray(generator.next()[0])
# im.save("your_file.jpeg")

# scipy.misc.imsave('outfile.jpg', generator.next()[0])
## print model.predict_on_batch(generator.next())
# print model.predict(generator)


## For image each in the array of new data images, make a prediction and print!
predictions=[]
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



# test_images = []
# ## Cycle through Nikes new data to test the model
# for i in range(0,len(list(os.listdir(ndir)))):
#   img = load_img(ndir + '/' + os.listdir(ndir)[i]) 
#   img = img.resize((150,150)) 
#   x = img_to_array(img)
#   x = x.reshape(1,3,150,150)
#   test_images.append((os.listdir(ndir)[i], x))
# test_images.sort()

# print "-" * 50
# print "PREDICTING NIKES:"
# print "-" * 50

# predictions = []
# ## For each image in the array of new data images, make a prediction and print!
# for image in test_images:
#   predictions.append(model.predict(image[1]))
#   print "Prediction for " + image[0] + str(model.predict(image[1]))

# print "Predicted 0:" + str(len(predictions) - sum(predictions))
# print "Predicted 1:" + str(sum(predictions))
# print "Total Predictions:" + str(len(predictions))
# print "Accuracy:" + str(sum(predictions)/len(predictions))


