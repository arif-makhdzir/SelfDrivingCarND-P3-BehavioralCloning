import os
import csv
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda

#from keras.optimizers import SGD, Adam, RMSprop
from keras.optimizers import Adam
from keras.utils import np_utils

#from keras.callbacks import ModelCheckpoint
#from keras.models import model_from_json

sample_set = []

def add_csv_to_sample_set(filepath, sample_set):
    with open(filepath) as csv_file:
        first_line = True
        reader     = csv.reader(csv_file)
        
        for line in reader:
            if first_line == True:
                first_line = False
            else:
                sample_set.append(line)
    
    return sample_set

#sample_set = add_csv_to_sample_set('mydata/driving_log.csv', sample_set)    
sample_set = add_csv_to_sample_set('data/driving_log.csv', sample_set)

training_set, validation_set = train_test_split(sample_set, test_size=0.1)

print("length: " + str(len(sample_set)))
print(sample_set[0])

def generator(input_set, batch_size=32):
    num_samples = len(input_set)
    
    while True: # Loop forever so the generator never terminates
        shuffle(input_set)
        for offset in range(0, num_samples, batch_size):
            batch_samples = input_set[offset:offset+batch_size]

            images = []
            angles = []
            count  = 0
            for batch_sample in batch_samples:
             
                name = 'test'
                if 'home' not in batch_sample[0]:
                    name = './data/'+batch_sample[0]
                else:
                    name = batch_sample[0]
           
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])

                if count % 2 == 0:
                    center_image = np.fliplr(center_image)
                    center_angle = -1*center_angle

                images.append(center_image)
                angles.append(center_angle)

                name_left = 'test'
                if 'home' not in batch_sample[1]:
                    name_left = './data/'+batch_sample[1].strip()
                else:
                    name_left = batch_sample[1].strip()
                #name_left = './data/'+batch_sample[1].strip()
                left_image = mpimg.imread(name_left)
                left_angle = float(batch_sample[3]) +  0.229
                images.append(left_image)
                angles.append(left_angle)

                name_right = 'test'
                if 'home' not in batch_sample[2]:
                    name_right = './data/'+batch_sample[2].strip()
                else:
                    name_right = batch_sample[2].strip()
                #name_right = './data/'+batch_sample[2].strip()
                right_image = mpimg.imread(name_right)
                right_angle = float(batch_sample[3]) -  0.229
                images.append(right_image)
                angles.append(right_angle)

                count = count + 1

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)

train_generator      = generator(training_set, batch_size=32)
validation_generator = generator(validation_set, batch_size=32)

def resize_comma(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (40, 160))


## 3. Model (data preprocessing incorporated into model)

# Model adapted from Comma.ai model

model = Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                     dim_ordering='tf', # default
                     input_shape=(160, 320, 3)))

# Resize the data
model.add(Lambda(resize_comma))

# Normalise the data
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# Conv layer 1
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())

# Conv layer 2
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())

# Conv layer 3
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())

# Fully connected layer 1
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())

# Fully connected layer 2
model.add(Dense(50))
model.add(ELU())

model.add(Dense(1))


## Compile
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

print("Model summary:\n", model.summary())

## 4. Train model
batch_size = 128
nb_epoch = 3

# Train model using generator
model.fit_generator(train_generator, 
	            samples_per_epoch=len(training_set), 
	            validation_data=validation_generator,
	            nb_val_samples=len(validation_set), nb_epoch=nb_epoch)
	            #callbacks=[checkpointer])

## Save model
model.save('model.h5')

