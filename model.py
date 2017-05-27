#Generic imports
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#TF & sklearn imports
import tensorflow as tf 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Keras imports
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU, Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam

######## 1. Helper functions ##########
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

#This is a helper function to find the proper full image path because of discrepency between
#udacity's data image path and simulator generated data image path
def get_image_name(input_image_path):
    image_name = 'invalid'

    if 'home' not in input_image_path:
        image_name = './data/'+input_image_path
    else:
        image_name = input_image_path

    return image_name

def data_generator(input_data, batch_size=32):
    num_samples = len(input_data)
    
    while True: 
        shuffle(input_data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = input_data[offset:offset+batch_size]

            images = []
            angles = []
            count  = 0

            for batch_sample in batch_samples:
             
                center_name  = get_image_name(batch_sample[0])
                center_image = mpimg.imread(center_name)
                center_angle = float(batch_sample[3])

                #For every odd frame, flip image and make steering angle negative
                #to simulate right turn. 
                if count % 2 == 0:
                    center_image = np.fliplr(center_image)
                    center_angle = -1*center_angle

                images.append(center_image)
                angles.append(center_angle)

                #Add left camera image with steering angle to simulate return to center
                name_left  = get_image_name( batch_sample[1].strip() )
                left_image = mpimg.imread(name_left)
                left_angle = float(batch_sample[3]) +  0.230
                images.append(left_image)
                angles.append(left_angle)

                #Add right camera image with steering angle to simulate return to center
                name_right  = get_image_name( batch_sample[2].strip() )
                right_image = mpimg.imread(name_right)
                right_angle = float(batch_sample[3]) -  0.230
                images.append(right_image)
                angles.append(right_angle)

                count = count + 1

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)

def resize_image(image):
    import tensorflow as tf 
    return tf.image.resize_images(image, (40, 160))

######## Helper functions end ##########

sample_set = []

# 1. Add udacity's data and my data to sample set and split them into training & test set
sample_set = add_csv_to_sample_set('mydata/driving_log.csv', sample_set)    
sample_set = add_csv_to_sample_set('data/driving_log.csv', sample_set)

print("sample set amount of data: " + str(len(sample_set)))

training_set, validation_set = train_test_split(sample_set, test_size=0.1)

# 2. Model definition
model = Sequential()
#Crop 20 pix from bottom and 70 pix from top to filter out unneeded areas of image for training
model.add(Cropping2D(cropping=((70, 20), (0, 0)), dim_ordering='tf', input_shape=(160, 320, 3)))

#Resize to 160 x 40
model.add(Lambda(resize_image))

# Normalisation
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# NOTE: Model is copied from ai.comma's model that can be found here: https://github.com/commaai/research/blob/master/train_steering_model.py
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(1))

# 3. Compile
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

# 4. Model training
batch_size = 32
nb_epoch   = 7

training_set_generator   = data_generator(training_set, batch_size=batch_size)
validation_set_generator = data_generator(validation_set, batch_size=batch_size)

model.fit_generator(training_set_generator,  
	            validation_data=validation_set_generator,
	            samples_per_epoch=len(training_set),
	            nb_val_samples=len(validation_set), nb_epoch=nb_epoch)

# 5. Save the model
model.save('model.h5')
