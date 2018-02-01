import os
import csv

# Read in driving_log csv file
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)    # skip the headers
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32, correction=[0.0, 0.2, -0.2]):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    bgr_image = cv2.imread(name) 
                    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    image_flipped = np.fliplr(image)    # flip image
                    angle = float(batch_sample[3]) + correction[i]    # apply angle correction for side camera
                    angle_flipped = -angle
                    images.append(image)
                    images.append(image_flipped)
                    angles.append(angle)
                    angles.append(angle_flipped)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Training model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(p=0.5))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(p=0.5))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(p=0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')