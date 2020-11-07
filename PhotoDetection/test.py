# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:54:19 2019

@author: ocasciotti
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classes = ['five','twenty','fifty']

datadir = os.getcwd() + '/dataset'

batch_size = 30
IMG_SHAPE = (480,640) 

image_gen = ImageDataGenerator(
                    rescale=1./255, 
                    rotation_range=45, 
                    horizontal_flip=False, 
                    )

train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size, 
                                                directory=datadir, 
                                                shuffle=True, 
                                                target_size=IMG_SHAPE,
                                                class_mode='sparse'
                                                )

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    


model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE[0],IMG_SHAPE[1], 3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 60

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
)

acc = history.history['accuracy']

loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

cap1 = cv2.VideoCapture(1)


picture = 1

while True:
    ret1, frame1 = cap1.read()
    cv2.imshow('frame1' , frame1)
    
    
    key = cv2.waitKey(1) & 0xFF
    if ret1 and key == ord('p'):
        frame1 = np.expand_dims(frame1, axis=0)
        guess = np.argmax(model.predict(frame1))
        print(classes[guess])
        picture += 1
    elif key == ord('q'):
        cap1.release()
        cv2.destroyAllWindows()
        break
