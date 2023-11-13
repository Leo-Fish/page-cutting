from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

def plot_graph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def save_result(model,history):
    base_dir = 'C:\projects\machine learning'

    # Create a unique directory name based on accuracy and current time
    accuracy = history.history['val_acc'][-1]  # Get the last validation accuracy
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    dir_name = f"model_acc{accuracy:.4f}_{current_time}"

    # Full directory path
    full_dir_path = os.path.join(base_dir, dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(full_dir_path, exist_ok=True)

    # Save the trained model
    model.save(os.path.join(full_dir_path, 'trained_model.h5'))
    
    with open(os.path.join(full_dir_path, 'training_history.pkl'), 'wb') as history_file:
        json.dump(history.history, history_file)

# Path to your data directory
base_dir = "C:/Users/86189/Desktop/single-double/"
train_dir = os.path.join(base_dir, 'train') 
validation_dir = os.path.join(base_dir, 'val')

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                target_size=(150, 150), 
                                                batch_size=20,
                                                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,  # Path to the validation images
                                                        target_size=(150, 150),
                                                        batch_size=19,
                                                        class_mode='binary')

# Load the model
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# Add the dense layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Set the trainable layers
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=15,
    epochs=80,
    validation_data=validation_generator,
    validation_steps=5)

plot_graph(history)

save_result(model,history)

