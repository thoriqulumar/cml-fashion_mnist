import tensorflow as tf
from tensorflow.datasets import fashion_mnist
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn




(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#model tensorflow expect data to be 4 dimensional (batch_size, width, height, channel)
x_train = np.expand_dims(x_train, axix=1)
x_test = np.expand_dims(x_test, axix=1)

#Rescale 
x_train = x_train/255.0
x_test = x_test/255.0

model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
	    tf.keras.layers.MaxPooling2D(2, 2),
	    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	    tf.keras.layers.MaxPooling2D(2,2),
	    tf.keras.layers.Flatten(),
	    tf.keras.layers.Dense(128, activation='relu'),
	    tf.keras.layers.Dense(10, activation='softmax')
	])


log_dir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


model.fit(x_train,
          y_train,
          epochs=20,
          verbose=0, # Suppress chatty output
          callbacks=[tensorboard_callback, cm_callback],
          validation_data=(x_test, y_test))