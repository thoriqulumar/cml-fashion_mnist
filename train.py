import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import io
import itertools 
from sklearn.metrics import confusion_matrix



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#model tensorflow expect data to be 4 dimensional (batch_size, width, height, channel)
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

print(x_train.shape)
print(x_test.shape)
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


def plot_confusion_matrix(cm, class_names): 
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('histogram.png')

    return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    
    plt.savefig(buf, format='png')
    
    plt.close(figure)
    buf.seek(0)
    
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    image = tf.expand_dims(image, 0)
    
    return image


def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(x_test)
    
    test_pred = np.argmax(test_pred_raw, axis=1)
    
    cm = confusion_matrix(y_test, test_pred)
    
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


log_dir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,
		          y_train,
		          epochs=5,
		          verbose=1, # Suppress chatty output
		          callbacks=[tensorboard_callback, cm_callback],
		          validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('plot_training.png')