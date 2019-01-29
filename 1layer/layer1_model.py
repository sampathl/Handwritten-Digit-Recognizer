from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np 
from numpy import genfromtxt

train_file = '../train.csv'
test_file = '../test.csv'
model_file = 'layer_1_model.json'
model_weights_file = 'layer_1_model_weights.hdf5'

image_size = None

def load_training_data(file_name):
    raw_data = genfromtxt(file_name, delimiter=',', skip_header=1)
    raw_sample_image = raw_data[0][1:]
    image_size = int(np.sqrt(len(raw_sample_image)))
    print(image_size)
    X_shape = (len(raw_data), image_size, image_size, 1)
    y_shape = (len(raw_data), 10)
    X_data = np.zeros(X_shape)
    y_data = np.zeros(y_shape)
    for index, datum in enumerate(raw_data):
        X_data[index] = np.array(datum[1:]/255).reshape(image_size, image_size, 1)
        y_data[index] = np_utils.to_categorical(int(datum[0]), 10)        
    return X_data, y_data

print("Loading training data")
X_train, y_train = load_training_data(train_file)
print("Loaded training data")
