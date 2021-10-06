import os
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import StratifiedKFold
import glob
from keras.layers import Convolution1D, MaxPooling1D, Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential

def preprocessing():
    X = []
    y = []
    path_main = 'Output_RGB_fold/'
    os.chdir(path_main)
    dirs = filter(os.path.isdir, os.listdir(os.curdir))
    for dir in dirs:
        path_to_subdir = str(dir)
        for im_path in os.listdir(path_to_subdir):
            im_frame = Image.open(path_to_subdir + '/' + im_path)
            np_frame = np.array(im_frame.getdata()) 
            X.append(np_frame)
            y.append(path_to_subdir.split('.')[0])

    unique = list(dict.fromkeys(y))
    dct = {}
    cnt = 0
    for lab in unique:
        dct[str(lab)] = cnt
        cnt += 1
    
    nb_classes = len(dct)
    new_label = []
    for l in y:
        new_label.append(dct[l])

    y = new_label
    return X,y, nb_classes

X_data, y_data, nb_classes=preprocessing()
batch_size=64
epoch=10
X_data= np.array(X_data)
y_data = np.array(y_data)
X_data = X_data.reshape((-1, 64, 64, 3))
X_data = X_data.astype('float32')
print('data shape: {}'.format(X_data.shape))
print('data labels shape: {}'.format(y_data.shape))
print('nb_classes: {}'.format(nb_classes))
shape = X_data.shape[1:]

def create_cnn():
    model = Sequential()
    model.add(Input(shape))
    model.add(Convolution2D(8, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
tmp=1
for train_index, test_index in skf.split(X_data, y_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    print('Fold'+str(tmp)+':')
    model = create_cnn()
    history=model.fit(X_train[:], y_train[:],
          batch_size=batch_size,
          epochs=epoch,
          validation_data=(X_test[:], y_test[:]))
    print('Fold'+str(tmp)+'is finished')



   
    