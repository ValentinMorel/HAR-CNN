import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers
from keras import regularizers


segment_size = 128
num_input_channels = 3

num_training_iterations = 15000
batch_size = 128

l2_reg = 5e-4
learning_rate = 1e-4
dropout_rate = 0.03
eval_iter = 1000

n_filters = 196
filters_size = 16
n_hidden = 1024
n_classes = 6


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

def norm(x):
    temp = x.T - np.mean(x.T, axis = 0)
    #x = x / np.std(x, axis = 1)
    return temp.T


## Loading the dataset

print('Loading train data...')

# Reading training data

fx = open("data_processing/wisdm_data/data_x_" + str(segment_size) + ".csv")
fy = open("data_processing/wisdm_data/data_y_" + str(segment_size) + ".csv")
fz = open("data_processing/wisdm_data/data_z_" + str(segment_size) + ".csv")

data_x = np.loadtxt(fname = fx, delimiter = ',')
data_y = np.loadtxt(fname = fy, delimiter = ',')
data_z = np.loadtxt(fname = fz, delimiter = ',')

print('Loading done !')
fx.close(); fy.close(); fz.close();
data_train = np.hstack((data_x, data_y, data_z))


print('Loading test data...')
# Reading test data

fx = open("data_processing/wisdm_data/data_x_test_" + str(segment_size) + ".csv")
fy = open("data_processing/wisdm_data/data_y_test_" + str(segment_size) + ".csv")
fz = open("data_processing/wisdm_data/data_z_test_" + str(segment_size) + ".csv")

data_x = np.loadtxt(fname = fx, delimiter = ',')
data_y = np.loadtxt(fname = fy, delimiter = ',')

print('Loading done !')
fx.close(); fy.close(); fz.close();
data_test = np.hstack((data_x, data_y, data_z,))


print('Loading train labels....')
# Reading training labels

fa = open("data_processing/wisdm_data/answers_vectors_" + str(segment_size) + ".csv")
labels_train = np.loadtxt(fname = fa, delimiter = ',')
fa.close()

print('Loading done !')
print('Loading test labels....')
# Reading test labels

fa = open("data_processing/wisdm_data/answers_vectors_test_" + str(segment_size) + ".csv")
labels_test = np.loadtxt(fname = fa, delimiter = ',')
fa.close()
print('Loading done !')


for i in range(num_input_channels):
    x = data_train[:, i * segment_size : (i + 1) * segment_size]
    data_train[:, i * segment_size : (i + 1) * segment_size] = norm(x)
    x = data_test[:, i * segment_size : (i + 1) * segment_size]
    data_test[:, i * segment_size : (i + 1) * segment_size] = norm(x)

train_size = data_train.shape[0]
test_size = data_test.shape[0]


data_test = np.reshape(data_test, [data_test.shape[0], segment_size * num_input_channels])
labels_test = np.reshape(labels_test, [labels_test.shape[0], n_classes])
labels_test_unary = np.argmax(labels_test, axis=1)


batchSize = 16
trainSplitRatio = 1


data_trained = data_train.reshape(data_train.shape[0], 128,3,1)

#reshape of the INPUT to fit in the model

data_tested = data_test.reshape(data_test.shape[0], 128,3,1)

print('number of train samples :', data_trained.shape[0])
print('number of test samples :', data_tested.shape[0])
print('number of classes :', 6)
print(" ")
print('1 : Standing')
print('2 : Standing_up')
print('3 : Sitting')
print('4 : Downstairs')
print('5 : Upstairs')
print('6 : Walking')


model =Sequential()
model.add(Conv2D(196,kernel_size=(12,1), input_shape=(128,3,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides = (2,1), padding='valid'))
model.add(Conv2D(196,kernel_size=(12,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,1), strides = (4,1), padding='valid'))
model.add(Dropout(0.15))
model.add(Flatten())
#2nd FCL : not improving results -- twice more parameters to train
#model.add(Dense(2048, activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(6, activation='softmax'))
# Compiling the model to generate a model
adam = optimizers.Adam(lr = 0.0005, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
#for layer in model.layers:
    #print(layer.name)


model.fit(data_trained, labels_train,validation_split=1-trainSplitRatio, epochs=5,batch_size=128,verbose=1)
#evaluate with the data test - labels test
score = model.evaluate(data_tested,labels_test,verbose=1)
print('Baseline Error: %.2f%%' %(100-score[1]*100))
model.save('model_cnn.h5')
np.save('truth.npy', labels_test)
np.save('DataTest.npy', data_tested)
