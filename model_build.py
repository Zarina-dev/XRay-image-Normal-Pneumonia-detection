import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D,MaxPool2D, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split

print('============================  PICKLE OUT XRAY TRAIN DATA ============================')

def pickle_out():

    with open('xray_train_data','rb') as f:
        xray_train = pickle.load(f)

    print('pickle 완료!')
    return xray_train

xray_train = pickle_out()

print(xray_train[:120][1])

print('============================  PREPROCESSING ============================')

X = []
y = []

for sl in xray_train:
    X.append(sl[0])
    y.append(sl[1])

X = np.array(X)
y = np.array(y)

print('X shape :',X.shape) # X shape : (6240, 60, 60, 1)
print('y shape :',y.shape) # y shape : (6240,)

print('X : ',X[:10])
print('y : ', y[:20])


print('============================ PLOT 2 NORMAL, 2 PNEUMONIA ============================')

normal_index = [0,3]
pneumon_index = [2, 6]

plt.figure(figsize=(12,12))

plt.subplot(2,2,1)
plt.title('norm 1')
plt.imshow(X[0].reshape(55,55), cmap='gray') # (55,55, 1) --> (55,55)

plt.subplot(2,2,2)
plt.title('norm 2')
plt.imshow(X[3].reshape(55,55), cmap='gray')

plt.subplot(2,2,3)
plt.title('pneumonia 1')
plt.imshow(X[2].reshape(55,55), cmap='gray')

plt.subplot(2,2,4)
plt.title('pneumonia 2')
plt.imshow(X[6].reshape(55,55), cmap='gray')

plt.savefig('norm_pneumonia_plt_imgs.png')
plt.show()
print('image is saved! ')

print('============================ SPLIT ============================')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 2)

print('X_train.shape, X_test.shape, y_train.shape, y_test.shape : \n ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print('============================ BUILD ARTIFICIAL NEURAL NETWORK============================')


model = Sequential([
    Conv2D(32, input_shape=X_train.shape[1:], kernel_size=(3,3), padding='SAME', activation=tf.nn.relu),
    AvgPool2D(pool_size=(2,2), strides=(2,2), padding='SAME'),
    Conv2D(64, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu),
    AvgPool2D(pool_size=(2,2),strides=(2,2), padding='SAME'),

    Flatten(), # converts 2D into 1D, that's because Dense layer expects 1D data
    Dropout(rate=0.25),  # technique to avoid overfitting

    Dense(900, activation=tf.nn.relu),
    Dense(1, activation=tf.nn.sigmoid)

])
print(model.summary())

print('============================ ANN COMPILE  ============================')

model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('============================= TRAINING ... ==========================')

History = model.fit(X_train, y_train, batch_size = 8, epochs=4, verbose=2, validation_split=0.15)

def plot_loss_accuracy_history():
    plt.figure(figsize=(14, 4))
    plt.subplot(1,2,1)
    plt.title('LOSS ')
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.legend(['train_loss','val_loss'])
    plt.xlabel('loss')
    plt.ylabel('epochs')


    plt.subplot(1,2,2)
    plt.title('ACCURACY')
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.legend(['train_accuracy','val_accuracy'])
    plt.xlabel('accuracy')
    plt.ylabel('epochs')


    plt.savefig('plot_loss_accuracy_history.png')
    plt.show()
    print('image is saved! ')




eval_score = model.evaluate(X_test, y_test, verbose=2)

print(f'TEST LOSS : {eval_score[0]} ||| TEST ACCURACY : {eval_score[1]}')

# TEST LOSS : 0.10522958788954535 ||| TEST ACCURACY : 0.9661303758621216



model.save('xray_model_pickled')
print('save model 완료!')

plot_loss_accuracy_history()
