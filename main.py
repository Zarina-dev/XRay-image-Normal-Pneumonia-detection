import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

"""
1. normal data  balansing      O <- downloaded normal xray images from other site (given in sources.txt)

2. [data_prepare.py ] data prepare --> pickle data     O 

3. [model_building.py] model building --> pickle model  0  TEST LOSS : 0.10522958788954535 ||| TEST ACCURACY : 0.9661303758621216

[main.py ]

4. import pickled things            O

5. preprocess pickled data          O  (splitting X,y ..)

6. model.predict               O           


9. upload to GitHub

10. Use for RESUME     | YEP |

 
"""

""" PROBLEMS SOLVINGS: 
[ERROR]  Allocation of 51840000 exceeds 10% of system memory:
        Changed:    batch_size 40    --> 10 (while fitting data into model)
                    img_size (60,60) --> (55, 55)

"""

def pickle_out():

    with open('xray_val_data', 'rb') as f:
        xray_val_data = pickle.load(f)
        f.close()

    model =keras.models.load_model('xray_model_pickled')


    return xray_val_data, model

xray_val , model = pickle_out()



print('xray_val.shape : ', np.array(xray_val).shape)
print(model.summary())

print('=============================  PICKLED ! ====================')

print('=============================  SPlit valid_data into X, y  ====================')

X = []
y = []

for d in xray_val:
    X.append(d[0])
    y.append(d[1])

X = np.array(X)
y = np.array(y)

print('X shape: ', X.shape)
print('y shape: ', y.shape, )

print('=============================  PREDICT ====================')

y_pred = model.predict(X)
y_pred = np.rint(y_pred)



print('y_real:  ', y )
print('y_pred: ',np.array(y_pred).reshape(1,-1))



def plot_val_predicted():
    try:
        plt.figure(figsize=(18,18))


        for i, img in enumerate(X):
            plt.subplot(4,4,i+1)
            plt.imshow(np.array(img).reshape(55,55), cmap = 'gray')
            plt.title(f'y_real: {y[i]} \n y_pred: {y_pred[i]}')

        plt.tight_layout(pad=3.5)

        plt.savefig('plot_valid_predicted')
        plt.show()
    except:
        print('something went wrong while plotting')

plot_val_predicted()