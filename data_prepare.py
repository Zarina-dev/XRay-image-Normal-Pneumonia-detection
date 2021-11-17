import numpy as np
import os
import cv2
import  tensorflow as tf
import random
import pickle


from tensorflow.keras.preprocessing import image

print('========================= START ============================')

IMG_SIZE = 55
PATH_FOLDER_TRAIN = r'chest_xray/train'
PATH_FOLDER_VAL = r'chest_xray/val'




def read_train_images():

    dataset_train = []

    for i, category in enumerate(os.listdir(PATH_FOLDER_TRAIN)):
        path_category = os.path.join(PATH_FOLDER_TRAIN, category)

        for j, img in enumerate(os.listdir(path_category)):

            try:
                imgs = cv2.imread(os.path.join(path_category,img), cv2.IMREAD_GRAYSCALE)
                imgs = cv2.resize(imgs, (IMG_SIZE, IMG_SIZE))
                imgs = image.img_to_array(imgs)
                dataset_train.append([imgs, i])
            except:
                print(f'{category}- {j} passed!')

            print(f'{category}- {j} done!')


    print('READ 완료!')
    print(' ============================= [INFO] SHUFFLING... ================')
    random.shuffle(dataset_train)



    return dataset_train



def read_val_images():

    dataset_val = []

    for i, category in enumerate(os.listdir(PATH_FOLDER_VAL)):
        path_category = os.path.join(PATH_FOLDER_VAL, category)

        for j, img in enumerate(os.listdir(path_category)):

            try:
                imgs = cv2.imread(os.path.join(path_category,img), cv2.IMREAD_GRAYSCALE)
                imgs = cv2.resize(imgs, (IMG_SIZE, IMG_SIZE))
                imgs = image.img_to_array(imgs)
                dataset_val.append([imgs, i])
            except:
                print(f'{category}- {j} passed!')

            print(f'{category}- {j} done!')


    print('VALIDATION FOLDER READ 완료!')
    print(' ============================= [INFO] SHUFFLING... ================')
    random.shuffle(dataset_val)



    return dataset_val



dataset_train = read_train_images() # <-------------------------

print(' ============ *** ============== [INFO] VALIDATION FOLDER =========== **================  ')

dataset_val = read_val_images()     # <-------------------------


print("dataset_train.shape : \n ", np.array(dataset_train).shape)
print("dataset_val.shape : \n ", np.array(dataset_val).shape)

print("*************** \n dataset_train[: 10] : ", dataset_train[:10])
print("*************** \n dataset_val[: 10] : ", dataset_val[:10])


print(' ============================= [INFO] PICKLING... ================')

with open('xray_train_data','wb') as f:
    pickle.dump(dataset_train, f)
    f.close()
    print('train data pickle - DONE')

with  open('xray_val_data','wb') as f:
    pickle.dump(dataset_val, f)
    f.close()
    print('valid data pickle - DONE')
