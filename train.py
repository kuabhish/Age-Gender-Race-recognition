import dlib
import cv2
import glob
import keras
import tensorflow as tf
import scipy.io
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import EarlyStopping
import my_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
###########################################################################
## 1st task ## import data
data = scipy.io.loadmat('images\\data_base.mat')
images = data['image']
ages = data['age']
genders = data['gender']
races = data['race']

######################################################################
#####   Shuffle and split data ##############################
index = np.random.permutation(images.shape[0])
images = images[index]
ages = ages.T[index]
genders = genders.T[index]
races = races.T[index]
test_split = 0.15
test_size = int(test_split * images.shape[0])
print(ages.shape)

le = LabelEncoder()
le.fit(ages)
le.transform(ages)
ages = to_categorical(ages)
genders = to_categorical(genders , 2)
races = to_categorical(races , 5)


x_tr = images[test_size:]
y_tr_a = ages[test_size:]
y_tr_g = genders[test_size:]
y_tr_r = races[test_size:]

x_cv = images[:test_size]
y_cv_a = ages[:test_size]
y_cv_g = genders[:test_size]
y_cv_r = races[:test_size]
print(x_tr.shape , y_tr_a.shape , y_tr_g.shape , y_tr_r.shape , x_cv.shape , y_cv_r.shape , y_cv_a.shape , y_cv_g.shape)





###################################################################
## train#################
callbacks =[
			EarlyStopping(patience = 3 , monitor='val_acc')
			]
model = my_model.My_Model(weights_path = None)
model.fit(x_tr , [y_tr_g , y_tr_a , y_tr_r] , epochs = 20 , batch_size = 64 , validation_data = (x_cv , [y_cv_g , y_cv_a, y_cv_r]) )

model.save_weights('recognition_age_3.h5', save_format='h5')

##########################################################
## plot graph and save plot


#############################################################




