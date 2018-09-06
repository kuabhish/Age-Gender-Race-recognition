import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D , Dense , Flatten, MaxPooling2D , Dropout , BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical 
import random
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
##################################################################################################

def My_Model(weights_path = None):

	input = Input(shape = (64,64,3))
	X = BatchNormalization()(input)
	X = Conv2D(filters=32 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=32 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	
	X = BatchNormalization()(X)
	X = Conv2D(filters=64 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=64 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	
	X = BatchNormalization()(X)
	X = Conv2D(filters=128 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=128, kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=128, kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	
	X = BatchNormalization()(X)
	X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	# X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	# X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	# X = MaxPooling2D(pool_size=(2,2))(X)
	# X = Flatten()(X)
	
	X = BatchNormalization()(X)
	X = Conv2D(filters=512 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = Conv2D(filters=512 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	# X = Conv2D(filters=512 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	# X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Flatten()(X)
	
	pred_gender = Dense(2 , activation='softmax')(X)
	pred_age = Dense(117 , activation = 'softmax')(X)
	pred_race = Dense(5 , activation = 'softmax')(X)
	
	model = Model(inputs = input , outputs = [pred_gender , pred_age , pred_race])
	
	

	

	if weights_path is not None:
		model.load_weights(weights_path)


	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
				  loss=["categorical_crossentropy", "categorical_crossentropy" , "categorical_crossentropy"] ,
				  metrics=['accuracy'])
	
	######################################################### Graph of network#########################################################
	# import os
	# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	# from keras.utils.vis_utils import plot_model
	# plot_model(model, to_file='model.png')
	##########################################################################################################################
	
	# from IPython.display import SVG
	# from keras.utils.vis_utils import model_to_dot

	# SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
	print('Model Created') 
	print(model.summary())	
	return model
	
if __name__=='__main__':
	My_Model(weights_path=None)