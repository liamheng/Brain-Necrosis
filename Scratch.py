import os
from glob import glob
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import keras
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#########
Table = pd.ExcelFile('./NewAnnotation.xlsx').parse('Sheet1')
file_names=np.array(Table.iloc[:,0])
Label=np.array(Table.iloc[:,1])
#########
def loadimage(subject,Y):
	augfiles=glob('./AugData/'+subject+'*')
	if Y==1:
		augfiles=np.random.choice(augfiles,70, replace=False)
	data=[]	
	for name in augfiles:
		data.append(nib.load(name).get_data())
	AugY=np.ones((len(augfiles),1))*Y	
	return np.array(data),AugY

def loaddata(trainindex):	
	Data=np.array([]).reshape(0,220,220,5)	
	DataY=np.array([]).reshape(0,1)
	for subject,Y in zip(file_names[trainindex],Label[trainindex]):
		augfiles=glob('./AugData/'+subject+'*')
		if not augfiles:
			continue
		Augdata,AugY=loadimage(subject,Y)	
		Data=np.vstack([Data,Augdata])
		DataY=np.vstack([DataY,AugY])
	return Data,DataY

def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	image_input=Input(shape=(220,220,5),name='image_input')
	branch1_conv1=Conv2D(64, kernel_size=(3, 3), border_mode='same', input_shape=(220,220,5), activation='relu')(image_input)
	branch1_conv2=Conv2D(64, kernel_size=(1, 1), border_mode='same', activation='relu')(branch1_conv1)	
	branch1_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch1_conv1)
	branch2_conv1=Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu')(branch1_pool1)
	branch2_conv2=Conv2D(128, kernel_size=(1, 1), border_mode='same', activation='relu')(branch2_conv1)	
	branch2_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch2_conv2)
	branch3_conv1=Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu')(branch2_pool1)
	branch3_conv2=Conv2D(128, kernel_size=(1, 1), border_mode='same', activation='relu')(branch3_conv1)	
	branch3_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch3_conv2)
	branch4_conv1=Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu')(branch3_pool1)
	branch4_conv2=Conv2D(128, kernel_size=(1, 1), border_mode='same', activation='relu')(branch4_conv1)	
	branch4_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch4_conv2)
	branch5_conv1=Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu')(branch4_pool1)
	branch5_conv2=Conv2D(128, kernel_size=(1, 1), border_mode='same', activation='relu')(branch5_conv1)	
	branch5_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch5_conv2)
	branch6_conv1=Conv2D(128, kernel_size=(3, 3), border_mode='same', activation='relu')(branch5_pool1)
	branch6_conv2=Conv2D(128, kernel_size=(1, 1), border_mode='same', activation='relu')(branch6_conv1)	
	branch6_pool1=MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(branch6_conv2)
	branch1_flat=Flatten()(branch6_pool1)
	drop=Dropout(.3)(branch1_flat)
	# FC layers group
	dense1=Dense(512, activation='relu', name='fc1')(drop)
	drop1=Dropout(.3)(dense1)
	dense2=Dense(256, activation='relu', name='fc2')(drop1)
	drop3=Dropout(.3)(dense2)
	out=Dense(2, activation='softmax', name='fc4')(drop3)
	model=Model(inputs=image_input,outputs=out)
	return model	
##########
for round in range(0,10):
	np.random.seed(round)
	index=[]
	for group in np.unique(Label):
		indices=np.random.choice(np.unique(np.where(Label==group)[0]), int(len(np.where(Label==group)[0])*0.2), replace=False)
		index.append(indices)
	index=np.concatenate((index[0],index[1]))
	trainindex=np.setdiff1d(range(len(file_names)),index)	
	TrainX,TrainY=loaddata(trainindex)
	TrainY=keras.utils.to_categorical(TrainY)	
	TestX,TestY=loaddata(index)	
	TestY=keras.utils.to_categorical(TestY)	
	model=get_model()
	sgd=optimizers.SGD(lr=1e-3, momentum=0.0, decay=0.0, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
	earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
	saveBestModel=keras.callbacks.ModelCheckpoint('Round'+str(round)+'_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')	
	history=model.fit(TrainX, TrainY,epochs=200,batch_size=50,verbose=1,callbacks=[earlyStopping,saveBestModel],validation_data=(TestX, TestY))
	model.save('Round'+str(round)+'.hdf5')
	loss_history = np.array(history.history["loss"])
	val_loss_history = np.array(history.history["val_loss"])		
	acc_history = np.array(history.history["acc"])
	val_acc_history = np.array(history.history["val_acc"])
	np.savetxt('Round'+str(round)+"_history.txt", [loss_history,val_loss_history,acc_history,val_acc_history], delimiter=",")


	
