####This code is to prepare the input data for CNN model
import os
import scipy.ndimage.interpolation
import numpy as np
import scipy.io as sio
import nibabel as nib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
window_size=4
path='/scratch/hli17/BrainNecrosis/ImageData/Progression-'
outpath='/scratch/hli17/BrainNecrosis/DiffFeature/'
DataFile=glob(path+'Tc/*.mat')
DataSet=sio.loadmat(DataFile[0])['DataSetsInfo']
###Augmentation
def Aug(Img):	
	RoatateImg=[scipy.ndimage.interpolation.rotate(Img,angle,reshape=False,mode='nearest', prefilter=False) for angle in [0,-2,2,-4,4]]
	RoatateImg=np.array(RoatateImg,dtype='int64')
	AugImg=RoatateImg
	for shiftX in [0,-2,2,-4,4]:
		for shiftY in [-2,2,-4,4]:
			ShiftImg=np.roll(np.roll(RoatateImg, shiftX, axis=1), shiftY, axis=2)
			AugImg=np.concatenate((AugImg,ShiftImg),axis=0)
	return 	np.concatenate((AugImg,AugImg[:,::-1]),axis=0)

def cut(Img,slice):
	start=int((Img.shape[0]-220)/2)
	return Img[start:start+220,start:start+220,slice:slice+1]

def Combine(Flair,T1c,T1,T2,Seg):
	Location=np.where(Seg>0)
	counts = np.bincount(Location[2])
	Zslice=np.argmax(counts)
	ImgData=np.concatenate((cut(Flair,Zslice),cut(T1c,Zslice),cut(T1,Zslice),cut(T2,Zslice),cut(Seg,Zslice)),axis=2)
	return ImgData
	
###Prepare data
for Data in DataSet:
	MaskName=Data['MRN'][0][0]+Data['ROIName'][0][0]		
	Seg=nib.load(path+'Tc/'+MaskName+'.nii.gz').get_data()*254
	Name=Data['MRN'][0][0]
	Flair = nib.load(path+'Flair/'+Name+'Diff.nii.gz').get_data()
	T1c = nib.load(path+'Tc/'+Name+'Diff.nii.gz').get_data()
	T1 = nib.load(path+'T1/'+Name+'Diff.nii.gz').get_data()
	T2 = nib.load(path+'T2/'+Name+'Diff.nii.gz').get_data()
	start=int((Flair.shape[0]-220)/2)
	Location=np.where(Seg>0)
	counts = np.bincount(Location[2])
	Zslice=np.argmax(counts)
	if min(min(Flair.shape[:2]),min(T1c.shape[:2]),min(T1.shape[:2]),min(T2.shape[:2]))<220:
		continue
	ImgData=Combine(Flair,T1c,T1,T2,Seg)
	AugData=Aug(ImgData)	
	for nn in range(AugData.shape[0]):
		nib.save(nib.Nifti1Image(AugData[nn],np.eye(4)),'./AugData/'+MaskName[:-6]+'_'+str(nn)+'.nii.gz')