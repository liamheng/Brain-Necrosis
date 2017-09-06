'''
This script submits jobs to the department server
Author: Kaixian Yu
Email: kyu2@mdanderson.org
Date: 04/24/2017
You are free to modify and redistribute the script.
'''
import sys
import os
import numpy as np
import scipy.io as sio
import nibabel as nib
import shutil
import scipy.ndimage

path='/scratch/hli17/BrainNecrosis/ImageData/'
Folders=os.listdir(path)

for Folder in Folders:
	input_dir=os.path.join(path,Folder)
	Files=os.listdir(input_dir)
	DataFile=[ii for ii in Files if '.mat' in ii]
	DataSet=sio.loadmat(os.path.join(input_dir,DataFile[0]))['DataSetsInfo']
	for Data in DataSet:
		Name=Data['MRN'][0][0]
		ImageB=nib.load(os.path.join(input_dir, Name+'Nor.nii.gz')).get_data()
		ImageB=np.round((ImageB-ImageB.min())/(ImageB.max()-ImageB.min())*255)
		WarpedA=nib.load(os.path.join(input_dir, Name+'NorWarped.nii.gz')).get_data()
		ImageA=WarpedA
		Image=ImageA[ImageA>0]
		ImageA=np.round((ImageA-Image.min())/(Image.max()-Image.min())*255)
		Mask=[ii for ii in Files if Name in ii and 'GTV' in ii]
		NotROI=np.ones(ImageB.shape,dtype='int')
		for ROI in Mask:
			ROIFile=nib.load(os.path.join(input_dir, ROI)).get_data()
			NotROI=NotROI*(1-ROIFile)	
		#Imtensity Mapping
		Image = ImageA*NotROI
		source = np.concatenate((ImageA[Image>0],np.arange(256)))
		template = ImageB[Image>0]
		s_, bin_idx= np.unique(ImageA[ImageA>0], return_inverse=True)
		s_values, s_counts = np.unique(source, return_counts=True)
		t_values, t_counts = np.unique(template, return_counts=True)
		s_quantiles = np.cumsum(s_counts).astype(np.float64)
		s_quantiles /= s_quantiles[-1]
		t_quantiles = np.cumsum(t_counts).astype(np.float64)
		t_quantiles /= t_quantiles[-1]
		# interpolate linearly to find the pixel values in the template image
		# that correspond most closely to the quantiles in the source image
		interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
		MappedImageA=interp_t_values[bin_idx]
		ImageA[ImageA>0]=np.round(MappedImageA)
		nib.save(nib.Nifti1Image(ImageA, np.eye(4)),os.path.join(input_dir, Name+'TimeA.nii.gz'))
		nib.save(nib.Nifti1Image(ImageB, np.eye(4)),os.path.join(input_dir, Name+'TimeB.nii.gz'))
		# Difference=ImageB-ImageA
		# Image=Difference[WarpedA>0]
		# Difference=(Difference-Image.min())/(Image.max()-Image.min())*255
		# Difference[Difference>255]=255
		# nib.save(nib.Nifti1Image(Difference, np.eye(4)),os.path.join(input_dir, Name+'Diff.nii.gz'))
		
		