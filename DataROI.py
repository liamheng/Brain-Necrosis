'''This is code for generate Mask file'''
import os
import sys
import scipy.io as sio
import numpy as np
import nibabel as nib
import pandas as pd

input_dir='/scratch/hli17/BrainNecrosis/IBEX_DataSets/'
Files=os.listdir(input_dir)
for File in Files:
	matfile=os.listdir(input_dir+File+'/1FeatureDataSet_ImageROI/')
	mat_contents = sio.loadmat(input_dir+File+'/1FeatureDataSet_ImageROI/'+matfile[0])
	oct_struct = mat_contents['DataSetsInfo']
	for ii in range(oct_struct.shape[0]):
		val = oct_struct[ii,0]
		Name=val['MRN'][0]+val['ROIName'][0]
		Mask=val['ROIBWInfo']['MaskData'][0,0]
		Img=val['ROIImageInfo']['MaskData'][0,0]
		output_dir='/scratch/hli17/BrainNecrosis/Nifti_DataSets/'+File+'/'
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		nib.save(nib.Nifti1Image(Img, np.eye(4)), output_dir+Name+'.nii.gz')
		nib.save(nib.Nifti1Image(Mask, np.eye(4)), output_dir+Name+'Mask.nii.gz')

