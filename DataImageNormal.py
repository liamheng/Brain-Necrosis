'''this is code for transfer IBEX data to nifti'''
import os
import sys
import scipy.io as sio
import numpy as np
import nibabel as nib
import scipy.ndimage

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array((scan[0],scan[1],scan[2]), dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', prefilter=False)
    return image, new_spacing

path='/scratch/hli17/BrainNecrosis/IBEX_DataSets/'
Folders=os.listdir(path)


for Folder in Folders:
	input_dir=os.path.join(path,Folder,'1FeatureDataSet_ImageROI')
	Files=os.listdir(input_dir)
	DataFile=[ii for ii in Files if '.mat' in ii]
	DataSet=sio.loadmat(os.path.join(input_dir,DataFile[0]))['DataSetsInfo']
	for Data in DataSet:
		Name=Data['MRN'][0][0]
		Iminfo=nib.load(os.path.join(input_dir, Name+'.nii.gz'))
		Image=Iminfo.get_data()	
		scan=np.array((Data['XPixDim'][0][0][0],Data['XPixDim'][0][0][0],Data['ZPixDim'][0][0][0]), dtype=np.float32)
		ImageNor,_= resample(Image, scan, [0.1,0.1,0.8])
		nib.save(nib.Nifti1Image(ImageNor[:,:,::-1], Iminfo.affine),os.path.join(input_dir, Name+'Nor.nii.gz'))

		
	
		
	
for Folder in Folders:
	input_dir=os.path.join(path,Folder,'1FeatureDataSet_ImageROI')
	Files=os.listdir(input_dir)
	DataFile=[ii for ii in Files if '.mat' in ii]
	DataSet=sio.loadmat(os.path.join(input_dir,DataFile[0]))['DataSetsInfo']
	for Data in DataSet:
		Name=Data['MRN'][0][0]+Data['ROIName'][0][0]
		Image=nib.load(os.path.join(path,Folder,'Mask', Name+'.nii.gz')).get_data()	
		Image.dtype='int'
		Image[Image>0]=1
		scan=np.array((Data['XPixDim'][0][0][0],Data['XPixDim'][0][0][0],Data['ZPixDim'][0][0][0]), dtype=np.float32)
		ImageNor,_= resample(Image, scan, [0.1,0.1,0.8])
		nib.save(nib.Nifti1Image(ImageNor[:,:,::-1], np.diag([0.1,0.1,0.8,1])),os.path.join('/scratch/hli17/BrainNecrosis/ImageData/',Folder[:-2], Name+'.nii.gz'))	