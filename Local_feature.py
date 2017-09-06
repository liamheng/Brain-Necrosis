"""
ROI Preprocessing: Local feature map calculation for 3D images.
Usage: python .\pre_feature_3D.py .\data\ sample3D.nii .\feature\ 5

Author: Rongjie Liu (rongjie.liu.ee@gmail.com)
Last update: 2017-2-13
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy import stats
from skimage.feature import local_binary_pattern
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
from mahotas import features
from math import ceil
import timeit

"""
make sure that you have installed all the libraries above
"""


def feature(input_dir, input_img, mask, output_dir, window_size):
	"""
	Local feature map calculation on ROI files (PNG format).

	Args:
		input_dir (str): full path to the ROI files
		input_img (str): the name of ROI files
		output_dir (str): full path to the output feature images saved in PNG format
		window_size (num): the window size of local patch
	"""
	# Get the pixel array of the ROI
	ws = int(float(window_size))
	filename = os.path.join(input_dir, input_img)
	im = nib.load(filename)
	affine = im.affine
	img = im.get_data() 
	# img= (img - img.min()) / (np.sort(img,axis=None)[-10] - img.min()) * 255
	# img[img<0]=0
	# img[img>255]=255
	# fea_img = nib.Nifti1Image(img, affine)
	# nib.save(fea_img, output_dir + 'Img_255.nii.gz')	
	Mask=nib.load(os.path.join(input_dir, mask)).get_data()
	# Slices=np.unique(np.where(Mask==1)[2])
	# img=img[:,:,tuple(Slices)]
	# Mask=Mask[:,:,tuple(Slices)]
	""" Local feature calculation """
	if len(img.shape)>2:
		length, width, height = img.shape
	else:
		img=img.reshape(img.shape[0],img.shape[1],1)
		Mask=Mask.reshape(img.shape[0],img.shape[1],1)
		length, width, height = img.shape
	# number of the features
	n_fea = 99
	fea = np.zeros((length, width, height, n_fea))
	for h in range(height):
		for l in range(length):
			for w in range(width):
				if Mask[l,w,h]==0:
					continue
				patch0 = img[max((l - ws),0):min(length,(l + ws)), max((w - ws),0):min(width,(w + ws)), h]
				l0, w0 = patch0.shape
				patch = np.array(np.reshape(patch0, (1, l0 * w0)))

				"""
				first order statistics based feature
				list {max, min, median, 25percentile, 75percentile, std, skew, kurtosis, entropy}
				"""
				fea[l, w, h, 0] = np.max(patch[0])
				fea[l, w, h, 1] = np.min(patch[0])
				fea[l, w, h, 2] = np.median(patch[0])
				fea[l, w, h, 3] = np.percentile(patch[0], 25)
				fea[l, w, h, 4] = np.percentile(patch[0], 75)
				fea[l, w, h, 5] = np.percentile(patch[0], 75)-np.percentile(patch[0], 25)
				fea[l, w, h, 6] = np.std(patch[0])
				fea[l, w, h, 7] = stats.skew(patch[0])
				fea[l, w, h, 8] = stats.kurtosis(patch[0])
				hist = stats.histogram(patch[0], numbins=5)
				fea[l, w, h, 9] = stats.entropy(hist.count / np.sum(hist.count))

				"""
				GLCM based feature
				list {angular second moment, contrast, correlation, variance, inverse difference moment,
				sum average, sum variance, sum entropy, entropy, difference variance, difference entropy,
				info. measure. of corr. 1, info. measure. of corr. 2, max. correlation coefficient}
				"""
				patch2 = np.array((patch0 - img.min()) / (img.max() - img.min()) * 256, dtype=np.uint8)
				g_matrix = features.haralick(patch2)
				fea[l, w, h, 10:23] = np.mean(g_matrix, axis=0)

				"""
				Local Binary Patterns based shape descriptors {7 first order statistics on histogram of LBP}
				"""
				lbp = local_binary_pattern(patch0, 8, ceil(ws/2), 'default')
				fea[l, w, h, 23] = np.max(lbp)
				fea[l, w, h, 24] = np.min(lbp)
				fea[l, w, h, 25] = np.median(lbp)
				fea[l, w, h, 26] = np.percentile(lbp, 25)
				fea[l, w, h, 27] = np.percentile(lbp, 75)
				fea[l, w, h, 28] = np.percentile(lbp, 75)-np.percentile(lbp, 25)
				fea[l, w, h, 29] = np.std(lbp)
				
				"""
				Hu Moment based shape descriptor {7 moments}
				"""
				m = moments(np.array(patch0, dtype=np.float))
				cr = m[0, 1] / m[0, 0]
				cc = m[1, 0] / m[0, 0]
				cm = moments_central(np.array(patch0, dtype=np.float), cr, cc)
				ncm = moments_normalized(cm)
				hum = moments_hu(ncm)
				fea[l, w, h, 30:37] = hum

				"""
				Zernike moment based shape descriptors {first 8 moments}
				"""
				zm = features.zernike_moments(patch0, ws)
				fea[l, w, h, 37:45] = zm[1:9]		
				
				"""
				Threshold Adjacency Statistics based shape descriptors {9 statistics} * 6 = 54
				"""
				tas = features.tas(patch0)
				# lentas = len(tas)
				fea[l, w, h, 45:100] = tas[:54]
	# Save all the local feature maps in NIFTI format	
	featuredir=os.path.join(output_dir,'Size'+str(window_size))
	if not os.path.exists(featuredir):
		os.makedirs(featuredir)	
	for ii in range(n_fea):
		output_filename = featuredir+'/fea_'+str(ii+1)+'.nii.gz'
		data = np.reshape(fea[:, :, :, ii], (length, width, height))
		fea_img = nib.Nifti1Image(data, affine)
		nib.save(fea_img, output_filename)	


if __name__ == '__main__':
	input_dir0 = sys.argv[1]
	input_img0 = sys.argv[2]
	mask0 = sys.argv[3]
	output_dir0 = sys.argv[4]
	window_size0 = sys.argv[5]
	print(sys.argv[1])
	start = timeit.default_timer()
	feature(input_dir0, input_img0, mask0, output_dir0, window_size0)
	stop = timeit.default_timer()
	print(stop - start)
