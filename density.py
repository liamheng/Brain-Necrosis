"""
Density function estimation for 3D local feature images.
Usage: python .\density_estimation.py .\feature\ fea_1_ROI.nii .\Mask.nii .\dens\

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-2-19
"""

import os
import sys
import numpy as np
import scipy.io as sio
import nibabel as nib
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import timeit
import math
"""
make sure that you have installed all the libraries above
"""


def dens(input_dir, window_size):
	"""
	Local feature map calculation on ROI files (PNG format).

	Args:
		input_dir (str): full path to the ROI files
		input_img (str): the name of ROI files
		maskdir (str): mask path
		output_dir (str): full path to the output feature images saved in PNG format
	"""
	# fixlin= np.loadtxt('/scratch/hli17/BrainNecrosis/MaskBFeature/'+moduality+'BSize'+str(window_size)+'.txt',delimiter='	')
	for ii in range(99):
		input_img='fea_' + str(ii+1)+'.nii.gz'
		# Get the pixel array of the feature map
		filename = os.path.join(input_dir, 'Size'+str(window_size), input_img)
		img = nib.load(filename).get_data()
		Mask=nib.load(os.path.join(input_dir,'Size'+str(window_size),'fea_1.nii.gz')).get_data()			
		data = img[Mask>0]
		data0 = np.sort(np.array(np.reshape(data, (len(data),1))),axis=0)
		if data0.shape[0]>999:
			interval= math.floor(data0.shape[0]/500)
			data = data0[::interval]
		else:
			data = data0
		# density function estimation
		# def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
		#     """Kernel Density Estimation with Scikit-learn"""
		#     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
		#     kde_skl.fit(x[:, np.newaxis])
		#     # score_samples() returns the log-likelihood of the samples
		#     log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
		#     return np.exp(log_pdf)

		# bandwidth selection (10-fold cross-validation)
		data_min = np.percentile(data, 0.1)
		data_max = np.percentile(data, 99.9)
		m = 100      # number of grids
		x0 = np.linspace(data_min, data_max, m)		
		m=x0.shape[0]
		grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 10, 20)}, cv=10)
		grid.fit(data)
		kde = grid.best_estimator_
		pdf = np.exp(kde.score_samples(x0.reshape(m,1)))

		out_matrix = np.zeros((m, 2))
		out_matrix[:, 0] = x0
		out_matrix[:, 1] = pdf

		output_name = os.path.join(input_dir, 'Size'+str(window_size), input_img[:-7] + '_dens.mat')
		sio.savemat(output_name, {'density':out_matrix})

if __name__ == '__main__':
	input_dir0 = sys.argv[1]
	window_size0 = sys.argv[2]
	start = timeit.default_timer()
	dens(input_dir0, window_size0)
	stop = timeit.default_timer()
	print(stop - start)






