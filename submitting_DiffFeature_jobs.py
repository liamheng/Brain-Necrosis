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
import time
import scipy.io as sio
from numpy import genfromtxt
window_size=4
path='/scratch/hli17/BrainNecrosis/ImageData/'
outpath='/scratch/hli17/BrainNecrosis/DiffFeature/'
Folders=os.listdir(path)


for Folder in Folders:
	input_dir=path+Folder
	Files=os.listdir(input_dir)
	DataFile=[ii for ii in Files if '.mat' in ii]
	DataSet=sio.loadmat(os.path.join(input_dir,DataFile[0]))['DataSetsInfo']
	for Data in DataSet:
		Name=Data['MRN'][0][0]
		inputimg=Name+'Diff.nii.gz'
		Name=Data['MRN'][0][0]+Data['ROIName'][0][0]		
		inputmask=Name+'.nii.gz'
		output_dir=os.path.join(outpath,Folder,Name)		
		script = []
		script.append('#PBS -l nodes=1:ppn=1 -l walltime=16:00:00')
		script.append("#PBS -N "+ Name)
		script.append("#PBS -o "+ 'shell/'+ Name +'.out')
		script.append("#PBS -j oe")
		script.append("setenv PATH /workspace/hli17/tmp/Anaconda4.2/bin:$PATH")
		script.append("cd /scratch/hli17/BrainNecrosis")
		script.append('python Local_feature.py '+input_dir+' '+ inputimg +' ' + inputmask+ ' ' + output_dir + ' '+ str(window_size)+'\n')
		np.savetxt('shell/'+Name + '.pbs' ,script,fmt='%s',delimiter = '')
		os.system("qsub " + 'shell/'+Name + '.pbs') # this one submits the job.
