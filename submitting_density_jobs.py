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
from numpy import genfromtxt
moduality='-Flair'
window_size=8
path='/scratch/hli17/BrainNecrosis/DiffFeature/'
Folders=os.listdir(path)
Folders=[ii for ii in Folders if moduality in ii]
Folders=[ii for ii in Folders if '.txt' not in ii]
# Folders=[ii for ii in Folders if 'T1c' not in ii]
for Folder in Folders:
	input_dir=path+Folder
	Files=os.listdir(input_dir)
	# Files=[ii for ii in Files if 'B' in ii]
	Files=[ii for ii in Files if '.txt' not in ii]
	for File in Files:
		dir=os.path.join(input_dir,File)
		# modual=moduality
		script = []
		script.append('#PBS -l nodes=1:ppn=1 -l walltime=16:00:00')
		script.append("#PBS -N "+ File)
		script.append("#PBS -o "+ 'shell/'+ File +'.out')
		script.append("#PBS -j oe")
		script.append("setenv PATH /workspace/hli17/tmp/Anaconda4.2/bin:$PATH")
		script.append("cd /scratch/hli17/BrainNecrosis")
		script.append('python density.py '+dir+' ' + str(window_size)+'\n')
		np.savetxt('shell/'+File + '.pbs' ,script,fmt='%s',delimiter = '')
		os.system("qsub " + 'shell/'+File + '.pbs') # this one submits the job.
