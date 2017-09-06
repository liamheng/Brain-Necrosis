'''
This is code for regist the images
'''
import sys
import os
import numpy as np
import time
from numpy import genfromtxt
moduality='Flair-'
path='/scratch/hli17/BrainNecrosis/IBEX_DataSets/'
Folders=os.listdir(path)
Folders=[ii for ii in Folders if moduality in ii]

for Moving,Fix in zip(Folders[::2],Folders[1::2]):
	input_dir=os.path.join(path,Moving,'1FeatureDataSet_ImageROI')
	Files=os.listdir(input_dir)
	Files=[ii for ii in Files if 'Nor.'  in ii]
	for File in Files:
		Movingfile=os.path.join(input_dir,File)
		Fixfile=os.path.join(path,Fix,'1FeatureDataSet_ImageROI',File)
		script = []
		script.append('#PBS -l nodes=1:ppn=1 -l walltime=16:00:00')
		script.append("#PBS -N "+ File)
		script.append("#PBS -o "+ 'shell/'+ File +'.out')
		script.append("#PBS -j oe")
		script.append("module load ANTs")
		script.append("cd /scratch/hli17/BrainNecrosis")
		script.append('antsRegistrationSyNQuick.sh -d 3 -f '+Fixfile+' -m '+ Movingfile +' -o ' + os.path.join('/scratch/hli17/BrainNecrosis/ImageData/',Fix[0:-2],File[0:-7])+'\n')
		np.savetxt('shell/'+File + '.pbs' ,script,fmt='%s',delimiter = '')
		os.system("qsub " + 'shell/'+File + '.pbs') # this one submits the job.
