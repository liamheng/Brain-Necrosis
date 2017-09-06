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

#script=[line for line in open('ClassifypCR.m')]
Rounds=np.arange(1,41)
window_sizes=[8]
path='/scratch/hli17/BrainNecrosis/'
modualities=['Flair','T1','T1c','T2']
for moduality in range(4):
	for window_size in window_sizes:
		for Round in Rounds:
			print(Round,window_size)
			# this part first check how many jobs has been there, you should change some component highlighted
			stay = True
			while stay:
				stay = False
				tmp = os.popen('qstat | grep hli17').read() # change kyu2 to your userid
				Lines = len(tmp.split('\n'))
				if Lines >= 20:# here I set 100 jobs as a limit, you can do more or less.
					stay = True
					time.sleep(800)# if more than 100 jobs are there, then the script went sleep for 800 seconds, then come back to check if some jobs are finished 

			outfile1 = '/scratch/hli17/BrainNecrosis/Diff_'+modualities[moduality]+'_S' + str(window_size) + '_R' + str(Round) + '.mat' # this is to check whether or not a set of jobs have been finished, this file is the output file from my actual program. 
			#if you have multiple files, you can modify this accordinglyDiffResult/
			File='Diff_'+modualities[moduality]+'_S' + str(window_size) + '_R' + str(Round)
			if os.path.exists(outfile1):
				continue
			else:#actual submitting script		
				script = []
				script.append('#PBS -l nodes=1:ppn=1 -l walltime=24:00:00')
				script.append("#PBS -N "+ File)
				script.append("#PBS -o "+ 'shell/'+ File +'.out')
				script.append("#PBS -j oe")
				script.append("module load matlab")
				script.append("cd /scratch/hli17/BrainNecrosis")
				script.append('matlab -nosplash -nodesktop -nojvm -r "ClassifyDiffFunction('+ str(Round) + ',' + str(moduality+1) + ',' + str(window_size) +')" \n')
				np.savetxt('shell/'+File + '.pbs' ,script,fmt='%s',delimiter = '')
				os.system("qsub " + 'shell/'+File + '.pbs') # this one submits the job.
