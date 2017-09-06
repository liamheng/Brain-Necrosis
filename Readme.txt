1. DataImageNormal.py and DataROI.py transfer IBEX data to Nifti and generate the mask file.
2. Run submitting_Registration_jobs.py to regist the two points image. And use DifferenceImage.py to generate the difference image of two time points.
3. Run submitting_DiffFeature_jobs.py to capture local features, and submitting_density_jobs.py to generate density of features.
4. Run submitting_matlabDiff_jobs.py to classify the density.

Please use your path to replace the path in the code.