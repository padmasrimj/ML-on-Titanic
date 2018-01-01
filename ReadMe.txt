Course Project
CS6375 - Machine Learning
===========================================================================================================================
Languages, Version and Tools used: Python 3.5, R 3.4.0 and RStudio
How to execute course project ?

Data preprocessing:
		
		From command prompt and make sure 'train.csv' is in current directory

		python preprocessing.py 

		Preprocessing outputs 'preprocessed_data.csv' and later used by R code below.

Data visualization:
		
		From command prompt
		python data_visualization.py

Classifiers:

	All the installer dependencies will get resolved in your R studio
	> Run the Installer.R from RStudio or through command prompt  Rscript Installer.R
		
	> Run the FinalClassifierCode.R in RStudio or through command prompt  Rscript FinalClassifierCode.R
	[Make sure preprocessed_data.csv file in working directory ]

	> All the accuracies and the evaluation metrics will be displayed in the console.