"""
	Evaluates PPG2ABP using several metrics
"""

import pickle
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
sns.set()


def evaluate_BHS_Standard(filename):
	"""
		Evaluates PPG2ABP based on
		BHS Standard Metric
	"""

	def newline(p1, p2):
		"""
		Draws a line between two points
		
		Arguments:
			p1 {list} -- coordinate of the first point
			p2 {list} -- coordinate of the second point
		
		Returns:
			mlines.Line2D -- the drawn line
		"""
		ax = plt.gca()
		xmin, xmax = ax.get_xbound()

		if(p2[0] == p1[0]):
			xmin = xmax = p1[0]
			ymin, ymax = ax.get_ybound()
		else:
			ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
			ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

		l = mlines.Line2D([xmin, xmax], [ymin, ymax], linewidth=1, linestyle='--')
		ax.add_line(l)
		return l

	def BHS_metric(err):
		"""
		Computes the BHS Standard metric
		
		Arguments:
			err {array} -- array of absolute error
		
		Returns:
			tuple -- tuple of percentage of samples with <=5 mmHg, <=10 mmHg and <=15 mmHg error
		"""

		leq5 = 0
		leq10 = 0
		leq15 = 0

		for i in range(len(err)):

			if(abs(err[i]) <= 5):
				leq5 += 1
				leq10 += 1
				leq15 += 1

			elif(abs(err[i]) <= 10):
				leq10 += 1
				leq15 += 1

			elif(abs(err[i]) <= 15):
				leq15 += 1

		return (leq5*100.0/len(err), leq10*100.0/len(err), leq15*100.0/len(err))

	def calcError(Ytrue, Ypred, max_abp, min_abp):#, max_ppg, min_ppg):
		"""
		Calculates the absolute error of sbp,dbp,map etc.
		
		Arguments:
			Ytrue {array} -- ground truth
			Ypred {array} -- predicted
			max_abp {float} -- max value of abp signal
			min_abp {float} -- min value of abp signal
			max_ppg {float} -- max value of ppg signal
			min_ppg {float} -- min value of ppg signal
		
		Returns:
			tuple -- tuple of abs. errors of sbp, dbp and map calculation
		"""

		sbps = []
		dbps = []
		maps = []
	
		for i in (range(len(Ytrue))):
			y_t = Ytrue[i].ravel()
			y_p = Ypred[i].ravel()

			y_t = y_t * (max_abp - min_abp)
			y_p = y_p * (max_abp - min_abp) 

			dbps.append(abs(min(y_t)-min(y_p)))
			sbps.append(abs(max(y_t)-max(y_p)))
			maps.append(abs(np.mean(y_t)-np.mean(y_p)))

		return (sbps, dbps, maps)

	dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))				# loading test data
	X_test = dt['X_test']
	Y_test = dt['Y_test']

	dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))				# loading meta data
	# max_ppg = dt['max_ppg']
	# min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	Y_pred = pickle.load(open(filename, 'rb'))							# loading prediction

	(sbps, dbps, maps) = calcError(Y_test, Y_pred, max_abp, min_abp)#, max_ppg, min_ppg)   # compute errors

	sbp_percent = BHS_metric(sbps)											# compute BHS metric for sbp
	dbp_percent = BHS_metric(dbps)											# compute BHS metric for dbp
	map_percent = BHS_metric(maps)											# compute BHS metric for map

	print('----------------------------')
	print('|        BHS-Metric        |')
	print('----------------------------')

	print('----------------------------------------')
	print('|     | <= 5mmHg | <=10mmHg | <=15mmHg |')
	print('----------------------------------------')
	print('| DBP |  {} %  |  {} %  |  {} %  |'.format(round(dbp_percent[0], 1), round(dbp_percent[1], 1), round(dbp_percent[2], 1)))
	print('| MAP |  {} %  |  {} %  |  {} %  |'.format(round(map_percent[0], 1), round(map_percent[1], 1), round(map_percent[2], 1)))
	print('| SBP |  {} %  |  {} %  |  {} %  |'.format(round(sbp_percent[0], 1), round(sbp_percent[1], 1), round(sbp_percent[2], 1)))
	print('----------------------------------------')

	'''
		Plot figures
	'''

	## SBPS ##

	fig = plt.figure(figsize=(18, 4), dpi=120)
	ax1 = plt.subplot(1,3,1)
	ax2 = ax1.twinx()
	sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '3.67 \%', '7.34 \%',
						 '11.01 \%', '14.67 \%', '18.34 \%', '22.01 \%'])
	ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Absolute Error in SBP Prediction', fontsize=18)
	plt.xlim(xmax=60.0, xmin=0.0)
	plt.xticks(np.arange(0, 60+1, 5))
	p1 = [5, 0]
	p2 = [5, 10000]
	newline(p1, p2)
	p1 = [10, 0]
	p2 = [10, 10000]
	newline(p1, p2)
	p1 = [15, 0]
	p2 = [15, 10000]
	newline(p1, p2)
	plt.tight_layout()

	## DBPS ##

	
	ax1 = plt.subplot(1,3,2)
	ax2 = ax1.twinx()
	sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%',
						 '22.01 \%', '29.35 \%', '36.68 \%', '44.02 \%'])
	ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Absolute Error in DBP Prediction', fontsize=18)
	plt.xlim(xmax=60.0, xmin=0.0)
	plt.xticks(np.arange(0, 60+1, 5))
	p1 = [5, 0]
	p2 = [5, 10000]
	newline(p1, p2)
	p1 = [10, 0]
	p2 = [10, 10000]
	newline(p1, p2)
	p1 = [15, 0]
	p2 = [15, 10000]
	newline(p1, p2)
	plt.tight_layout()

	## MAPS ##

	
	ax1 = plt.subplot(1,3,3)
	ax2 = ax1.twinx()
	sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%', '22.01 \%',
						 '29.35 \%', '36.68 \%', '44.02 \%', '51.36 \%'])
	ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Absolute Error in MAP Prediction', fontsize=18)
	plt.xlim(xmax=60.0, xmin=0.0)
	plt.xticks(np.arange(0, 60+1, 5))
	p1 = [5, 0]
	p2 = [5, 10000]
	newline(p1, p2)
	p1 = [10, 0]
	p2 = [10, 10000]
	newline(p1, p2)
	p1 = [15, 0]
	p2 = [15, 10000]
	newline(p1, p2)
	plt.tight_layout()

	plt.show()


def evaluate_AAMI_Standard(filename):
	"""
		Evaluate PPG2ABP using AAMI Standard metric	
	"""

	def calcErrorAAMI(Ypred, Ytrue, max_abp, min_abp):#, max_ppg, min_ppg):
		"""
		Calculates error of sbp,dbp,map for AAMI standard computation
		
		Arguments:
			Ytrue {array} -- ground truth
			Ypred {array} -- predicted
			max_abp {float} -- max value of abp signal
			min_abp {float} -- min value of abp signal
			max_ppg {float} -- max value of ppg signal
			min_ppg {float} -- min value of ppg signal
		
		Returns:
			tuple -- tuple of errors of sbp, dbp and map calculation
		"""

		sbps = []
		dbps = []
		maps = []

		for i in (range(len(Ytrue))):
			y_t = Ytrue[i].ravel()
			y_p = Ypred[i].ravel()

			y_t = y_t * (max_abp - min_abp) 
			y_p = y_p * (max_abp - min_abp) 

			dbps.append(min(y_p)-min(y_t))
			sbps.append(max(y_p)-max(y_t))
			maps.append(np.mean(y_p)-np.mean(y_t))

		return (sbps, dbps, maps)

	dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))			# loading test data
	X_test = dt['X_test']
	Y_test = dt['Y_test']

	dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))			# loading metadata
	# max_ppg = dt['max_ppg']
	# min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	Y_pred = pickle.load(open(filename, 'rb'))						# loading prediction

	(sbps, dbps, maps) = calcErrorAAMI(Y_test, Y_pred, max_abp, min_abp)#, max_ppg, min_ppg)		# compute error

	print('---------------------')
	print('|   AAMI Standard   |')
	print('---------------------')

	print('-----------------------')
	print('|     |  ME   |  STD  |')
	print('-----------------------')
	print('| DBP | {} | {} |'.format(round(np.mean(dbps), 3), round(np.std(dbps), 3)))
	print('| MAP | {} | {} |'.format(round(np.mean(maps), 3), round(np.std(maps), 3)))
	print('| SBP | {} | {} |'.format(round(np.mean(sbps), 3), round(np.std(sbps), 3)))
	print('-----------------------')

	'''
		Plotting figures
	'''

	## DBPS ##

	fig = plt.figure(figsize=(18, 4), dpi=120)
	ax1 = plt.subplot(1, 3, 1)
	ax2 = ax1.twinx()
	sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%',
						 '22.01 \%', '29.35 \%', '36.68 \%', '44.02 \%'])
	ax1.set_xlabel('Error (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Error in DBP Prediction', fontsize=18)
	plt.xlim(xmax=50.0, xmin=-50.0)
	#plt.xticks(np.arange(0, 60+1, 5))
	plt.tight_layout()

	## MAPS ##

	#fig = plt.figure(figsize=(6,4), dpi=120)
	ax1 = plt.subplot(1, 3, 2)
	ax2 = ax1.twinx()
	sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%', '22.01 \%',
						 '29.35 \%', '36.68 \%', '44.02 \%', '51.36 \%'])
	ax1.set_xlabel('Error (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Error in MAP Prediction', fontsize=18)
	plt.xlim(xmax=50.0, xmin=-50.0)
	#plt.xticks(np.arange(0, 60+1, 5))
	plt.tight_layout()

	## SBPS ##

	ax1 = plt.subplot(1, 3, 3)
	ax2 = ax1.twinx()
	sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax1)
	sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax2)
	ax2.set_yticklabels(['0 \%', '3.67 \%', '7.34 \%',
						 '11.01 \%', '14.67 \%', '18.34 \%', '22.01 \%'])
	ax1.set_xlabel('Error (mmHg)', fontsize=14)
	ax1.set_ylabel('Number of Samples', fontsize=14)
	ax2.set_ylabel('Percentage of Samples', fontsize=14)
	plt.title('Error in SBP Prediction', fontsize=18)
	plt.xlim(xmax=50.0, xmin=-50.0)
	#plt.xticks(np.arange(0, 60+1, 5))
	plt.tight_layout()

	plt.show()


def evaluate_BP_Classification(filename):
	"""
		Evaluates PPG2ABP for BP Classification
	"""

	dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))		# loading test data
	Ytrue = dt['Y_test']


	dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))		# loading meta data
	# max_ppg = dt['max_ppg']
	# min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	Ypred = pickle.load(open(filename, 'rb'))					# loading prediction
	

	### DBPS ####

	cls_gt = []
	cls_pred = []

	for i in (range(len(Ytrue))):
		y_t = Ytrue[i].ravel()
		y_p = Ypred[i].ravel()

		dbp_gt = max_abp*min(y_t)+min_abp
		dbp_pred = max_abp*min(y_p)+min_abp

		if(dbp_gt <= 80):
			cls_gt.append('Normotension')
		elif((dbp_gt > 80)and(dbp_gt <= 90)):
			cls_gt.append('Pre-hypertension')
		elif(dbp_gt > 90):
			cls_gt.append('Hypertension')
		else:
			print('bump')				# this will never happen, check for error

		if(dbp_pred <= 80):
			cls_pred.append('Normotension')
		elif((dbp_pred > 80)and(dbp_pred <= 90)):
			cls_pred.append('Pre-hypertension')
		elif(dbp_pred > 90):
			cls_pred.append('Hypertension')
		else:
			print('bump')				# this will never happen, check for error

	print('DBPS Classification Accuracy')
	print(classification_report(cls_gt, cls_pred, digits=5))
	# print('-'*25)
	# print(accuracy_score(cls_gt,cls_pred))

	cm = confusion_matrix(cls_gt, cls_pred)
	classes = ['Hypertension', 'Normotension', 'Prehypertension']
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	fig = plt.figure(figsize=(16, 6), dpi=120)
	ax = plt.subplot(1,2,1)
	im = ax.imshow(cm, interpolation='nearest', cmap='GnBu')			# draw confusion matrix

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.25)

	ax.figure.colorbar(im, cax=cax)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=classes, yticklabels=classes)

	ax.set_title('Hypertension Classification Accuracy Using DBP', fontsize=18)
	ax.set_ylabel('True label', fontsize=16)
	ax.set_xlabel('Predicted label', fontsize=16)

	plt.setp(ax.get_xticklabels(), rotation=45, fontsize=15,  ha="right",
			 rotation_mode="anchor")

	plt.setp(ax.get_yticklabels(), fontsize=15)

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center", fontsize=15,
					color="white" if cm[i, j] > thresh else "black")

	ax.grid(False)

	fig.tight_layout()


	### SBPS ####

	cls_gt = []
	cls_pred = []

	for i in (range(len(Ytrue))):
		y_t = Ytrue[i].ravel()
		y_p = Ypred[i].ravel()

		sbp_gt = max_abp*max(y_t)+min_abp
		sbp_pred = max_abp*max(y_p)+min_abp

		if(sbp_gt <= 120):
			cls_gt.append('Normotension')
		elif((sbp_gt > 120)and(sbp_gt <= 140)):
			cls_gt.append('Prehypertension')
		elif(sbp_gt > 140):
			cls_gt.append('Hypertension')
		else:
			print('bump')				# this will never happen, check for error

		if(sbp_pred <= 120):
			cls_pred.append('Normotension')
		elif((sbp_pred > 120)and(sbp_pred <= 140)):
			cls_pred.append('Prehypertension')
		elif(sbp_pred > 140):
			cls_pred.append('Hypertension')
		else:
			print('bump')				# this will never happen, check for error

	print('SBPS Classification Accuracy')
	print(classification_report(cls_gt, cls_pred, digits=5))
	# print('-'*25)
	# print(accuracy_score(cls_gt,cls_pred))

	cm = confusion_matrix(cls_gt, cls_pred)
	classes = ['Hypertension', 'Normotension', 'Prehypertension']
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	ax = plt.subplot(1,2,2)
	im = ax.imshow(cm, interpolation='nearest', cmap='GnBu')		# draw confusion matrix

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.25)

	ax.figure.colorbar(im, cax=cax)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=classes, yticklabels=classes)

	ax.set_title('Hypertension Classification Accuracy Using SBP', fontsize=18)
	ax.set_ylabel('True label', fontsize=16)
	ax.set_xlabel('Predicted label', fontsize=16)

	plt.setp(ax.get_xticklabels(), rotation=45, fontsize=15, ha="right",
			 rotation_mode="anchor")

	plt.setp(ax.get_yticklabels(), fontsize=15)

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center", fontsize=15,
					color="white" if cm[i, j] > thresh else "black")

	ax.grid(False)

	fig.tight_layout()

	plt.show()


def bland_altman_plot(filename):
	"""
		Draws the Bland Altman plot
	"""

	def bland_altman(data1, data2):
		"""
		Computes mean +- 1.96 sd
		
		Arguments:
			data1 {array} -- series 1
			data2 {array} -- series 2
		"""
	
		data1     = np.asarray(data1)
		data2     = np.asarray(data2)
		mean      = np.mean([data1, data2], axis=0)
		diff      = data1 - data2                   # Difference between data1 and data2
		md        = np.mean(diff)                   # Mean of the difference
		sd        = np.std(diff, axis=0)            # Standard deviation of the difference

		plt.scatter(mean, diff,alpha=0.1, s=4)
		plt.axhline(md,  color='gray', linestyle='--')
		plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
		plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
		plt.ylim(ymin=-75,ymax=75)
		plt.xlabel('Avg. of Target and Estimated Value (mmHg)',fontsize=14)
		plt.ylabel('Error in Prediction (mmHg)',fontsize=14)
		print(md+1.96*sd,md-1.96*sd)



	dt = pickle.load(open(os.path.join('data','test.p'),'rb'))			# loading test data
		
	Ytrue = dt['Y_test']

	dt = pickle.load(open(os.path.join('data','meta.p'),'rb'))			# loading meta data
	# max_ppg = dt['max_ppg']
	# min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	Ypred = pickle.load(open(filename,'rb'))						# loading prediction

	Ytrue = Ytrue * max_abp + min_abp
	Ypred = Ypred * max_abp + min_abp

	sbpTrues = []
	sbpPreds = []

	dbpTrues = []
	dbpPreds = []

	mapTrues = []
	mapPreds = []

	for i in (range(len(Ytrue))):
		y_t = Ytrue[i].ravel()
		y_p = Ypred[i].ravel()

		sbpTrues.append(max(y_t))
		sbpPreds.append(max(y_p))

		dbpTrues.append(min(y_t))
		dbpPreds.append(min(y_p))

		mapTrues.append(np.mean(y_t))
		mapPreds.append(np.mean(y_p))

	'''
		Plots the Bland Altman plot
	'''

	fig = plt.figure(figsize=(18,5), dpi=120)
	plt.subplot(1,3,1)
	print('---------DBP---------')
	bland_altman(dbpTrues,dbpPreds)
	plt.title('Bland-Altman Plot for DBP Prediction',fontsize=18)
	
	plt.subplot(1,3,2)
	print('---------MAP---------')
	bland_altman(mapTrues,mapPreds)
	plt.title('Bland-Altman Plot for MAP Prediction',fontsize=18)
	
	plt.subplot(1,3,3)
	print('---------SBP---------')
	bland_altman(sbpTrues,sbpPreds)	
	plt.title('Bland-Altman Plot for SBP Prediction',fontsize=18)
	
	plt.show()
	

def regression_plot(filename):
	"""
		Draws the Regression Plots
	"""

	dt = pickle.load(open(os.path.join('data','test.p'),'rb'))		# loading the test data
		
	Ytrue = dt['Y_test']
	
	dt = pickle.load(open(os.path.join('data','meta.p'),'rb'))		# loading the meta data
	# max_ppg = dt['max_ppg']
	# min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	Ypred = pickle.load(open(filename,'rb'))					# loading the prediction

	Ytrue = Ytrue * max_abp + min_abp
	Ypred = Ypred * max_abp + min_abp

	sbpTrues = []
	sbpPreds = []

	dbpTrues = []
	dbpPreds = []

	mapTrues = []
	mapPreds = []

	for i in (range(len(Ytrue))):
		y_t = Ytrue[i].ravel()
		y_p = Ypred[i].ravel()

		sbpTrues.append(max(y_t))
		sbpPreds.append(max(y_p))

		dbpTrues.append(min(y_t))
		dbpPreds.append(min(y_p))

		mapTrues.append(np.mean(y_t))
		mapPreds.append(np.mean(y_p))

	'''
		Drawing the regression plots
	'''

	plt.figure(figsize=(18,6),dpi=120)
	
	plt.subplot(1,3,1)
	sns.regplot(dbpTrues,dbpPreds,scatter_kws={'alpha':0.2,'s':1},line_kws={'color':'#e0b0b4'})
	plt.xlabel('Target Value (mmHg)',fontsize=14)
	plt.ylabel('Estimated Value (mmHg)',fontsize=14)
	plt.title('Regression Plot for DBP Prediction',fontsize=18)
	
	plt.subplot(1,3,2)
	sns.regplot(mapTrues,mapPreds,scatter_kws={'alpha':0.2,'s':1},line_kws={'color':'#e0b0b4'})
	plt.xlabel('Target Value (mmHg)',fontsize=14)
	plt.ylabel('Estimated Value (mmHg)',fontsize=14)
	plt.title('Regression Plot for MAP Prediction',fontsize=18)

	plt.subplot(1,3,3)
	sns.regplot(sbpTrues,sbpPreds,scatter_kws={'alpha':0.2,'s':1},line_kws={'color':'#e0b0b4'})	
	plt.xlabel('Target Value (mmHg)',fontsize=14)
	plt.ylabel('Estimated Value (mmHg)',fontsize=14)
	plt.title('Regression Plot for SBP Prediction',fontsize=18)
	plt.show()
	
	'''
		Printing statistical analysis values like r and p value
	'''
	print('DBP')
	print(scipy.stats.linregress(dbpTrues,dbpPreds))
	print('MAP')
	print(scipy.stats.linregress(mapTrues,mapPreds))
	print('SBP')
	print(scipy.stats.linregress(sbpTrues,sbpPreds))
	
def evaluate_metrics(filename):   
    def calcError(Ytrue, Ypred, max_abp, min_abp):#, max_ppg, min_ppg):
        sbp_t = []
        sbp_p = []
        dbp_t = []
        dbp_p = []
        map_t = []
        map_p = []
        
        x = 0
        y = 0
        
        
        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()
            
            y_t = y_t * (max_abp - min_abp) 
            y_p = y_p * (max_abp - min_abp) 
            
            sbp_p.append(abs(max(y_p)))
            dbp_p.append(abs(min(y_p)))
            map_p.append(abs(np.mean(y_p)))
            sbp_t.append(abs(max(y_t)))
            dbp_t.append(abs(min(y_t)))
            map_t.append(abs(np.mean(y_t)))
            
        
        print("SBP")
        print("Mean Absolute Error : ", round(mean_absolute_error(sbp_t, sbp_p), 3))
        print("Root Mean Squared Error : ", round(mean_squared_error(sbp_t, sbp_p, squared=False),3))
        print("R2 : ", r2_score(sbp_t, sbp_p))
              
        print("")
        
        print("DBP")
        print("Mean Absolute Error : ", round(mean_absolute_error(dbp_t, dbp_p),3))
        print("Root Mean Squared Error : ", round(mean_squared_error(dbp_t, dbp_p, squared=False),3))
        print("R2 : ", r2_score(dbp_t, dbp_p))
        
        print("")
        
        print("MAP")
        print("Mean Absolute Error : ", mean_absolute_error(map_t, map_p))
        print("Root Mean Squared Error : ", round(mean_squared_error(map_t, map_p, squared=False), 2))
        print("R2 : ", r2_score(map_t, map_p))
        
        print("------------------------------------------------------------------------")
        
    dt = pickle.load(open(os.path.join('data', 'test.p'), 'rb'))				# loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']
    
    dt = pickle.load(open(os.path.join('data', 'meta.p'), 'rb'))				# loading meta data
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']
    

    Y_pred = pickle.load(open(filename, 'rb'))							# loading prediction
    calcError(Y_test, Y_pred, max_abp, min_abp)#, max_ppg, min_ppg)