import torch
import numpy
import h5py

train_seiz = {'01':1,'02':1,'03':1,'04':2,'05':1,'06':1,'07':2,'08':2,'09':2,'10':1,'11':1,'12':2,'13':2,'14':1,'15':1,'16':1,'17':1,'18':1}
patients = train_seiz.keys()
bias_constant = 0.0153 #computed with the script bias_constant_compute.py
tc = 0.95
#offset from begin of first seizure to define the training segment: we train using segments from seizure_1_beg to seizure_1_end
seizure_1_beg = {'01':10, '02':10, '03':10, '04':35, '05':10, '06':3, \
'07':4,'08':0, '09':0, '10':20, '11':0, '12':33, '13':12, '14':15, '15':5, '16':10, '17':13, '18':50}
seizure_1_end = {'01':40, '02':30, '03':30, '04':43, '05':16, '06':30,\
'07':12, '08':6, '09':8, '10':54, '11':30, '12':58, '13':35, '14':30, '15':40, '16':40, '17':40, '18':80}
#offset for second training seizure: patient 12 and 17 use the second seizure only for validation, but no segment from them is used for training
beg = {'04':12, '07':10,'12':28, '08':30}
end = {'04':43, '07':40,'12':45, '08':60}
#second training seizure: is always the second seizure. Only patient 04 makes exception, because first 3 seizures were almost identical
seiz_2_choice = {'04':3, '07':1,'12':1, '08': 67, '09': 1, '13': 1}

def loadPatientIntermediateData(patient):
	f = h5py.File('/usr/scratch/sassauna3/msc18f3/longTermiEEGanonym/' + patient + '.mat', 'r')
	#vectors with start-end of seizures. For some patients, 08,09,14 there are more seizures then the one analized. This because near seizures (< 10 minutes)
	#are merged togheter and only leading seizures are considered for patients 08 and 14.
	seizure_begin = numpy.array(f['timeCollStart']).astype(int)
	seizure_end = numpy.array(f['timeCollEnd']).astype(int)
	#reliability and prediction label already smoothed by a 10 sample average window, shifting of 1 point.
	pred_mean, rel_mean = torch.load('../intermediate_results/Pat' + patient + '/' + 'Pred_rel_mean'+'.txt')
	return seizure_begin, seizure_end, pred_mean, rel_mean

def segmentAndConcatData(data, seizure_begin, seizure_end, numSeizures):
	data_interictal, data_ictal = [], []
	endIdx = 0
	for seizureIdx in range(numSeizures):
		# extract all interictal data (sequences always start with interictal)
		if seizureIdx == 0:
			prevIdx = 0
		else:
			prevIdx = endIdx + 10*60*2
		endIdx = seizure_begin[seizureIdx][0]*2
		data_interictal = numpy.concatenate((data_interictal, data[prevIdx:endIdx]), axis=0)

		# extract all ictal data
		prevIdx = endIdx + 10
		endIdx = seizure_end[seizureIdx][0]*2
		data_ictal = numpy.concatenate((data_ictal, data[prevIdx:endIdx]), axis=0)
	return data_interictal, data_ictal

#threshold computed with the script tr_computation.py
tr = {'01': 0.010000, '02': 0.020000, '03': 0.150000, '04': 0.000000, '05': 0.040000, '06': 0.010000, '07': 0.050000, '08': 0.030000, '09': 0.000000, '10': 0.010000, '11': 0.020000, '12': 0.030000, '13': 0.040000, '14': 0.020000, '15': 0.080000, '16': 0.070000, '17': 0.010000, '18': 0.030000}
