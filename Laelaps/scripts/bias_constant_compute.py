import matplotlib.pyplot as plt
import numpy
import h5py
import torch
import dataLoader as dl

#postprocessing constant on labels: we set to all 1 in the postprocessing window
tc = 1.0
#offset from begin of first seizure to define the training segment: we train using segments from seizure_1_beg to seizure_1_end
seizure_1_beg = dl.seizure_1_beg
seizure_1_end = dl.seizure_1_end
#offset for second training seizure: patient 12 and 17 use the second seizure only for validation, but no segment from them is used for training
beg = dl.beg
end = dl.end
#second training seizure: is always the second seizure. Only patient 04 makes exception, because first 3 seizures were almost identical
seiz_2_choice = dl.seiz_2_choice
#number of training seizures
train_seiz = dl.train_seiz
bias_constant = 0
i=0
for patient in dl.patients:
	seizure_begin, seizure_end, Pred_mean, rel_mean = dl.loadPatientIntermediateData(patient)
	#different start and end points: they comprise start-end of trained seizures and start-end of trained segments of that seizures
	beg0 = seizure_begin[0][0]*2
	end0 = seizure_end[0][0]*2
	off0beg = seizure_1_beg[patient]*2
	off0end = seizure_1_end[patient]*2
	#include segment of first trained seizure
	rel_meanseiz1=rel_mean[beg0+10:end0]
	Pred_meanseiz1=Pred_mean[beg0+10:end0]
	#include segment of second trained seizure
	if train_seiz[patient]==2:
		beg1 = seizure_begin[seiz_2_choice[patient]][0]*2
		end1 = seizure_end[seiz_2_choice[patient]][0]*2
		rel_meanseiz2=rel_mean[beg1+10:end1]
		Pred_meanseiz2=Pred_mean[beg1+10:end1]
		rel_meanseiz1 = numpy.concatenate((rel_meanseiz1,rel_meanseiz2),axis=0)
		Pred_meanseiz1 = numpy.concatenate((Pred_meanseiz1,Pred_meanseiz2),axis=0)
	#mean reliability of the entire trained seizures
	rel_seiz_unbias = numpy.mean(numpy.array(rel_meanseiz1[Pred_meanseiz1==tc]))
	#same procedure, but including only the training segments of that seizures, to correct the bias
	rel_meanseiz1=rel_mean[beg0+off0beg+10:beg0+off0end]
	Pred_meanseiz1=Pred_mean[beg0+off0beg+10:beg0+off0end]
	if patient in beg.keys():
		off1beg = beg[patient]*2
		off1end = end[patient]*2
		rel_meanseiz2=rel_mean[beg1+10+off1beg:end1+off1end]
		Pred_meanseiz2=Pred_mean[beg1+10+off1beg:end1+off1end]
		rel_meanseiz1 = numpy.concatenate((rel_meanseiz1,rel_meanseiz2),axis=0)
		Pred_meanseiz1 = numpy.concatenate((Pred_meanseiz1,Pred_meanseiz2),axis=0)
	rel_seiz_bias = numpy.mean(numpy.array(rel_meanseiz1[Pred_meanseiz1==tc]))
	#index of correction for that patient: different of reliability between the training segments and the whole training seizures
	rel_gain = numpy.array(rel_seiz_bias-rel_seiz_unbias)
	if numpy.isnan(rel_gain) or rel_gain <= 0:
		bias_constant = bias_constant
	else:
		bias_constant = bias_constant + rel_gain
		i = i+1
	print(patient +': ' + str(rel_gain))
# mean of the correction index, including only positive ones.
bias_constant = bias_constant / i
print('mean_rel_bias' +': ' + str(bias_constant))
