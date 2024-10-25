import torch
import matplotlib.pyplot as plt
import numpy
import h5py
import dataLoader as dl

train_seiz = dl.train_seiz
tr = dl.tr
#default value of tc for all patients
tc = dl.tc
for patient in dl.patients:
	begin, end, Pred_mean, rel_mean = dl.loadPatientIntermediateData(patient)
	begin = begin*2
	end = end*2
	#begin and end of seizures
	Pred_mean = torch.add(Pred_mean, -tc)
	rel_mean = torch.add(rel_mean, -tr[patient])
	Final_prediction = numpy.array(torch.sign(torch.add(torch.sign(Pred_mean),torch.sign(rel_mean))))
	ict_pred=numpy.where(Final_prediction==1)[0]
	#computing the FDR: errors near to seizures are not considered since they could be some form of late epileptic form or early epileptic form
	for aa in range(len(begin)):
		for ii in range(len(ict_pred)):
			if ict_pred[ii] < end[aa][0]+15*60*2 and ict_pred[ii] > begin[aa][0]-15*60*2:
				ict_pred[ii]=0
	old_value = 0
	#two false detections that are nearer than 15 minutes are recognized to be a single error since they could belong to the same fake seizure
	for ii in range(len(ict_pred)):
		if ict_pred[ii]!=0 and old_value<ict_pred[ii]-15*60*2:
			old_value = ict_pred[ii]
			ict_pred[ii]=1
		else:
			if ict_pred[ii]!=0:
				old_value = ict_pred[ii]
			ict_pred[ii]=0
	print(patient +': errors: ' + str(sum(ict_pred)))
	#for three patients we don't include all seizures, but only leading one or seizures farer than 10 minutes one from the other
	ict_pred=numpy.where(Final_prediction==1)[0]
	if patient == '08':
		begin = begin[[0,67,68,69],:]
		end = end[[0,67,68,69],:]
	elif patient == '09':
		begin = begin[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,22,23,26],:]
		end = end[[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,22,23,26],:]
	elif patient =='14':
		begin = begin[[0,1],:]
		end = end[[0,1],:]
	predicted_seizure = 0
	delay = 0
	#for each seizure in begin vector we see if we predict it and we calculate the delay if we correctly predict.
	#3 cases:
	#	1 trained seizure
	# 	2 trained seizures
	#	2 trained seizures + ID04: it uses seizure 4 with respect to seizure 2
	if train_seiz[patient]==1:
		for aa in range(1,len(begin)):
			if (Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1).any():
				delay_part=(numpy.where(Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1)[0][0]-10*60*2)/float(2)
				delay = delay + delay_part
				predicted_seizure = predicted_seizure+1
		print(patient +': sensitivity: ' + str(float(predicted_seizure)/(len(begin)-1)*100) +'\nDelay: '+ str(delay/(len(begin)-1))+'\nIdentified Seizures: ' + str(predicted_seizure) + '\nTested seizures: ' + str(len(begin)-1))
	elif train_seiz[patient]==2 and patient != '04':
		for aa in range(2,len(begin)):
			if (Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1).any():
				delay_part=(numpy.where(Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1)[0][0]-10*60*2)/float(2)
				delay = delay + delay_part
				predicted_seizure = predicted_seizure+1
		print(patient +': sensitivity: ' + str(float(predicted_seizure)/(len(begin)-2)*100)+'\nDelay: '+ str(delay/(len(begin)-2))+'\nIdentified Seizures: ' + str(predicted_seizure) + '\nTested seizures: ' + str(len(begin)-2))
	else:
		for aa in range(1,len(begin)):
			if aa !=3:
				if (Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1).any():
					delay_part=(numpy.where(Final_prediction[begin[aa][0]-10*60*2:end[aa][0]+10*60*2]==1)[0][0]-10*60*2)/float(2)
					delay = delay + delay_part
					predicted_seizure = predicted_seizure+1
		print(patient +': sensitivity: ' + str(float(predicted_seizure)/(len(begin)-2)*100)+'\nDelay: '+ str(delay/(len(begin)-2))+'\nIdentified Seizures: ' + str(predicted_seizure) + '\nTested seizures: ' + str(len(begin)-2))
