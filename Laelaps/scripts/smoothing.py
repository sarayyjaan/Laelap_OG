import torch
import numpy
import dataLoader as dl

number_seizures = range(0,110)
prediction = torch.zeros(1,1)
prediction = prediction[0,:]
dSeiz = torch.zeros(1,1)
dSeiz = dSeiz[0,:]
dInter = torch.zeros(1,1)
dInter = dInter[0,:]

for patient in dl.patients:
	with open('../intermediate_results/Pat' + patient + '/' + 'Total_period_randomiM'+'.txt', 'rb') as f1:
		prediction, dSeiz,dInter = torch.load(f1)
	Pred_mean = torch.FloatTensor(prediction.size()[0]).zero_()
	reliability = torch.abs(torch.add(dSeiz,-dInter))
	rel_mean = torch.FloatTensor(reliability.size()[0]).zero_()
	#we simply load the files computed with iEEG_HD_analysis_prediction.py and smooth them using a window of 10 predictions (5.5 seconds in total)
	#with cumulative sum for Pred_mean, we can use again integer measurement (tc = 10)
	for i in numpy.arange(10,prediction.size()[0]):
	    Pred_mean[i] = torch.mean(prediction[i-10:i])
	    rel_mean[i] = torch.mean(reliability[i-10:i])
	    print i
	with open('/usr/scratch/sassauna3/msc18f3/Predictions/Pat' + patient + '/' + 'Pred_rel_mean'+'.txt', 'wb') as f1:
	        torch.save((Pred_mean, rel_mean),f1)
