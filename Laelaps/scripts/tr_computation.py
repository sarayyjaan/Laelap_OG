import torch
import numpy
import h5py
import dataLoader as dl

#postprocessing constant on labels: we set to all 1 in the postprocessing window
tc, tr = 1.0, {}

for patient in dl.patients:
	# load data
	seizure_begin, seizure_end, pred_mean, rel_mean = dl.loadPatientIntermediateData(patient)

	### extract vectors of ictal/interictal samples of the reliability score where the data was classified as ictal
	pred_mean_interictal, pred_mean_ictal = dl.segmentAndConcatData(pred_mean, seizure_begin, seizure_end, dl.train_seiz[patient])
	rel_mean_interictal, rel_mean_ictal = dl.segmentAndConcatData(rel_mean, seizure_begin, seizure_end, dl.train_seiz[patient])

	# extract the samples in regions labeled as ictal based on tc-threshold
	rel_interictal = rel_mean_interictal[pred_mean_interictal == tc]
	rel_ictal = rel_mean_ictal[pred_mean_ictal == tc]

	# define rounding function for later...
	def round(x, roundingPrecision=1e-2):
		return x - numpy.mod(x, roundingPrecision)

	if len(rel_ictal) == 0:
		#in case that no ictal samples were correctly classified, we set tr to 0 to not further worsen the situation
		tr[patient] = 0
		raise 'no ictal samples were classified correctly -- this should not occur in final HD model'
	else:
		if len(rel_interictal) == 0:
			# if the tc-based filtering results in no false alert, we set tr to the minimum reliability(Delta)-score of any point during the seizures
			tr[patient] = round(min(rel_ictal))
		else:
			#otherwise, we set tr2, that is the min reliability of seizures and trmax, which is the max reliability of seizure minus the bias constant.
			#otherwise, we set tr to the maximum reliability of the interictal points (i.e. to have 0 FDR in validation period) and we increase it in an iterative
			#manner, till we reach the trmax (always less than this max threshold)
			#this is an additional bias_constant that is not explained in the pseudocode of the article
			trmax = max(rel_ictal) - dl.bias_constant

			i = 1
			tr[patient] = round(max(rel_interictal))
			while True:
				i += 1
				v = round(max(rel_interictal)*i)
				if v < trmax:
					tr[patient] = v
				else:
					break

			#we add 0.01 to achieve 0 FDR in validation interictal despite rounding
			if tr[patient] < max(rel_interictal):
				tr[patient] += 0.01
	#we use tc, tr in floating measures: however, tc could be computed as cumulative sum (10 sample out of 10 in the last 10 predictions)
	#and tr could be computed as threshold on the distance between vectors (simply multypling tr * 10000, Delta of the paper)

trExport = ', '.join(["'%s': %f" % (p, tr[p]) for p in dl.patients])
trExport = 'tr = {%s}' % trExport
print(trExport)
