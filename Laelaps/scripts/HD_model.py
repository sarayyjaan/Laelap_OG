#!/usr/bin/env python

'''
Local sensitive Hashing function
'''
import time, sys
import numpy as np
import scipy.special
import torch
import matplotlib.pyplot as plt

__author__ = "Alessio Burrello"
__email__ = "s238495@studenti.polito.it"

class model:
	def __init__(self, N_seats1,HD_dim, N_seats2, device, T,cuda = True):
		'''
		This function inzialize the model;
		INPUTS:
		N_seats1: dimension-2 of first iM
		HD_dim: dimension-1 of iM
		N_seats2: dimension-2 of second iM
		device: GPU number
		OUTPUTS:
		proj_mat_LBP, proj_mat_channels: iM. They are composed by -1 and 1 and not
		by 0 and 1 to speed up the computation. However, there is a corrispondence for
		every operation between binary and bipolar components
		Sum&Threshold -> Sum and 0 crossing
		XOR -> dot product
		Hamming distance -> dot product and then sum (Domain: [0,1]->[-1,1])
		'''
		self.training_done = False
		self.N_seats1 = N_seats1
		self.N_seats2 = N_seats2
		self.HD_dim = HD_dim
		self.T = T
		self.device = device
		if cuda:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1).cuda(device = device)
		else:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1)
		self.proj_mat_LBP[self.proj_mat_LBP >=0] = 1
		self.proj_mat_LBP[self.proj_mat_LBP < 0] = -1
		if cuda:
			self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2).cuda(device = device)
		else:
			self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2)
		self.proj_mat_channels[self.proj_mat_channels >=0] = 1
		self.proj_mat_channels[self.proj_mat_channels < 0] = -1

	def learn_HD_proj(self,EEG):
		'''
		This function encodes a 1 second window in an hypervector.
		INPUTS:
		EEG: fragment of EEG to encode (1 second window)
		OUTPUTS:
		queeryVector: hypervector that encodes the 1 second window -> histogram of LBP
		projected on the hyperspace
		'''
		queeryVector = torch.cuda.FloatTensor(1,1).zero_()
		N_channels,learningEnd = EEG.size()
		#weights of each sample to create the index to adress the iM
		LBP_weights = torch.cuda.FloatTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
		for iStep in range(learningEnd-6):
			x = EEG[:,iStep:(iStep+self.T)].float()
			bp = (torch.add(-x[:,0:self.T-1], 1,x[:,1:self.T])>0).float()
			value = torch.sum(torch.mul(LBP_weights,bp), dim=1)
			bindingVector=torch.mul(self.proj_mat_channels,self.proj_mat_LBP[:,value.long()])
			#the output vector encodes the LBP of a single time step of each channel
			output_vector = torch.sum(bindingVector,dim=1)
			if N_channels%2==0:
				output_vector = torch.add(torch.mul(self.proj_mat_LBP[:,1],self.proj_mat_LBP[:,2]),1,output_vector)
			queeryVector=torch.add(queeryVector,1,torch.sign(output_vector))
		queeryVector = torch.sign(queeryVector)
		return queeryVector

	def learn_HD_proj_big(self,EEG,fs):
		'''
		This function encodes a segment of EEG of multiple seconds (training segment)
		INPUTS:
		EEG: fragment of EEG to encode (n second window)
		OUTPUTS:
		queeryVector: hypervector that encodes the segment window -> mean histogram of LBP
		projected on the hyperspace
		'''
		queeryVector = torch.cuda.FloatTensor(1,1).zero_()
		N_channels,learningEnd = EEG.size()
		index = np.arange(0,learningEnd,fs/2)
		for iStep in index:
			temp = self.learn_HD_proj(EEG[:,iStep:iStep+fs])
			queeryVector=torch.add(queeryVector,1,temp)
		queeryVector = torch.sign(queeryVector)
		return queeryVector

	def predict(self,testVector,Ictalprot, Interictalprot,D):
		'''
		This function produces a label for an hypervector that is given as testVector
		INPUTS:
		testVector: the Vector that we want to classify.
		Ictalprot: prototype of a seizure
		Interictalprot: prototype of interictal
		D: dimension of hypervector
		OUTPUTS:
		distanceVectorsS, distanceVectorsnS: distance between vector and prototypes. It is computed
		dividing the dot product by D. Of course, this is only to ease computation, but as said in the paper
		all could be done in integer number (without dividing by D and usin binary operations)
		'''
		distanceVectorsS = torch.div(torch.mm(testVector,Ictalprot.t()),D)
		distanceVectorsnS = torch.div(torch.mm(testVector,Interictalprot.t()),D)
		if torch.gt(distanceVectorsS[0], distanceVectorsnS[0])[0]:
			prediction = 1
		else:
			prediction = 0
		return distanceVectorsS[0,0],distanceVectorsnS[0,0],prediction
