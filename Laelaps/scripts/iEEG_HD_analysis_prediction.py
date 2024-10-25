#!/usr/bin/env python

import sys
import numpy as np
import os
import h5py
from HD_model import model   #here there are all the functions that we use
import torch
import time
import pdb
import json
import dataLoader as dl

__author__ = "Alessio Burrello"
__email__ = "s238495@studenti.polito.it"
if __name__ == '__main__':
    #GPU to be used
    device = 0
    torch.cuda.set_device(device)
    #ID of the patient to analyze
    patient = '01'
    # dimension of hypervector
    D = 10000
    # LBP length + 1
    T = 7
    #Please note that the dataset could be uploaded in a different version, hence the procedure to load could change.
    # total number of possible symbols, i.e. 64 for LBP length = 6
    totalNumberBP = 2**(T-1)
    #offset from begin of first seizure to define the training segment: we train using segments that starts from seizure_1_beg and end in seizure_1_end
    seizure_1_beg = dl.seizure_1_beg
    seizure_1_end = dl.seizure_1_end
    #offset for second training seizure: patient 12 and 17 use the second seizure only for validation, but no segment from them is used for training
    beg = dl.beg
    end = dl.end
    #second training seizure: is always the second seizure. Only patient 04 makes exception, because first 3 seizures were almost identical
    sez_2_choice = dl.seiz_2_choice
    #two patients have not fs defined inside.
    fs_choice = {'05':512, '11':1024}
    #first seizure used for training, always the first one
    ict = 0
    seizure_1 = [seizure_1_beg[patient], seizure_1_end[patient]]
    #in case of two segments of seizure trained, here we add the second one
    if patient in beg.keys():
        ict2 = sez_2_choice[patient]
        seizure_2 = [beg[patient], end[patient]]
    loading_file = '/usr/scratch/sassauna3/msc18f3/longTermiEEGanonym/' +  patient + '.mat'
    f = h5py.File(loading_file, 'r')
    #defining fs
    if patient == '05' or patient == '11':
        fs = fs_choice[patient]
    else:
        fs = np.array(f['fs']).astype(int)
        fs = fs[0][0]
    seizure_begin = (np.array(f['timeCollStart'])*fs).astype(int)
    second = fs
    minutes = second*60
    #taking segments of recording that are needed for training and save in variables
    EEGSez1 = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]+seizure_1[0]*fs:seizure_begin[ict][0]+seizure_1[1]*fs])).cuda()
    if patient in beg.keys():
        EEGSez2 = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict2][0]+seizure_2[0]*fs:seizure_begin[ict2][0]+seizure_2[1]*fs])).cuda()
    EEGInterictal = torch.from_numpy(np.array(f['EEGRefMedian'][:,seizure_begin[ict][0]- 10*minutes:seizure_begin[ict][0] - 10*minutes +30*fs])).cuda()
    nSignals = EEGSez1.size(0)
    torch.manual_seed(1);
    #importing the model, with the creation of the iM, with dimendion DxtotalNumberBP and DxnSignals
    model0 = model(totalNumberBP,D,nSignals,device,T)
    #####TRAINING STEP#######
    #here we create the prototypes of the Associative memory. If two segments are used for seizures, we sum them together.
    print('Learning Seizure 1')
    temp1=model0.learn_HD_proj_big(EEGSez1,fs)
    if patient in beg.keys():
        print('Learning Seizure 2')
        temp2=model0.learn_HD_proj_big(EEGSez2,fs)
    print('Learning Interictal period')
    temp3=model0.learn_HD_proj_big(EEGInterictal,fs)
    if patient in beg.keys():
        #breaking ties in a deterministic manner
        queeryVectorS0 = torch.sign(torch.add(torch.add(temp1,1,temp2),1,torch.mul(temp1,temp2)))
    else:
        queeryVectorS0 = torch.sign(temp1)
    queeryVectornS0 = torch.sign(temp3)
    ####TESTING####
    i = 0
    j = 0
    begin_test = 0
    end_test = f['EEGRefMedian'].shape[1]
    #divide the recording in slice of 1 hour: in this way we avoid to charge the full recording in RAM
    seizure_slices = np.arange(begin_test,end_test,60*minutes)
    distanceVectorsS0 = torch.zeros(1,(end_test-begin_test)/fs*2).cuda(device = device)
    distanceVectorsnS0 = torch.zeros(1,(end_test-begin_test)/fs*2).cuda(device = device)
    prediction0 = torch.zeros(1,(end_test-begin_test)/fs*2).cuda(device = device)
    for seizures in seizure_slices:
        if seizures == seizure_slices[-1]:
            EEGtest=torch.from_numpy(np.array(f['EEGRefMedian'][:,seizures:end_test])).cuda()
        else:
            EEGtest=torch.from_numpy(np.array(f['EEGRefMedian'][:,seizures:seizures+60*minutes])).cuda()
        # we make a single prediction every 0.5 seconds
        index = np.arange(0,EEGtest.size(1),fs/2)
        j = j+1
        for iStep in index:
            #we use a window of 1 second to predict the label
            temp = model0.learn_HD_proj(EEGtest[:,iStep:iStep+fs])
            #we output label and distances from the two hypervectors: this will be used to calculate reliability.
            [distanceVectorsS0[0,i],distanceVectorsnS0[0,i],prediction0[0,i]] = model0.predict(temp, queeryVectorS0, queeryVectornS0, D)
            i = i + 1
            print 'hour: ' + str(j) +'; half second: ' + str(i%7200)
    with open('/usr/scratch/sassauna3/msc18f3/Predictions/Pat' + patient + '/' + 'Total_period_randomiM'+'.txt', 'wb') as f1:
        torch.save((prediction0[0,:], distanceVectorsS0[0,:], distanceVectorsnS0[0,:]),f1)
