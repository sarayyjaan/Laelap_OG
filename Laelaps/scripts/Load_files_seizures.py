import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
#Example: How to load file corresponding to begin of seizure 1 of patient 1
patient = '16'
#Load info of a patient
string = '/baltic/users/msc18f3/longTermiEEGanonym32/ID' + patient +'_'+'info'+ '.mat'
info = sio.loadmat(string)
#Load fs and begin of the seizures
fs = info['fs'][0][0]
seizure_begin = info['seizure_begin']
#file index computing and offset from the beginning of that recording
file_index = (seizure_begin[0]/3600 + 1)[0].astype(int)
offset = ((seizure_begin[0]%3600)*fs)[0].astype(int)
#plot to understand if a seizure is correct loaded 
string = '/baltic/users/msc18f3/longTermiEEGanonym32/ID' + patient +'_'+str(file_index)+'h'+ '.mat'
EEG = sio.loadmat(string)
EEG = EEG['EEG']
plotEEG = EEG[:, offset-2*fs*60:offset+fs*60]
plt.plot(plotEEG[0,:])
plt.show()
