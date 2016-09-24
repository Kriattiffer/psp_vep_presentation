import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from record import butter_filt

sampling_rate = 256

#get data
mat = scipy.io.loadmat(r'./training_set/s1.mat')#, variable_names  = ['test', 'train'])
mat = mat[mat.keys()[0]]
mat = mat[0,0]
test =  mat['test'].T
train = mat['train'].T

# EEG = train
EEG = test

###filter data
EEG[:,:-2] = butter_filt(EEG[:,:-2], [1,9], fs = sampling_rate)

### get epochs
letter_ind_aim = np.arange(np.shape(EEG)[0])[np.logical_and(EEG[:,-2] != 0, EEG[:,-1] == 1 )] 
letter_ind_non_aim = np.arange(np.shape(EEG)[0])[np.logical_and(EEG[:,-2] != 0, EEG[:,-1] == 0 )] 

slices_aim = []
for a in letter_ind_aim:
	slice = EEG[a:a+sampling_rate,:]
	slices_aim.append(slice)
slices_aim = np.array(slices_aim)

slices_non_aim = []
for a in letter_ind_non_aim:
	slice = EEG[a:a+sampling_rate,:]
	slices_non_aim.append(slice)
slices_non_aim = np.array(slices_non_aim)

### postprocess epochs
slices_aim = slices_aim[:,::4,:] #downsample to 64 hz 
slices_non_aim = slices_non_aim[:,::4,:] #downsample to 64 hz 
pass # windsorize
slices_aim = slices_aim - slices_aim[:,0:1,:] # subtract 1st sample
slices_non_aim = slices_non_aim - slices_non_aim[:,0:1,:] # subtract 1st sample
print np.shape(slices_aim)
print np.shape(slices_non_aim)



# plot average ERP

avgaim = np.average(slices_aim, axis = 0)
avgnonaim = np.average(slices_non_aim, axis = 0)

avg_ep = [avgaim[:,1:-2], avgnonaim[:,1:-2]]
fig,axs = plt.subplots(nrows =3, ncols = 3)
channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
for a in range(8):
	delta = avg_ep[0][:,a] - avg_ep[1][:,a]

	delta = avg_ep[0][:,a]
	axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta)
	delta = avg_ep[1][:,a]
	axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta)
	delta = avg_ep[0][:,a] - avg_ep[1][:,a]
	axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6)
	axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
	axs.flatten()[a].set_title(channels[a])
plt.show()
