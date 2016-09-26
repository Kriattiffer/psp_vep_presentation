import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from record import butter_filt

sampling_rate = 256

#get data
def get_data(file = r'./training_set/s2.mat'):
	mat = scipy.io.loadmat(file)#, variable_names  = ['test', 'train'])
	mat = mat[mat.keys()[0]]
	mat = mat[0,0]
	test =  mat['test'].T
	train = mat['train'].T

	# EEG = train
	EEG = test
	return EEG
def preprocess(EEG):
	###filter data
	EEG[:,:-2] = butter_filt(EEG[:,:-2], [1,9], fs = sampling_rate)
	
	### get epochs
	bb1 = EEG[:,-1] == 1 
	bb1 =  [bb1[a] if (bb1[a] and not bb1[a-1]) else False for a in range(len(bb1))] 
	letter_ind_aim = np.arange(np.shape(EEG)[0])[np.array(bb1)]

	bb2 = np.logical_and(EEG[:,-2] != 0, EEG[:,-1] == 0 )
	bb2 =  [bb2[a] if (bb2[a] and not bb2[a-1]) else False for a in range(len(bb2))] 
	letter_ind_non_aim = np.arange(np.shape(EEG)[0])[np.array(bb2)]

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
	# slices_aim = slices_aim[:,::4,:] #downsample to 64 hz 
	# slices_non_aim = slices_non_aim[:,::4,:] #downsample to 64 hz 
	pass # windsorize
	slices_aim = slices_aim - slices_aim[:,0:1,:] # subtract 1st sample
	slices_non_aim = slices_non_aim - slices_non_aim[:,0:1,:] # subtract 1st sample
	print np.shape(slices_aim)
	print np.shape(slices_non_aim)
	return slices_aim, slices_non_aim


def plot_ep(slices_aim, slices_non_aim):
	avgaim = np.average(slices_aim, axis = 0)
	avgnonaim = np.average(slices_non_aim, axis = 0)

	avg_ep = [avgaim[:,1:-2], avgnonaim[:,1:-2]]
	fig,axs = plt.subplots(nrows =3, ncols = 3)
	channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
	for a in range(8):
		delta = avg_ep[0][:,a] - avg_ep[1][:,a]
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), slices_aim[:,:,a].T)
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6)
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
		axs.flatten()[a].set_title(channels[a])
	plt.show()

eeg = get_data()
sa, sna =  preprocess(eeg)
plot_ep(sa, sna)