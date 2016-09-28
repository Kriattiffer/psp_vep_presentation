import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from record import butter_filt

sampling_rate = 256

#get data
def get_data_ds2(file = r'./training_set/s2.mat'):
	mat = scipy.io.loadmat(file)#, variable_names  = ['test', 'train'])
	mat = mat[mat.keys()[0]]
	mat = mat[0,0]
	test =  mat['test'].T
	train = mat['train'].T

	# EEG = train
	EEG = test
	return EEG

def get_data_ds1():
	mat = scipy.io.loadmat(r'./training_set/boostingp300/data.mat')
	x = mat['x'].T
	y = mat['y']
	M = np.array(y, dtype='bool')
	# oz = x[-1,]
	# print np.shape(x.T)
	aiml = x[M[0],:,:]
	# aavg = np.average( aiml, axis = 1)
	naiml = x[(np.logical_not(M))[0],:,:]
	# naavg = np.average( naiml, axis = 1)
	return aiml, naiml

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
	channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz', 'HRz']
	for a in range(8):
		delta = avg_ep[0][:,a] - avg_ep[1][:,a]
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), slices_aim[:,:,a].T) #individual eps
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6)
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
		axs.flatten()[a].set_title(channels[a])
	# plt.suptitle = 
	print 'averaged EP: N=%i' % np.shape(slices_aim)[0]
	plt.show()

def train_classifier(aims, non_aims):
	 # reshape epocs into feature vectors
	shp = np.shape(aims)
	aims = aims.reshape(shp[-1]*shp[1], shp[0])
	shp = np.shape(non_aims)
	non_aims = non_aims.reshape(shp[-1]*shp[1], shp[0])
	print np.shape(aims), np.shape(non_aims)

	# y is list aim and non aim feature vectors
	y = np.concatenate((np.ones(len(aims)), np.zeros(len(non_aims))),  axis=0)
	# p is p(x_i|)
	p = np.ones(len(aims)+ len(non_aims))*0.5

	# compute gradient
	nabla = y-p
	print nabla

if __name__ == '__main__':
	
	# eeg = get_data_ds2()
	# sa, sna =  preprocess(eeg)
	sa, sna =  get_data_ds1()
	# plot_ep(sa, sna)
	train_classifier(sa, sna)
