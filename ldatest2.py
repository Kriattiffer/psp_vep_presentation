# test classifier on publically availible P300 datasets
import numpy as np
import scipy.io
from record import butter_filt

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sampling_rate = 256
downsample_div = 4
averaging_bin = 4


def plot_ep(slices_aim, slices_non_aim, number_of_channels = 8):
	''' get arrays of aim and non-aim epocs - plot group averages for 8 channels '''
	avgaim = np.average(slices_aim, axis = 0)
	avgnonaim = np.average(slices_non_aim, axis = 0)

	avg_ep = [avgaim[:,1:-2], avgnonaim[:,1:-2]]
	fig,axs = plt.subplots(nrows =3, ncols = 3)
	channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz', 'HRz']
	for a in range(number_of_channels):
		delta = avg_ep[0][:,a] - avg_ep[1][:,a]
		# axs.flatten()[a].plot(range(0, len(delta)*4, 4), slices_aim[:,:,a].T) #individual eps
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), avgnonaim[:,a]) # individual eps
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6) # delta
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
		axs.flatten()[a].set_title(channels[a])
	print 'averaged EP: N=%i' % np.shape(slices_aim)[0]
	plt.show()

def get_data_ds2(file = r'./training_set/s3.mat', set = 'train'):
	'''Guger P300 BCI dataset'''
	mat = scipy.io.loadmat(file)#, variable_names  = ['test', 'train'])
	mat = mat[mat.keys()[0]]
	mat = mat[0,0]
	test =  mat[set].T
	train = mat[set].T
	EEG = test
	return EEG

def get_data_ds1(set):
	'''boostingp300 BCI dataset'''
	mat = scipy.io.loadmat(r'./training_set/boostingp300/data.mat')
	x = mat['x'].T
	y = mat['y']
	M = np.array(y, dtype='bool')
	if set == 'test':
		aims = x[M[0],:,:][::2,:,:]
		non_aims = x[(np.logical_not(M))[0],:,:][::2,:,:]
	elif set == 'train':
		aims = x[M[0],:,:][1::2,:,:]
		non_aims = x[(np.logical_not(M))[0],:,:][1::2,:,:]
	return aims, non_aims

def get_data_ds3(set):
	'''MSU P300 BCI dataset'''
	if set == 'test':
		mat = scipy.io.loadmat("./training_set/Data_Processing/mov01/mov_epo_filt_01_01.mat")
	else:
		mat = scipy.io.loadmat("./training_set/Data_Processing/mov01/mov_epo_filt_01_04.mat")
	aims = mat['epo_T']
	non_aims = mat['epo_NT']
	aims = aims.reshape((np.shape(aims)[2], np.shape(aims)[0], np.shape(aims)[1]))
	non_aims = non_aims.reshape((np.shape(non_aims)[2], np.shape(non_aims)[0], np.shape(non_aims)[1]))
	print np.shape(aims)
	return aims, non_aims

def preprocess(EEG):
	''' Filter EEG; cut in into aim and non-aim epocs '''
	###filter data
	EEG[:,:-2] = butter_filt(EEG[:,:-2], [1,9], fs = sampling_rate)
	### get offsets of epocs
	bb1 = EEG[:,-1] == 1 
	bb1 =  [bb1[a] if (bb1[a] and not bb1[a-1]) else False for a in range(len(bb1))] 
	letter_ind_aim = np.arange(np.shape(EEG)[0])[np.array(bb1)]

	bb2 = np.logical_and(EEG[:,-2] != 0, EEG[:,-1] == 0 )
	bb2 =  [bb2[a] if (bb2[a] and not bb2[a-1]) else False for a in range(len(bb2))] 
	letter_ind_non_aim = np.arange(np.shape(EEG)[0])[np.array(bb2)]
	# collect epocs of 1 second length
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

	return slices_aim, slices_non_aim

def downsample(slices):
	slices = slices[:,::downsample_div,:] #downsample  
	return slices

def subtrfirst(slices):
	slices = slices - slices[:,0:1,:] # subtract 1st sample
	return slices

def average_epocs(slices):
	'''Average aim and non_aim epocs according to given averaging_bin. '''
	shp = np.shape(slices)
	slices = slices[shp[0]%averaging_bin:,:,:]
	slices = slices.reshape((shp[0]/averaging_bin, averaging_bin, shp[1], shp[2]))
	slices = np.average(slices, axis = 1)
	return slices

def prepare_epocs(aims, non_aims, session= 'train'):
	''' input epocs - output array of feature vectors ans list of 1 and 0 labels.
	Downsample according to given downsample_div; subtract first element from all
	epocs to mke them start from 0; average epocs according to averaging_bin. 
	Then transform to feature vectors and create y'''
	aims, non_aims = downsample(aims), downsample(non_aims)
	aims, non_aims = subtrfirst(aims), subtrfirst(non_aims)
	aims, non_aims = average_epocs(aims), average_epocs(non_aims)

	shpa= np.shape(aims)
	aim_feature_vectors = aims.reshape(shpa[0], shpa[1]*shpa[2])
	shpn= np.shape(non_aims)
	non_aim_feature_vectors = non_aims.reshape(shpn[0], shpn[1]*shpn[2])
	x = np.concatenate((aim_feature_vectors, non_aim_feature_vectors), axis = 0)
	if session == 'play':
		return x
	else:
		y = [1 if a < shpa[0] else 0 for a in range(np.shape(x)[0]) ]
		return x, y


if __name__ == '__main__':
	
	a, na = preprocess(get_data_ds2(set = 'train'))
	# a, na  = get_data_ds3(set = 'train')
	# a, na   = np.random.randint(0,100, size = np.shape(a))/1000.0,   np.random.randint(100,200, size = np.shape(na))/1000.0
	data, y = prepare_epocs(a, na)
	plot_ep(a, na)

	###########
	a, na = preprocess(get_data_ds2(set = 'test'))
	# a, na = get_data_ds3(set = 'test')
	# a, na   = np.random.randint(0,100, size = np.shape(a))/1000.0,   np.random.randint(100,200, size = np.shape(na))/1000.0
	data2, y2 = prepare_epocs(a,na)
	# plot_ep(a, na)

	lda=LDA(solver = 'lsqr', shrinkage='auto')
	lda.fit(data, y)

	answer = lda.predict(data2)

	###########
	print answer == y2
	print np.shape([np.array(y2) ==1])
	print (sum(answer[np.array(y2) ==1] == 1))/float(np.shape([np.array(y2) ==1])[0])
	print (sum(answer[np.array(y2) ==1] == 1) + sum(answer[np.array(y2) ==0] == 0))/float(len(y2))
	print sum(answer)
