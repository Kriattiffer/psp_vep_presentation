import numpy as np
import scipy.io
from record import butter_filt

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sampling_rate = 256
downsample_div = 4
averaging_bin = 1


def plot_ep(slices_aim, slices_non_aim):
	print np.shape(slices_aim)
	avgaim = np.average(slices_aim, axis = 0)
	avgnonaim = np.average(slices_non_aim, axis = 0)

	avg_ep = [avgaim[:,1:-2], avgnonaim[:,1:-2]]
	fig,axs = plt.subplots(nrows =3, ncols = 3)
	channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz', 'HRz']
	for a in range(8):
		delta = avg_ep[0][:,a] - avg_ep[1][:,a]
		# axs.flatten()[a].plot(range(0, len(delta)*4, 4), slices_aim[:,:,a].T) #individual eps
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), avgnonaim[:,a]) #individual eps

		axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6)
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
		axs.flatten()[a].set_title(channels[a])
	print 'averaged EP: N=%i' % np.shape(slices_aim)[0]
	plt.show()

def get_data_ds2(file = r'./training_set/s2.mat', set = 'train'):
	mat = scipy.io.loadmat(file)#, variable_names  = ['test', 'train'])
	mat = mat[mat.keys()[0]]
	mat = mat[0,0]
	test =  mat[set].T
	train = mat[set].T
	EEG = test
	return EEG

def preprocess(EEG):
	###filter data
	# EEG[:,:-2] = butter_filt(EEG[:,:-2], [1,9], fs = sampling_rate)
	### get epochs
	bb1 = EEG[:,-1] == 1 
	print np.shape(EEG)

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
	slices_aim = slices_aim[:,::downsample_div,:] #downsample to 64 hz  DUMB
	slices_non_aim = slices_non_aim[:,::downsample_div,:] #downsample to 64 hz DUMB 


	slices_aim = slices_aim - slices_aim[:,0:1,:] # subtract 1st sample
	slices_non_aim = slices_non_aim - slices_non_aim[:,0:1,:] # subtract 1st sample
	# print np.shape(slices_aim)
	# slices_aim = slices_aim[:,:,6:8]
	# slices_non_aim = slices_non_aim[:,:,6:8]
	print np.shape(slices_aim)

	return slices_aim, slices_non_aim

def average_eps(x, y):
	x1 = x[y==1]
	x0 = x[y==0]
	shp = np.shape(x1)
	dlt = shp[0]%averaging_bin
	if dlt !=0:
		x1 = x1[:-dlt, :]
	x1 = x1.reshape((averaging_bin, shp[0]/averaging_bin, shp[-1]))
	x1 = np.average(x1, axis = 0)

	shp = np.shape(x0)
	dlt = shp[0]%averaging_bin
	if dlt !=0:
		x0 = x0[:-dlt, :]
	x0 = x0.reshape((averaging_bin, shp[0]/averaging_bin, shp[-1]))
	x0 = np.average(x0, axis = 0)
	x = np.concatenate((x1, x0), axis = 0)
	y = np.concatenate((np.ones(np.shape(x1)[0], dtype = 'int'), np.zeros(np.shape(x0)[0], dtype = 'int')),  axis=0)
	return x, y

def prepare_epocs(aims, non_aims):
	shp= np.shape(aims)
	aim_feature_vectors = aims.reshape(shp[0], shp[1]*shp[2])
	shp= np.shape(non_aims)
	non_aim_feature_vectors = non_aims.reshape(shp[0], shp[1]*shp[2])
	print np.shape(aim_feature_vectors)
	print np.shape(non_aim_feature_vectors)

	# x, y = average_eps(x, y)
	print 'sdfdf'
	# print np.shape(x)
	return x, y


if __name__ == '__main__':
	
	# for a in range(2, 50):
		# downsample_div = a
	a, na = preprocess(get_data_ds2(set = 'train'))
	# a, na   = np.random.randint(0,100, size = np.shape(a))/1000.0,   np.random.randint(100,200, size = np.shape(na))/1000.0

	data, y = prepare_epocs(a, na)
	# a, na = np.ones(np.shape(a)), np.zeros(np.shape(na))
	# plot_ep(a, na)

	##########
	##########

	a, na = preprocess(get_data_ds2(set = 'test'))
	# a, na = np.ones(np.shape(a))+, np.zeros(np.shape(na))
	# a, na   = np.random.randint(0,100, size = np.shape(a))/1000.0,   np.random.randint(100,200, size = np.shape(na))/1000.0

	data2, y2 = prepare_epocs(a, na)


	# print np.shape(data)
	# print np.shape(data2)
	# print np.shape(y)

	lda=LDA(solver = 'lsqr', shrinkage='auto')
	lda.fit(data, y)
	# print lda
	# print  
	print sum(lda.predict(data2)[y2 ==1] == 1)/float(sum(y2[y2 ==1]))
	print y
	print np.shape(data2)
	print lda.predict(data2)

	plot_ep(a, na)
	a = np.average(data2[y2==1], axis = 0)
	na = np.average(data2[y2==0], axis = 0)

	a = a.reshape(np.shape(a)[0]/11, 11)
	na = na.reshape(np.shape(na)[0]/11, 11)
	print np.shape(a)
	a = np.repeat(a[np.newaxis,:], 75, axis = 0)
	na = np.repeat(na[np.newaxis,:], 75, axis = 0)
	plot_ep(a, na)
	print np.shape(a)

	# plt.plot(a - na)

	# plot_ep(a,na)
	# plt.plot(a)
	# plt.plot(na)
	# plt.plot(a - na, linewidth = 6)

	plt.show()
	# print
