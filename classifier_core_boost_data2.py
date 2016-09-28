import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from record import butter_filt
from sklearn.lda import LDA
import random

sampling_rate = 256

#get data
def get_data_ds2(file = r'./training_set/s2.mat', set = 'train'):
	mat = scipy.io.loadmat(file)#, variable_names  = ['test', 'train'])
	mat = mat[mat.keys()[0]]
	mat = mat[0,0]
	test =  mat[set].T
	train = mat[set].T

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
	slices_aim = slices_aim[:,::2,:] #downsample to 64 hz 
	slices_non_aim = slices_non_aim[:,::2,:] #downsample to 64 hz 
	pass # windsorize
	slices_aim = slices_aim - slices_aim[:,0:1,:] # subtract 1st sample
	slices_non_aim = slices_non_aim - slices_non_aim[:,0:1,:] # subtract 1st sample
	print np.shape(slices_aim)
	print np.shape(slices_non_aim)
	sss = range(2625)
	random.shuffle(sss)
	slices_non_aim = slices_non_aim[sss[0:75],:,:]
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
		# axs.flatten()[a].plot(range(0, len(delta)*4, 4), slices_aim[:,:,a].T) #individual eps
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), delta, linewidth = 6)
		axs.flatten()[a].plot(range(0, len(delta)*4, 4), np.zeros(np.shape(delta))) #baseline
		axs.flatten()[a].set_title(channels[a])
	# plt.suptitle = 
	print 'averaged EP: N=%i' % np.shape(slices_aim)[0]
	plt.show()

class Classifer():
	"""docstring for Classifer"""
	def __init__(self, aims, non_aims):
		# self.lda=LDA()
		self.lda=LDA(solver = 'lsqr')
		self.aims, self.non_aims = aims, non_aims

	def prepare_epocs(self, aims, non_aims):
		shp = np.shape(aims)
		aims = aims.reshape(shp[-1]*shp[1], shp[0])
		shp = np.shape(non_aims)
		non_aims = non_aims.reshape(shp[-1]*shp[1], shp[0])
		# reshape epocs into lists of feature vectors
		self.x = np.concatenate((aims, non_aims), axis=1).T
		self.y = np.concatenate((np.ones(np.shape(aims)[1]), np.zeros(np.shape(non_aims)[1])),  axis=0)
		# self.x_train,self.y_train = self.subset(self.x,self.y)

	def subset(self, x,y):
		self.sample_ind = range(len(self.y))
		random.shuffle(self.sample_ind)
		return self.x[self.sample_ind[0:200]], self.y[self.sample_ind[0:200]]

	def train_classifier(self):
		self.lda.fit(self.x, self.y)
	
	def validate(self):
		tr1 = 0
		tr0 = 0
		fa1 = 0
		fa0 = 0
		for ind in range(len(self.y)):
			if self.lda.predict([self.x[ind]])[0] == self.y[ind]:
				if self.y[ind] == 1:
					tr1 +=1
				if self.y[ind] == 1:
					tr0 +=1
			else:
				if self.y[ind] == 1:
					fa1 +=1
				if self.y[ind] == 1:
					fa0 +=1
		print (tr1+tr0)/(float(fa1+fa0) + (tr1+tr0))

	def classify(self):
		pass


if __name__ == '__main__':
	
	eeg = get_data_ds2(set = 'test')
	sa, sna =  preprocess(eeg)
	# sa, sna =  get_data_ds1()
	# plot_ep(sa, sna)
	clf = Classifer(sa, sna)
	clf.prepare_epocs(clf.aims, clf.non_aims)
	clf.train_classifier()

	eeg = get_data_ds2(set = 'train')
	clf.aims, clf.non_aims = preprocess(eeg)
	clf.prepare_epocs(clf.non_aims, clf.aims)

	clf.validate()