import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from record import butter_filt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
	downsample_div = 4
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
	slices_aim = slices_aim[:,::downsample_div,:] #downsample to 64 hz 
	slices_non_aim = slices_non_aim[:,::downsample_div,:] #downsample to 64 hz 
	pass # windsorize
	slices_aim = slices_aim - slices_aim[:,0:1,:] # subtract 1st sample
	slices_non_aim = slices_non_aim - slices_non_aim[:,0:1,:] # subtract 1st sample
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
	print 'averaged EP: N=%i' % np.shape(slices_aim)[0]
	plt.show()

class Classifer():
	"""docstring for Classifer"""
	def __init__(self, aims, non_aims):
		# self.lda=LDA()
		# self.lda=LDA(solver = 'lsqr')
		self.lda=LDA(solver = 'lsqr', shrinkage='auto')

		self.aims, self.non_aims = aims, non_aims

	def prepare_epocs(self, aims, non_aims):
		shp = np.shape(aims)
		aims = aims.reshape(shp[-1]*shp[1], shp[0])
		shp = np.shape(non_aims)

		non_aims = non_aims.reshape(shp[-1]*shp[1], shp[0])
		# reshape epocs into lists of feature vectors
		self.x = np.concatenate((aims, non_aims), axis=1).T
		self.y = np.concatenate((np.ones(np.shape(aims)[1]), np.zeros(np.shape(non_aims)[1])),  axis=0)
		self.x, self.y = self.average_eps(self.x, self.y)

	def subset(self):
		# equalize number of aims and nonaims
		x,y = self.x, self.y
		n_1 = int(np.sum(y))
		x_1 = x[y==1]
		x_0_ind = np.random.choice(np.shape(x[y==0])[0], n_1, replace = False)
		x_0 = x[y==0][x_0_ind,:]
		self.x = np.concatenate((x_1, x_0), axis = 0)
		# print np.shape(self.x)
		self.y = np.concatenate((np.ones(n_1), np.zeros(n_1)),  axis=0)

	def average_eps(self,x, y):
		averaging_bin = 5
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
		print y
		return x, y

	def train_classifier(self):
		print np.shape(self.x)
		print self.x[1]
		self.lda.fit(self.x, self.y)
	
	def validate(self):
		print 'start crossvalidation \n'
		tr1 = 0
		tr0 = 0
		fa1 = 0
		fa0 = 0
		print self.x[16], self.y[16]
		for ind in range(len(self.y)):
			EP = [self.x[ind]]
			if self.classify(EP) == self.y[ind]:
				if self.y[ind] == 1:
					tr1 +=1
				elif self.y[ind] == 0:
					tr0 +=1
			else:
				if self.y[ind] == 1:
					fa1 +=1
				elif self.y[ind] == 0:
					fa0 +=1
		print tr1, tr0, fa1, fa0, (tr1+tr0)/(float(fa1+fa0) + float(tr1+tr0)), tr0/float(tr0 + fa0), tr1/float(tr1 + fa1)

	def classify(self, EP):
		return self.lda.predict(EP)[0]


if __name__ == '__main__':
	
	eeg = get_data_ds2(set = 'test')
	sa, sna =  preprocess(eeg)
	sa, sna = np.ones(np.shape(sa)), np.zeros(np.shape(sna))

	print 'building classifier \n'
	clf = Classifer(sa, sna)
	clf.prepare_epocs(clf.aims, clf.non_aims)
	clf.subset()
	# print clf.x, clf.y
	# print clf.aims, clf.non_aims

	clf.train_classifier()

	eeg = get_data_ds2(set = 'train')
	clf.aims, clf.non_aims = preprocess(eeg)
	clf.aims, clf.non_aims = np.ones(np.shape(clf.aims)), np.zeros(np.shape(clf.non_aims))


	clf.prepare_epocs(clf.aims, clf.non_aims)
	# clf.subset()
	
	# print clf.x, clf.y
	
	clf.validate()