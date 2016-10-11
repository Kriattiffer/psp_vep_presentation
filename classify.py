import numpy as np
import sys, time, warnings
from matplotlib import pyplot  as plt 
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.externals import joblib
from record import butter_filt
import socket

class Classifier():
	"""docstring for Classifier"""
	def __init__(self, mapnames, online = False,
				top_exp_length = 60, number_of_channels = 9, 
				sampling_rate = 500, downsample_div = 20, saved_classifier = False):
	    

		self.sampling_rate = sampling_rate
		self.downsample_div = downsample_div
		self.x_learn, self.y_learn = [], []
		self.mode = 'LEARN'

		self.sock = socket.socket()
		self.sock.connect(('localhost', 22828))
		
		if saved_classifier:
			self.mode = 'PLAY'
			self.lda = joblib.load(saved_classifier) 
		
		self.letter_counter = 0
		self.learn_aims = np.genfromtxt('aims_learn.txt') -1
		print self.learn_aims
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		self.eegstream = np.memmap(mapnames['eeg'], dtype='float', mode='r', shape=(array_shape))
		self.markerstream = np.memmap(mapnames['markers'], dtype='float', mode='r', shape=(500*60*1.2, 2))
		self.im=  self.create_stream()

	def create_stream(self, stream_name_markers = 'CycleStart', stream_name_gui = 'GUI', recursion_meter = 0, max_recursion_depth = 3):
		''' Opens LSL stream for markers, If error, tries to reconnect several times'''
		
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
			recursion_meter +=1
		else:
			print 'exiting'
			sys.exit()
			inlet_markers = []

		print ("Classifier connecting to markers stream...")
		# inlet for markers
		if stream_name_markers in [stream.name() for stream in resolve_stream()]:
			sterams_markers = resolve_stream('name', stream_name_markers)
			inlet_markers = StreamInlet(sterams_markers[0])   
			try:
				inlet_markers
				print '...done \n'
			except NameError:
				print ("Error: Classifier cannot conect to markers stream\n")
				sys.exit()
		else:
			print 'Error: markers stream is not available\n'
			return self.create_stream(stream_name_markers,recursion_meter)
		return inlet_markers

	def prepare_letter_slices(self, codes, EEG, MARKERS):
		def downsample(slices):
			slices = slices[:,:,::self.downsample_div,:] #downsample  
			return slices
		EEG[:,1:] = butter_filt(EEG[:,1:], [1,20], fs = self.sampling_rate) # filter
		letters = [[] for a in codes]
		letter_slices = [[] for a in codes]
		for i,code in enumerate(codes):
			offs = MARKERS[MARKERS[:,1]==code][:,0]
			letters[i] = offs
			for off in offs:
				eegs = EEG[np.logical_and((EEG[:,0]*1000>off*1000), (EEG[:,0]*1000<(off*1000+self.sampling_rate*2))),1:] # create 1-second epocs for each letter
				eegs = eegs - eegs[0,:] # make all slices start from 0
				letter_slices[i].append(eegs)
		letter_slices = np.array(letter_slices)
		letter_slices = downsample(letter_slices)
		return letter_slices

	def create_feature_vectors(self, letter_slices):
		shp = np.shape(letter_slices)
		lttrs = range(shp[0])
		if self.mode == 'PLAY':
			xes = [[] for a in lttrs]
			for letter in lttrs:
				aims = letter_slices[letter,:,:,:]
				shpa= np.shape(aims)
				non_aims = letter_slices[[a for a in lttrs if a != letter]].reshape((shp[0]-1)*shp[1], shp[2], shp[3])
				aim_feature_vectors = aims.reshape(shpa[0], shpa[1]*shpa[2])
				shpn= np.shape(non_aims)
				non_aim_feature_vectors = non_aims.reshape(shpn[0], shpn[1]*shpn[2])
				x = np.concatenate((aim_feature_vectors, non_aim_feature_vectors), axis = 0)
				xes[letter] = x
				
			return np.array(xes)

		elif self.mode == 'LEARN':
			aim_let = int(self.learn_aims[self.letter_counter])
			print aim_let
			print lttrs
			aims = letter_slices[lttrs[aim_let],:,:,:]
			shpa= np.shape(aims)
			non_aims = letter_slices[[a for a in lttrs if a != aim_let]].reshape((shp[0]-1)*shp[1], shp[2], shp[3])
			aim_feature_vectors = aims.reshape(shpa[0], shpa[1]*shpa[2])
			shpn= np.shape(non_aims)
			non_aim_feature_vectors = non_aims.reshape(shpn[0], shpn[1]*shpn[2])
			x = np.concatenate((aim_feature_vectors, non_aim_feature_vectors), axis = 0)
			y = [1 if a < shpa[0] else 0 for a in range(np.shape(x)[0]) ]
			self.letter_counter +=1
			return x, y


	def mainloop(self):
		trialstart = 0
		trialend = 0
		while  1:
			marker, timestamp_mark = self.im.pull_sample()
			if marker == [777]:
				trialstart = timestamp_mark
			if  marker == [888]: # end of letter trial
				trialend = timestamp_mark
			if marker == [888999]: # end of learning session
				self.mode = 'PLAY'
				print 'PLAY'
				x, y = self.xyprepare()
				self.learn(x, y)
				self.letter_counter = 0
				trialend, trialstart = 0,0

			if trialend > trialstart:
				print "TARGET CONFIRMED"
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					EEG = self.eegstream[np.logical_and(self.eegstream[:,0]>trialstart, self.eegstream[:,0]<trialend),:]
					MARKERS = self.markerstream[np.logical_and( self.markerstream[:,0]>trialstart,
					 											self.markerstream[:,0]<trialend),:]
				lnames = np.unique(MARKERS[:,1])
				lnames = lnames[lnames!=777]
				lnames = lnames[lnames!=888]

				eeg_slices = self.prepare_letter_slices(lnames, EEG, MARKERS)
				if self.mode == 'LEARN':
					x,y = self.create_feature_vectors(eeg_slices)	
					self.x_learn.append(x), self.y_learn.append(y)
				elif self.mode == 'PLAY':
					xes = self.create_feature_vectors(eeg_slices)	
					self.classify(xes)

	def xyprepare(self):
		shp = np.shape(self.x_learn)
		x = np.array(self.x_learn).reshape(shp[0]*shp[1], shp[2])
		y = np.array(self.y_learn).flatten()
		return x, y

	def validate_learning(self,x):
			print self.lda.predict(x)
			pass

	def plot_letters_erps(self, x, y):
		xaim = x[y==1]
		xnonaim = x[y==0]
		print np.shape(xaim)
		print np.shape(xnonaim)

		xaim = np.average(xaim, axis = 0)
		xnonaim = np.average(xnonaim, axis = 0)
		print np.shape(xaim)
		print np.shape(xnonaim)
		plt.plot(xnonaim)
		plt.plot(xaim)
		plt.show()
		pass

	def learn(self, x, y):
		self.lda=LDA(solver = 'lsqr', shrinkage='auto')
		self.lda.fit(x, y)
		print 'saving classifier...'
		joblib.dump(self.lda, 'classifier_%i.cls' %(time.time()*1000)) 
		print 'Starting online session'
		self.sock.send('startonlinesession')
		print '22'
		self.plot_letters_erps(x, y)
		self.validate_learning(x)

	def classify(self, xes):
		print np.shape(xes)
		for vector in xes:
			print np.shape(vector)
			answer = self.lda.predict(vector)
			print answer
			self.sock.send('answer is blah blah blah')
		# probs for every run  - if wouldnt work otherwise

		

if __name__ == '__main__':
	mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
	BLDA = Classifier(mapnames, online = True)
