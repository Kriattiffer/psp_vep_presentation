# -*- coding: utf-8 -*- 

import numpy as np
import os, sys, time, warnings, ast
from matplotlib import pyplot  as plt 
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.externals import joblib
from record import butter_filt
import socket
import subprocess
from multiprocessing import Process

class Classifier():
	"""docstring for Classifier"""
	def __init__(self, mapnames, online = False,
				top_exp_length = 60, number_of_channels = 8, classifier_channels = [],
				sampling_rate = 500, downsample_div = 10, saved_classifier = False, config = './circles.bcicfg'):
		
		self.LEARN_BY_DELTAS = False
		self.sampling_rate = sampling_rate
		self.downsample_div = downsample_div
		self.number_of_EEG_channels = number_of_channels
		self.channel_names = range(self.number_of_EEG_channels)
		
		self.SPEAK =True
		self.stimuli_names = ast.literal_eval(open(config).read())['names']
		
		if classifier_channels == []:
			self.classifier_channels = range(number_of_channels)
		else:
			assert (len(classifier_channels)<= number_of_channels), 'number of classifier channels cannot be larger then total number of channels' 
			assert not bool(sum([a >=number_of_channels for a in classifier_channels])), 'some classifier channels are not present in the data'
			assert len(classifier_channels) == len(set(classifier_channels)), 'there are duplicates in classifier channels'
			self.classifier_channels = classifier_channels

		self.x_learn, self.y_learn = [], [] #lists of feature vectors and class labels for learning
		self.mode = 'LEARN'

		self.sock = socket.socket() # socket for interaction with present.py
		self.sock.connect(('localhost', 22828))
		
		if saved_classifier: # if provided previouslly saved classifier, load it and skip learning stage
			self.mode = 'PLAY'
			self.CLASSIFIER = joblib.load(saved_classifier) 
		
		self.letter_counter = 0 # this is to know what stimuli is aim now for learning_stage
		self.learn_aims = np.genfromtxt('aims_learn.txt') -1
		
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels+1)  # include timestamp channel
		self.eegstream = np.memmap(mapnames['eeg'], dtype='float', mode='r', shape=(array_shape))
		self.markerstream = np.memmap(mapnames['markers'], dtype='float', mode='r', shape=(500*60*1.2, 2))
		self.im=self.create_stream() # LSL stream for markers

	def create_stream(self, stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
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
		'''Make array of epocs corresponding to single stimuli
			from raw EEG and markers '''

		def downsample(slices):
			'''Take every Nth EEG sample '''
			slices = slices[:,:,::self.downsample_div,:] #downsample  
			return slices

		def filter_eeg(EEG, frequencies = [1,20]):
			EEG[:,1:] = butter_filt(EEG[:,1:], frequencies, fs = self.sampling_rate) # filter
			return EEG

		EEG = filter_eeg(EEG)

		letters = [[] for a in codes]
		letter_slices = [[] for a in codes]
		for i,code in enumerate(codes):
			offs = MARKERS[MARKERS[:,1]==code][:,0]
			# print offs
			letters[i] = offs
			for off in offs:
				eegs = EEG[np.logical_and(  (EEG[:,0]*1000>off*1000), 
											(EEG[:,0]*1000<(off*1000+self.sampling_rate*2))),1:] # create 1-second epocs for each letter
				eegs = eegs - eegs[0,:] # make all slices start from 0
				letter_slices[i].append(eegs)
		letter_slices = np.array(letter_slices)
		letter_slices = downsample(letter_slices)
		return letter_slices

	def create_feature_vectors(self, letter_slices):
		'''Create feature vectors from letter slices.
			In learn mode returns array of feature vectors and list of class labels.
			In play mode  returns array of arrays of feature vectors for every stimuli '''
		shp = np.shape(letter_slices)
		lttrs = range(shp[0])

		if self.mode == 'PLAY':
			xes = [[] for a in lttrs]
			for letter in lttrs:
				aims = letter_slices[letter,:,:,:]
				shpa= np.shape(aims)
				# non_aims = letter_slices[[a for a in lttrs if a != letter]].reshape((shp[0]-1)*shp[1], shp[2], shp[3])
				# shpn= np.shape(non_aims)
				
				aim_feature_vectors = aims.reshape(shpa[0], shpa[1]*shpa[2], order = 'F')
				# non_aim_feature_vectors = non_aims.reshape(shpn[0], shpn[1]*shpn[2])
				if self.LEARN_BY_DELTAS == True:
					pass
				# x = np.concatenate((aim_feature_vectors, non_aim_feature_vectors), axis = 0)
				x = aim_feature_vectors
				xes[letter] = x
			xes = np.array(xes)
			# xes = np.average(xes, axis = 1)
			return xes

		elif self.mode == 'LEARN':
			aim_let = int(self.learn_aims[self.letter_counter])
			# print aim_let
			aims = letter_slices[lttrs[aim_let],:,:,:]
			non_aims = letter_slices[[a for a in lttrs if a != aim_let]].reshape((shp[0]-1)*shp[1], shp[2], shp[3])
			
			shpa= np.shape(aims)
			shpn= np.shape(non_aims)

			aim_feature_vectors = aims.reshape(shpa[0], shpa[1]*shpa[2], order = 'F') # usage of Fortran-like order is critical for plotting, although it dosen't really matter for classification
			non_aim_feature_vectors = non_aims.reshape(shpn[0], shpn[1]*shpn[2], order = 'F')
			
			if self.LEARN_BY_DELTAS == True:
				pass
			x = np.concatenate((aim_feature_vectors, non_aim_feature_vectors), axis = 0)
			y = [1 if a < shpa[0] else 0 for a in range(np.shape(x)[0]) ]
			self.letter_counter +=1
			return x, y


	def mainloop(self):
		''' Main cycle of Classifier class. 
			Waits for specific markes from present.py to start cutting EEG and classifying EPs'''
		trialstart = 0
		trialend = 0
		while  1:
			marker, timestamp_mark = self.im.pull_sample()
			if marker == [777]: # begining of letter trial
				trialstart = timestamp_mark
			if  marker == [888]: # end of letter trial
				trialend = timestamp_mark
			if marker == [888999]: # end of learning session
				self.mode = 'PLAY'
				print 'PLAY'
				x, y = self.xyprepare()
				self.learn_LDA(x, y)

				self.letter_counter = 0
				trialend, trialstart = 0,0

			if trialend > trialstart:
				print "TARGET CONFIRMED"
				with warnings.catch_warnings(): # >< operators generate warnings on arrays with NaNs, like our EEG array
					warnings.simplefilter("ignore")
					EEG = self.eegstream[np.logical_and(self.eegstream[:,0]>trialstart, 
														self.eegstream[:,0]<trialend),:]
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
					trialend, trialstart = 0,0


	def xyprepare(self):
		''' reshape matrix of feature vectors to fit classifier'''
		shp = np.shape(self.x_learn)
		x = np.array(self.x_learn).reshape(shp[0]*shp[1], shp[2])
		y = np.array(self.y_learn).flatten()
		return x, y

	def select_x_channels(self,x):
	 	ch_length = np.shape(x)[1]/self.number_of_EEG_channels
	 	indlist = sum([range(ch_length*chnl,ch_length*(chnl+1))  for chnl in self.classifier_channels], [])
	 	x =  x[:, indlist]
	 	print 'building classifier on '+ str(np.shape(x)) + ' shaped array'
		return x

	def validate_learning(self,x):
			print self.CLASSIFIER.predict(x)
			pass

	def plot_ep(self, x, y):
		''' get arrays of feature vectors and class labels;
			reshape them to lists of aim and non-aim epocs;	 plot averages for 8 channels '''
		if self.number_of_EEG_channels <=8:
			fig,axs = plt.subplots(nrows =3, ncols = 3)
		elif self.number_of_EEG_channels >=9:
			fig,axs = plt.subplots(nrows =4, ncols = 5)
		else :
			self.say_aloud(word = 'Too many 5 to plot.')
		
		xaim =x[y==1]
		xnonaim = x[y==0]
		avgxaim = np.average(xaim, axis = 0)
		avgxnonaim = np.average(xnonaim, axis = 0)
		avgaim = np.split(avgxaim, self.number_of_EEG_channels)
		avgnonaim = np.split(avgxnonaim, self.number_of_EEG_channels)
		avg_ep = [avgaim, avgnonaim]

		# channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz', 'HRz']
		for a in range((self.number_of_EEG_channels)):
			delta = avg_ep[0][a] - avg_ep[1][a]
			
			axs.flatten()[a].plot(range(len(delta)), avgaim[a]) # aim eps
			axs.flatten()[a].plot(range(len(delta)), avgnonaim[a]) # non-aim eps
			axs.flatten()[a].plot(range(len(delta)), delta, linewidth = 6) # delta
			axs.flatten()[a].plot(range(len(delta)), np.zeros(np.shape(delta))) #baseline
			axs.flatten()[a].set_title(self.channel_names[a])
		print 'averaged EP: N_aim=%i, N_nonaim=%i' % (sum(y==1), sum(y==0))
		plt.show()


	def learn_LDA(self, x, y):
		''' gets list of feature vectors and their class labels;
			plots ERPs;
			learns LDA;	saves LDA model to disk;
			sends packet to record.py process to allow start of online session with feedback
		'''
		print '\nBuilding classifier for ...'
		self.CLASSIFIER=LDA(solver = 'lsqr', shrinkage='auto')
		self.CLASSIFIER.fit(self.select_x_channels(x), y)
		self.plot_ep(x,y)
		print 'saving classifier...'
		joblib.dump(self.CLASSIFIER, 'classifier_%i.cls' %(time.time()*1000)) 
		print 'Starting online session'
		self.sock.send('startonlinesession')
		# self.validate_learning(x)

	def say_aloud(self, ans = False, index = False, word = False):
		if not word:
			word = self.stimuli_names[index] # name of max-scored stimuli
			if ans[index] == 0:
				word = 'No command selected'
		else:
			pass
		try:
			subprocess.call(['C:\\Program Files (x86)\\eSpeak\\command_line\\espeak.exe', word])
		except WindowsError:
			print 'install eSpeak for speech synthesys'

	def classify(self, xes):
		''' Function gets list of lists of feature vectors for all stimuli; 
		predicts classes for all vectors; returns index of the stimuli that scored maximum;
		says the name of corresponding command aloud using eSpeak;
			sends index of command to record.py process'''
		ans = []
		probs = []
		for vector in xes:
			answer = self.CLASSIFIER.predict(vector)
			prob = self.CLASSIFIER.predict_proba(vector)
			probs.append(prob)
			ans.append(sum(answer))
		probs = [np.prod(prob, axis = 0) for prob in probs] 
		probs =  [b[1]/(b[0]+ b[1]) for b in probs] # estimate probability of each stimuli being aim

		# np.set_printoptions(precision=5)
		np.set_printoptions(suppress=True)
		print probs
		print ans
		index = max(xrange(len(probs)), key = lambda x: probs[x]) # index of max-scored stimuli		
		# index = max(xrange(len(ans)), key = lambda x: ans[x]) # index of max-scored stimuli		
		if self.SPEAK ==True:
			self.say_aloud(ans, index)
		self.sock.send('answer is %i' %index) # start presentation for next aim stimuli
		return index
		# probs for every run  - if wouldnt work otherwise
	
	def test_offline(self):
		pass
		

if __name__ == '__main__':
	mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
	BLDA = Classifier(mapnames, online = True)
