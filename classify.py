import numpy as np
import sys, time, warnings
from matplotlib import pyplot  as plt 
from pylsl import StreamInlet, resolve_stream
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class Classifier():
	"""docstring for Classifier"""
	def __init__(self, mapnames, online = False,
				top_exp_length = 60, number_of_channels = 9, sampling_rate = 500):
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		self.eegstream = np.memmap(mapnames['eeg'], dtype='float', mode='r', shape=(array_shape))
		self.markerstream = np.memmap(mapnames['markers'], dtype='float', mode='r', shape=(500*60*1.2, 2))
		self.im =  self.create_stream()

		# self.lda=LDA(solver = 'lsqr', shrinkage='auto')
		self.data = False
		self.y = False
		print 'classifier template created.'

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

	def mainloop(self):
		trialstart = 0
		trialend = 0
		while  1:
			marker, timestamp_mark = self.im.pull_chunk()
			if [777] in marker:
				trialstart = timestamp_mark
			if  [888] in marker: # start new letter - 
				trialend = timestamp_mark
			if trialend > trialstart:
				print "TARGET CONFIRMED"
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					EEG = self.eegstream[np.logical_and(self.eegstream[:,0]>trialstart, self.eegstream[:,0]<trialend),:]
					MARKERS = self.markerstream[np.logical_and(self.markerstream[:,0]>trialstart, self.markerstream[:,0]<trialend),:]

				lnames = np.unique(MARKERS[:,1])
				letters = [[] for a in lnames]
				for i, lname in enumerate(lnames):
					print i, lname
					# letters[a]


	def get_data(self, start):
		def create_epocs():
			pass
		def preprocess_data():
			pass
		def create_feature_vectors_from_epocs():
			pass

	

	
	def validate_learning(self, number_of_reps):
		# classify the same data
		for a in range(number_of_reps):
			pass


	def learn(self, start):
		data, y = self.get_data(start = start, mode = 'learn')
		self.lda.fit(data, y)
		# validate_learning(number_of_reps)

	def classify(self, data2):
		answer = self.lda.predict(data2)
		# probs for every run  - if wouldnt work otherwise

		

if __name__ == '__main__':
	mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
	BLDA = Classifier(mapnames, online = True)
	# while 1:
		# Classifier.
		# pass

	# WAIT FOR INPUT
		# TAKE DATA
		# DO WHAT IN NESSESARY (LEARN/PLAY)
		# Sys.exit