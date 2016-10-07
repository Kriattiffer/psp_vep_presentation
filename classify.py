import numpy as np
import socket
from matplotlib import pyplot  as plt 
from scipy import signal

class Classifier():
	"""docstring for Classifier"""
	def __init__(self, mapnames, online = False):
		if online:
			self.eeg = np.memmap(mmapnames['eeg'], dtype='float', mode='r', shape=(array_shape))
			self.markers = np.memmap(mmapnames['markers'], dtype='float', mode='r', shape=(array_shape))
		self.lda=LDA(solver = 'lsqr', shrinkage='auto')
		self.data = False
		self.y = False
		# PARAMS = PARAMS
		# classifier = classifier
	def get_data(self, stat):
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
	Classifier(mapnames)
	# WAIT FOR INPUT
		# TAKE DATA
		# DO WHAT IN NESSESARY (LEARN/PLAY)
		# Sys.exit