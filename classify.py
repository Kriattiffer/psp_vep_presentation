import numpy as np
import socket
from matplotlib import pyplot  as plt 
from scipy import signal

class Classifier():
	"""docstring for Classifier"""
	def __init__(self, mapnames):
		pass
		# self.eeg = np.memmap(mmapnames['eeg'], dtype='float', mode='r', shape=(array_shape))
		# self.markers = np.memmap(mmapnames['markers'], dtype='float', mode='r', shape=(array_shape))


if __name__ == '__main__':
	mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
	Classifier(mapnames)