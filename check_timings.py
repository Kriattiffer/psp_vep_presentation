import numpy as np
from matplotlib import pyplot as plt
""" Check reliability of time intervals between EEG offsets  and screen refreshes"""
def delta_t(file):
	array =  np.loadtxt(file)[:,0]	
	# plt.plot(array, 'o')
	delta_arr = array[1:] - array[:-1]
	# delta_arr =  np.round(delta_arr*1000)
	plt.plot(delta_arr)
	plt.show()

delta_t('_markers.txt')
# delta_t('_data.txt')
