import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt
from scipy import signal
import os

def slice_eeg(offsets,eeg, sample_length = 200):
	slices = [] 
	for offset in offsets:
		ind = np.argmax(eeg[:,0] > offset) #+8
		slice = eeg[ind:ind+sample_length,1]
		# slice = slice - np.average(slice, axis = 0) #?
		# slice = slice - slice[0,:] #?
		
		if np.shape(slice)[0]<sample_length:
			pass
		else:
			slices.append(slice)
	return np.array(slices)


def from_LSL(mend = 999, mstart = 777):
	markers = np.genfromtxt('./_markers.txt')

	eeg = np.genfromtxt('./_data.txt')
	# eeg[:,1:] = butter_filt(eeg[:,1:], (1,40))

	print np.shape(eeg)

	aims = [int(a)-1 for a in np.genfromtxt('aims_play.txt')]


	mmm = np.logical_and(markers[:,1]!=mstart, markers [:,-1] !=mend)

	markers =  markers[mmm]
	print np.shape(markers)
	letter_slices = [[] for a in range(np.shape(markers)[0])]
	offsts = markers[:,0]
	deltaof = offsts[1:] - offsts[:-1]
	deltaof = deltaof[deltaof<1]


	# plt.hist(deltaof)
	# plt.show()
	# plt.clf()
	# plt.plot(deltaof, 'o')
	# plt.show()
	# plt.clf()

	sleeg = slice_eeg(offsts, eeg)
	return sleeg

def from_plain_eeg():
	markers = np.genfromtxt('./_markers.txt')
	eeg = np.genfromtxt('./_data.txt')
	print np.shape(eeg)

	aims = [int(a)-1 for a in np.genfromtxt('aims_play.txt')]

	mmm = np.logical_and(markers[:,1]!=mstart, markers [:,-1] !=mend)

	markers =  markers[mmm]
	plt.plot(markers)
	plt.show()

	letter_slices = [[] for a in range(np.shape(markers)[0])]
	offsts = markers[:,0]
	deltaof = offsts[1:] - offsts[:-1]
	deltaof = deltaof[deltaof<3]
	# plt.plot(deltaof, 'o')
	# plt.show()
	sleeg = slice_eeg(offsts, eeg)
	return sleeg


slices = from_LSL()

print np.shape(slices)
plt.plot(slices.T)
slices = np.average(slices, axis = 0)
plt.plot(slices, linewidth = 6)

plt.show()
