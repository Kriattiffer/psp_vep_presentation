import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt


def slice_eeg(offsets,eeg):
		slices = [] 
		for offset in offsets:
			ind = np.argmax(eeg[:,0] > offset)
			slice = eeg[ind:ind+512]

			# slice = slice - np.average(slice, axis = 0) #?
			slice = slice - slice[0,:] #?
			
			if np.shape(slice)[0]<512:
				pass
			else:
				slices.append(slice)
		slices = np.average(slices, axis = 0)
		return np.array(slices)


def from_LSL():
	markers = np.genfromtxt('_markers.txt')
	eeg = np.genfromtxt('_data.txt')

	print np.shape(eeg)
	eeg[:,1:] = butter_filt(eeg[:,1:], (0.5,40))

	aim_list = [111,222,333,444]*12

	mmm = markers[:,1]==555
	mmm[-1] = True
	letter_slices = [[],[],[],[],[],[],[],[],[],[],[],[]]
	cc = -1

	for a , mrk in zip(mmm[:-1], markers):
		if a:
			cc +=1
		else:		
			letter_slices[cc].append(mrk)

	slices =  np.array(letter_slices)
	aim_eeg = []
	non_aim_eeg = []

	for aim, letter  in zip(aim_list, letter_slices):
		letter = np.array(letter)
		aim_offsets =  letter[letter[:,1] == aim][:,0]
		non_aim_offsets = letter[letter[:,1] != aim][:,0]
		
		aim_slices = slice_eeg(aim_offsets, eeg)
		non_aim_slices = slice_eeg(non_aim_offsets, eeg)

		aim_eeg.append(aim_slices)
		non_aim_eeg.append(non_aim_slices)

	aim_eeg = np.array(aim_eeg)
	non_aim_eeg = np.array(non_aim_eeg)

	# aim_eeg = np.average(aim_eeg, axis = 0)
	# non_aim_eeg = np.average(non_aim_eeg, axis = 0)
	# plt.plot(aim_eeg[:,1:] - non_aim_eeg[:,1:])
	# plt.show()

	for a in range(12):
		plt.plot(aim_eeg[a,:,1:] - non_aim_eeg[a,:,1:])
		plt.show()
def from_easyfile():
	eeg = np.genfromtxt('20160831193537_Patient01.easy')
	eeg[:,1:] = butter_filt(eeg[:,1:], (0.5,30))
	print np.shape(eeg)
	aim_list = [111,222,333,444]*12
	letter_slices = [[],[],[],[],[],[],[],[],[],[],[],[]]

from_LSL()
# from_easyfile()
