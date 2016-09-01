import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt
import os

def slice_eeg(offsets,eeg):
		slices = [] 
		for offset in offsets:
			ind = np.argmax(eeg[:,0] > offset) #+8
			slice = eeg[ind:ind+sample_length]

			# slice = slice - np.average(slice, axis = 0) #?
			# slice = slice - slice[0,:] #?
			
			if np.shape(slice)[0]<sample_length:
				pass
			else:
				slices.append(slice)
		slices = np.average(slices, axis = 0)
		return np.array(slices)


def from_LSL():
	markers = np.genfromtxt('_markers.txt')
	eeg = np.genfromtxt('_data.txt')

	print np.shape(eeg)
	eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,40))

	aim_list = [111,222,333,444]*12
	# aim_list = [111,222,333,444,555,666]*12

	mmm = markers[:,1]==mstart
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

	aim_eeg = np.average(aim_eeg, axis = 0)
	non_aim_eeg = np.average(non_aim_eeg, axis = 0)
	plt.plot(non_aim_eeg[:,1:])
	plt.title('LSL')

	# plt.show()
	plt.clf()
	return [aim_eeg[:,1:], non_aim_eeg[:,1:]]

	# for a in range(12):
	# 	plt.plot(aim_eeg[a,:,1:] - non_aim_eeg[a,:,1:])
	# 	plt.show()

def from_easyfile():
	eeg = np.genfromtxt([a for a in os.listdir('.') if 'Patient01.easy' in a][0])
	eeg[:,:-2] = butter_filt(eeg[:,:-2], (0.1,40))

	print np.shape(eeg)
	aim_list = [111,222,333,444]*12
	# aim_list = [111,222,333,444,555,666]*12

	letter_slices_ind = np.arange(np.shape(eeg)[0])[np.logical_or(eeg[:,-2] == mstart, eeg[:,-2] == mend )] 
	LETTERS = np.split(eeg, letter_slices_ind, axis = 0)[1:-1]
	print len(LETTERS)

	aim_eeg = []
	non_aim_eeg = []

	for LETTER, aim  in zip(LETTERS, aim_list):
		aim_ind = np.arange(np.shape(LETTER)[0])	 [LETTER[:,-2] == aim]
		non_aim_ind = np.arange(np.shape(LETTER)[0]) [np.logical_and(LETTER[:,-2] != aim, LETTER[:,-2] != 0)]
		# print aim_ind
		# print non_aim_ind
		for a in aim_ind:
			lett = LETTER[a:a+sample_length,:]
			if np.shape(lett)[0] == sample_length:
				aim_eeg.append(LETTER[a:a+sample_length,:])
		for a in non_aim_ind:
			lett = LETTER[a:a+sample_length,:]
			if np.shape(lett)[0] == sample_length:
				non_aim_eeg.append(lett)

	aim_eeg = np.array(aim_eeg)
	print np.shape(aim_eeg)
	print np.shape(non_aim_eeg)
	aim_eeg = np.average(aim_eeg, axis = 0)[:,:-2]
	non_aim_eeg = np.average(non_aim_eeg, axis = 0)[:,:-2]

	print np.shape(aim_eeg)
	plt.plot(non_aim_eeg)
	plt.title('easy')
	# plt.show()
	plt.clf()
	return [aim_eeg, non_aim_eeg]

		# aim_eeg = 
		# print aim_ind
mend = 666
mstart = 555
os.chdir('./_data/0109_squares/square_eeg_drl_4')
sample_length = 512

lsleeg = from_LSL()
efeeg = from_easyfile()
# plt.plot(lsleeg[1]-efeeg[1]/1000)
# plt.plot(lsleeg[0]-efeeg[0]/1000)

plt.plot(lsleeg[0]) 
plt.plot(efeeg[0]/1000) 
print 'plotted'
plt.show()
# a = np.array([0,1,0,1,0,2,1,0,1,3])
# print np.logical_or(a==1, a==2)