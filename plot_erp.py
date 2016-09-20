import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt
from scipy import signal
import os

def slice_eeg(offsets,eeg, sample_length = 300):
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

	# plt.plot(eeg[:,1])
	print np.shape(eeg)
	# plt.show()
	eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,40))

	aim_list = [111]
	aim_list = [111,222,333,444,555,666]*3
	aim_list = [111,222,333,444,]*47
	aim_list  = aim_list[0:47]


	mmm = markers[:,1]==mstart
	mmm[-1] = True
	letter_slices = [[] for a in range(len(aim_list)+1)]
	cc = -1

	for a , mrk in zip(mmm[:-1], markers):
		print cc

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
	# plt.plot(non_aim_eeg[:,1:])
	# plt.title('LSL')

	# plt.show()
	# plt.clf()
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
mend = 999
mstart = 777
# os.chdir('./_data/0109_squares/square_eeg_drl_5')
# os.chdir('./_data')
# eeg1 = np.genfromtxt([a for a in os.listdir('.') if 'Patient01.easy' in a][0])
# eeg1[:,:-2] = butter_filt(eeg[:,:-2], (0.1,40))
# markers = np.genfromtxt('_markers.txt')
# eeg2 = np.genfromtxt('_data.txt')




# eeg = np.genfromtxt('_data.txt')
# eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,40))
# # ex = eeg[:,0]
# # exx = ex[0:-2] - ex[1:-1]
# # print exx

# plt.plot(exx, 'o')
# plt.show()
# eeg = eeg[:,:2]
# print np.shape(eeg)
# print 'plotted'

# maxeeg = eeg[eeg[:,1]>20]
# ex = signal.argrelextrema(maxeeg[:,1], np.greater)
# ex = ex[0]
# # ex = maxeeg[ex][:,1]
# exx = ex[0:-2] - ex[1:-1]
# plt.plot(exx, 'o')
# plt.show()
# # 
############################################################
# sample_length = 250
# eeg = np.genfromtxt('_data.txt')[320:,:]
# print eeg
# print  np.shape(eeg)
# eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,40))

# nps = np.shape(eeg)
# # print nps

# cut = (nps[0] - nps[0]%sample_length)
# eeg = eeg[:cut,:]
# eeg = eeg.reshape(cut/sample_length, sample_length, nps[1])
# tmpl = np.average(eeg, axis = 0)
# print np.shape(eeg)
# # print tmpl[:,1:]
# # plt.show()

lsleeg = from_LSL()
# # efeeg = from_easyfile()
# # plt.plot(lsleeg[1]-efeeg[1]/1000)
# # plt.plot(lsleeg[0]-efeeg[0]/1000)
# # plt.plot(eeg2[:,1])
fig,axs = plt.subplots(nrows =3, ncols = 3)
channels = ['cpz', 'p3', 'p4', 'po3', 'poz', 'po4', 'o1', 'o2']
for a in range(8):
	# ax[a]
	delta = lsleeg[0][:,a] - lsleeg[1][:,a]
	axs.flatten()[a].plot(range(0, len(delta)*2, 2), delta)
	axs.flatten()[a].plot(range(0, len(delta)*2, 2), np.zeros(np.shape(delta)))

	axs.flatten()[a].set_title(channels[a])
	# delta = lsleeg[1][:,a]
	# axs.flatten()[a].plot(range(0, len(delta)*2, 2), delta)
	# axs.flatten()[a].plot(range(0, len(delta)*2, 2), np.zeros(np.shape(delta)))

	# plt.plot(lsleeg[1][:,a])
plt.show()
	# plt.clf()
# plt.plot(tmpl[:,1:])


# # plt.plot(efeeg[0]/1000) 
# print np.logical_or(a==1, a==2)