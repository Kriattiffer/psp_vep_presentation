import numpy as np
import socket
from matplotlib import pyplot  as plt 
from scipy import signal
pass

max_fft_freq = 100

data = np.genfromtxt('_data.txt')
markers = np.genfromtxt('_markers.txt')

def butter_filt(data, cutoff_array, fs = 500, order=4):
    nyq = 0.5 * fs
    normal_cutoff = [a /nyq for a in cutoff_array]
    b, a = signal.butter(order, normal_cutoff, btype = 'bandpass', analog=False)
    data = signal.filtfilt(b, a, data, axis = 0)
    return data

def PSDA(data_chunk):
	pass

offsets = []
for a in markers[:-1]:
	offsets.append(np.where(data[:,0]>a[0])[0][0])
letters =  np.reshape(offsets, (12,2))
print letters

data =  butter_filt(data, [4, 80])

for letter in letters:
	data_chunk = data[letter[0]: letter[-1],:]
	print  np.shape(data_chunk)
	FFT = np.zeros((max_fft_freq,9))
	# for i in range(2,5):
	# 	fft =  np.abs(np.fft.fft(data_chunk[500*i: 500*(i+1),:], axis = 0))[:max_fft_freq,:]
	# 	FFT +=fft
	# fft  = FFT/len(range(1,5))
	
	fft =  np.abs(np.fft.fft(data_chunk, axis = 0))[:max_fft_freq,:]
	fft = fft[:,1]
	
	x = np.arange(0, max_fft_freq/2, 0.5,)	
	plt.plot(x, fft)
	plt.title(np.argmax(fft)/2)
	plt.get_current_fig_manager().window.wm_geometry("-1920+0") # move FFT window tio second screen. Frame redraw in pesent.py starts to suck ==> possible problem with video card

	plt.show()
