from pylsl import StreamInlet, resolve_stream
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import socket

class FFT_PLOT():
	"""class for real-time plotting of FFT fot every channel"""
	def __init__(self):
		''' here we createth plot object, to update it in loop. for performance we useth here powerful magic called 'blitting',  
		that helpeth us to redraw only data, while keenping backround and axes intact. How it worketh remains unknown >>> look into it later!'''
		
		channels = ['1','2','3','4','5','6','7','8'] # need to set from config file		
		self.sample_length = 1000 # number of samples to analyze
		T = 1/500.0				#sampling rate
		self.xf = np.linspace(0.0, 1.0/(2.0*T), self.sample_length/2)
		self.fig,self.axes = plt.subplots(nrows =3, ncols = 3, figsize = (15,10))
		self.axes = self.axes.flatten()[:-1]
		self.fig.show()
		self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
		self.lines = [ax.plot(np.arange(119), np.arange(0,1,1/119.0))[0] for ax in self.axes] #np.arange(0,1,1/119.0)
		[self.axes[i].set_title(channels[i], fontweight= 'bold',) for i in range(len(self.axes))]
		self.fig.canvas.draw()
		# return self.fig, lines, self.axes, backgrounds

	def update_fft(self, FFT):
		''' receives FFT vector, trimmes it to 119 points (whth 500 hz refresh rate it is 60 hz maximum), redraws plot.'''
		FFT = np.abs(FFT[0:119,:])
		for line, ax, background, channel  in zip(self.lines, self.axes, self.backgrounds, range(len(self.axes))):
			self.fig.canvas.restore_region(background)			
			line.set_ydata(FFT[:, channel])
			ax.draw_artist(line)
			self.fig.canvas.blit(ax.bbox)
		self.fig.canvas.start_event_loop(0.001) #0.1 ms seems enough

class EEG_STREAM(object):
	""" class for EEG\markers streaming, plotting and recording. """
	def __init__(self):
		''' create objects for later use'''
		self.stop = False 
		self.ie, self.im =  self.create_streams(StreamMarkers = True)
		self.EEG_ARRAY = self.create_array()
		self.MARKER_ARRAY = self.create_array(top_exp_length = 1, number_of_channels = 2)
		self.plot = FFT_PLOT()
		self.line_counter = 0
		self.line_counter_mark = 0
	
	def create_streams(self, stream_name_eeg = 'NIC', stream_name_markers = 'CycleStart', StreamEeg = True, StreamMarkers = True):
		''' Opens two LSL streams: one for EEG, another for markers'''
		inlet_eeg = []; inlet_markers = []
		
		if StreamEeg == True:
			print ("Connecting to NIC stream...")
			streams_eeg = resolve_stream('type', 'EEG')
			inlet_eeg = StreamInlet(streams_eeg[0])   
			try:
				inlet_eeg
				print '...done \n'
			except NameError:
				print ("Error: NIC stream not available\n")
		if StreamMarkers == True:
			print ("Connecting to Psychopy stream...")
			sterams_markers = resolve_stream('name', stream_name_markers)	
			inlet_markers = StreamInlet(sterams_markers[0])   
			try:
				inlet_markers
				print '...done \n'
			except NameError:
				print ("Error: Psychopy stream not available\n")

		return inlet_eeg, inlet_markers
	
	def create_array(self, top_exp_length = 60, number_of_channels  = 9):
		'''Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes'''
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		print 'Creating array with dimensions %s...' %str(array_shape) 
		a = np.zeros(array_shape, dtype = 'float')
		a[:,0:9] = np.NAN
		print '... done'
		return a

	def fill_array(self, eeg_array, line_counter, data_chunk, timestamp_chunk, datatype = 'EEG'):
		'''Recieves preallocated array of NaNs, piece of data, piece of offsets and number of line, inserts everything into array. Works both with EEG and with markers '''
		length = len(timestamp_chunk)
		eeg_array[line_counter:line_counter+length, 0] = timestamp_chunk
		eeg_array[line_counter:line_counter+length,1:] = data_chunk
	
	def plot_and_record(self):
		''' Main cycle for recording and plotting. Pulls markers and eeg from lsl inlets, 
		fills preallocated arrays with data. After certain offset calculates FFT and updates plots. Records data on exit.'''
		while self.stop != True:	# set EEG_STREAM.stop to False to stop sutpid games and flush arrays to disc.
			marker, timestamp_mark = self.im.pull_chunk()
			EEG, timestamp_eeg = self.ie.pull_chunk()
			if timestamp_eeg:
				self.fill_array(self.EEG_ARRAY, self.line_counter, EEG, timestamp_eeg, datatype = 'EEG')
				self.line_counter += len(timestamp_eeg)
				if self.line_counter>1000 and self.line_counter % 20 == 0:
					FFT = compute_fft(self.EEG_ARRAY, self.line_counter, sample_length = 1000)
					self.plot.update_fft(FFT)
			if timestamp_mark:
				self.line_counter_mark += len(timestamp_mark)
				self.fill_array(self.MARKER_ARRAY, self.line_counter_mark, marker[0], timestamp_mark, datatype = 'MARKER')

		print 'saving experiment data...'
		np.savetxt('data.txt', self.EEG_ARRAY, fmt= '%.4f')
		np.savetxt('markers.txt', self.MARKER_ARRAY, fmt= '%.4f')
		print '...data saved.'
		# sys.exit()


def compute_fft(EEG_ARRAY,offset, sample_length = 1000):
	''' computes fourier transform from slice of EEG_ARRAY. slice is determined by current position and length of the sample to analyze.
	FT should be somehow normalized to fit into graph window - how?'''
	fft = np.fft.fft(EEG_ARRAY[offset-sample_length:offset,1:], axis = 0)
	# print np.shape(fft)
	#normalize to one:	
	# fft = fft/np.sum(fft)
	# fft = fft/np.max(fft)
	fft = fft/100000
	return fft

if __name__ == '__main__':
	Stream = EEG_STREAM()
	Stream.plot_and_record()
