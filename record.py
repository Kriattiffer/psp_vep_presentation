from pylsl import StreamInlet, resolve_stream
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import sys, os

class FFT_PLOT():
	"""class for real-time plotting of FFT fot every channel"""
	def __init__(self, max_fft_freq=140):
		''' here we createth plot object, to update it in loop. for performance we useth here powerful magic called 'blitting',  
			that helpeth us to redraw only data, while keenping backround and axes intact. How it worketh remains unknown >>> look into it later!
			max_fft_freq is DOUBLED cutoff frequency for fft_plot.'''

		channels = ['1','2','3','4','5','6','7','8'] # need to set from config file		
		self.sample_length = 1000 # number of samples to analyze
		T = 1/500.0				#sampling rate
		self.max_fft_freq = max_fft_freq

		### create_plot ###
		self.fig,self.axes = plt.subplots(nrows =3, ncols = 3, figsize = (15,10))
		self.axes = self.axes.flatten()[:-1]
		# plt.get_current_fig_manager().window.wm_geometry("-1920+0") # move FFT window tio second screen. Frame redraw in pesent.py starts to suck ==> possible problem with video card
		self.fig.show()
		self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
		x = np.arange(0, self.max_fft_freq/2, 0.5,)
		y = np.arange(0,1,1.0/self.max_fft_freq)
		self.lines = [ax.plot(x,y)[0] for ax in self.axes] #np.arange(0,1,1/119.0)	
		[self.axes[i].set_title(channels[i], fontweight= 'bold',) for i in range(len(self.axes))]
		self.fig.canvas.draw()

	def update_fft(self, FFT):
		''' receives FFT vector, trimmes it to several points (whth 500 hz refresh rate it is 60 hz maximum), redraws plot.'''
		if not plt.fignum_exists(1): # dosent't try to update closed figure # careful with additional figures!
			return
		FFT = np.abs(FFT[:self.max_fft_freq,:])
		for line, ax, background, channel  in zip(self.lines, self.axes, self.backgrounds, range(len(self.axes))):
			self.fig.canvas.restore_region(background)			
			line.set_ydata(FFT[:, channel])
			# print np.argmax(FFT[:, channel])
			ax.draw_artist(line)
			self.fig.canvas.blit(ax.bbox)
		self.fig.canvas.start_event_loop(0.001) #0.1 ms seems enough

class EEG_STREAM(object):
	""" class for EEG\markers streaming, plotting and recording. """
	def __init__(self,  StreamEeg = True, StreamMarkers = True, plot_fft = True):
		''' create objects for later use'''
		self.StreamEeg, self.StreamMarkers = StreamEeg, StreamMarkers
		self.plot_fft = plot_fft
		self.stop = False 
		self.ie, self.im =  self.create_streams()
		self.EEG_ARRAY = self.create_array()
		self.MARKER_ARRAY = self.create_array(top_exp_length = 1, number_of_channels = 2)
		self.line_counter = 0
		self.line_counter_mark = 0
		if self.plot_fft == True:
			self.plot = FFT_PLOT()

	def create_streams(self, stream_type_eeg = 'EEG', stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
		''' Opens two LSL streams: one for EEG, another for markers, If error, tries to reconnect several times'''
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
			recursion_meter +=1
		else:
			print 'exiting'
			plt.close()
			sys.exit()
		inlet_eeg = []; inlet_markers = []
		
		if self.StreamEeg == True:
			print ("Connecting to NIC stream...")
			if stream_type_eeg in [stream.type() for stream in resolve_stream()]:
				streams_eeg = resolve_stream('type', 'EEG')
				inlet_eeg = StreamInlet(streams_eeg[0])   
				try:
					inlet_eeg
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to NIC stream\n")
					plt.close()
					sys.exit()
			else:
				print 'Error: NIC stream is not available\n'
				plt.close()
				sys.exit()
		else:
			inlet_eeg = []

		if self.StreamMarkers == True:
			print ("Connecting to Psychopy stream...")
			if stream_name_markers in [stream.name() for stream in resolve_stream()]:
				sterams_markers = resolve_stream('name', stream_name_markers)
				inlet_markers = StreamInlet(sterams_markers[0])   
				try:
					inlet_markers
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to Psychopy stream\n")
					plt.close()
					sys.exit()
			else:
				print 'Error: Psychopy stream is not available\n'
				return self.create_streams(stream_type_eeg, stream_name_markers, StreamEeg, StreamMarkers, recursion_meter)
		else:
			inlet_markers = []
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
			# pull chunks if Steam_eeg and stream_markess are True
			if self.StreamMarkers ==True:
				marker, timestamp_mark = self.im.pull_chunk()
			else :
				marker, timestamp_mark = [],[]

			if self.StreamEeg == True:
				EEG, timestamp_eeg = self.ie.pull_chunk()
			else:
				EEG, timestamp_eeg = [], []

			if timestamp_eeg:
				self.fill_array(self.EEG_ARRAY, self.line_counter, EEG, timestamp_eeg, datatype = 'EEG')
				self.line_counter += len(timestamp_eeg)
				if self.line_counter>1000 and self.line_counter % 20 == 0 and self.plot_fft == True:
					FFT = compute_fft(self.EEG_ARRAY, self.line_counter, sample_length = 1000)
					self.plot.update_fft(FFT)
			if timestamp_mark:
				self.line_counter_mark += len(timestamp_mark)
				self.fill_array(self.MARKER_ARRAY, self.line_counter_mark, marker[0], timestamp_mark, datatype = 'MARKER')				
				if marker == [[666]]:
					self.stop = True
					plt.close() # oherwise get Fatal Python error: PyEval_RestoreThread: NULL tstate
					print '\nsaving experiment data...\n'
					eegdata = self.EEG_ARRAY[np.isnan(self.EEG_ARRAY[:,1]) != True,:]  # delete all unused lines from data matrix
					markerdata = self.MARKER_ARRAY[np.isnan(self.MARKER_ARRAY[:,1]) != True,:]
					np.savetxt('_data.txt', eegdata, fmt= '%.4f')
					np.savetxt('_markers.txt', markerdata, fmt= '%.4f')
					print '\n...data saved.\n Goodbye.\n'
		sys.exit()


def compute_fft(EEG_ARRAY,offset, sample_length = 1000):
	''' computes fourier transform from slice of EEG_ARRAY. slice is determined by current position and length of the sample to analyze.
	FT should be somehow normalized to fit into graph window - how?'''
	ARRAY_SLICE =  EEG_ARRAY[offset-sample_length:offset,1:]
	fft = np.fft.fft(ARRAY_SLICE, axis = 0)
	# print np.shape(fft)
	#normalize to one:	
	# fft = fft/np.sum(fft)
	# fft = fft/np.max(fft)
	fft = fft/100000
	return fft


# os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

if __name__ == '__main__':
	Stream = EEG_STREAM( plot_fft = True, StreamMarkers = False)
	Stream.plot_and_record()