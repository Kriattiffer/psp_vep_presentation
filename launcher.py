import multiprocessing, sys, os, time
from tempfile import mkdtemp
import numpy as np
import present
import record
import classify


mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
top_exp_length = 60
number_of_channels = 9
# print mkdtemp()
def view():
	'''create stimulation window'''
	ENV = present.ENVIRONMENT()
	ENV.Fullscreen = True
	ENV.refresh_rate = 60
	ENV.stimuli_number = 6
	ENV.build_gui(monitor = present.mymon, rgb = ENV.rgb)
	# ENV.run_SSVEP_exp()
	ENV.run_P300_exp(stim_duration_FRAMES = 1, ISI_FRAMES = 2, waitforS = False)
	# ENV.run_P300_exp()

	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def rec():
	''' create stream class and start recording and plotting'''
	Stream = record.EEG_STREAM(mapnames = mapnames, plot_fft = False, plot_to_second_screen = True,
								top_exp_length = top_exp_length, number_of_channels  = number_of_channels)
	Stream.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def class_():
	Classifier = classify.Classifier(mapnames = mapnames, online = True,
									top_exp_length = top_exp_length, number_of_channels = number_of_channels)
	Classifier.mainloop()
	sys.stdout = open(str(os.getpid()) + ".out", "w")


if __name__ == '__main__':
	print 'startig GUI...'
	pgui = multiprocessing.Process(target=view)
	print 'startig backend...'
	prec = multiprocessing.Process(target=rec)
	print 'startig classifier...'
	pclass = multiprocessing.Process(target=class_)

	prec.start()
	pgui.start()
	time.sleep(4)
	pclass.start()

# example of usage for mmap 
# whait while object is created!
	# import time
	# time.sleep(7)
	# markers = np.memmap(mapnames['markers'], dtype='float32', mode='r', shape=(36000.0, 2))
	# while 1:
	# 	markerdata = markers[np.isnan(markers[:,1]) != True,:]
	# 	for a in markerdata:
	# 		print a
	# 	time.sleep(3)