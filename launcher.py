import multiprocessing, sys, os, time
from tempfile import mkdtemp
import numpy as np
import present
import record
import classify

mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
top_exp_length = 60
number_of_channels = 9
savedclass = False
savedclass = 'classifier_1476717624752.cls'

def view():
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT()
	ENV.Fullscreen = True
	ENV.refresh_rate = 60
	ENV.stimuli_number = 6
	ENV.build_gui(monitor = present.mymon, rgb = ENV.rgb)
	if savedclass:
		ENV.LEARN = False
		print 'Using saved classifier from %s' % savedclass
	else:
		print 'Buildindg new classifier'
	ENV.run_P300_exp(stim_duration_FRAMES = 15, ISI_FRAMES = 5, 
					waitforS = False, repetitions=100)
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def rec():
	''' Create stream class and start recording and plotting'''
	STRM = record.EEG_STREAM(mapnames = mapnames, plot_fft = False, plot_to_second_screen = True,
								top_exp_length = top_exp_length, number_of_channels  = number_of_channels)
	STRM.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def class_():
	'''Create classifer class and wait for markers from present.py'''
	CLSF = classify.Classifier(mapnames = mapnames, online = True,
									top_exp_length = top_exp_length, 
									number_of_channels = number_of_channels, 
									saved_classifier = savedclass)
	CLSF.mainloop()
	sys.stdout = open(str(os.getpid()) + ".out", "w")


if __name__ == '__main__':
	pgui = multiprocessing.Process(target=view)
	prec = multiprocessing.Process(target=rec)
	pclass = multiprocessing.Process(target=class_)
	
	print 'startig backend...'
	prec.start()
	print 'startig GUI...'
	pgui.start()
	time.sleep(4)
	print 'startig classifier...'
	# pclass.start()