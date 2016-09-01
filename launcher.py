import multiprocessing, sys # os???
from tempfile import mkdtemp
import numpy as np
import present
import record
import classify

mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
# print mkdtemp()
def view():
	'''create stimulation window'''
	ENV = present.ENVIRONMENT()
	ENV.Fullscreen = True
	ENV.refresh_rate = 120
	ENV.build_gui(monitor = present.mymon, rgb = ENV.rgb)
	# ENV.run_exp()
	ENV.run_P300_exp()
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def rec():
	''' create stream class and start recording and plotting'''
	Stream = record.EEG_STREAM(mapnames = mapnames, plot_fft = False, plot_to_second_screen = True)
	Stream.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def class_():
	Classifier = classify.Classifier(mapnames = mapnames)
	sys.stdout = open(str(os.getpid()) + ".out", "w")


if __name__ == '__main__':
	print 'startig GUI...'
	p1 = multiprocessing.Process(target=view)
	print 'startig backend...'
	p2 = multiprocessing.Process(target=rec)
	print 'startig classifier...'
	p2 = multiprocessing.Process(target=class_)

	p2.start()
	p1.start()

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