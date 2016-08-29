import multiprocessing, sys
import numpy as np
import present
import record


def view():
	'''create stimulation window'''
	ENV = present.ENVIRONMENT()
	# ENV.Fullscreen = True
	ENV.refresh_rate = 60
	ENV.build_gui(monitor = present.mymon)
	ENV.run_exp(present.base_mseq)
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def rec():
	''' create stream class and start recording and plotting'''
	Stream = record.EEG_STREAM(plot_fft = False, plot_to_second_screen = False)
	Stream.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

if __name__ == '__main__':
	print 'startig GUI...'
	p1 = multiprocessing.Process(target=view)
	print 'startig backend...'
	p2 = multiprocessing.Process(target=rec)
	# p2.start()
	p1.start()

	# print np.shape(Stream.EEG_ARRAY)

