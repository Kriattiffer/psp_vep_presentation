import multiprocessing, sys
import present
import record


def view():
	ENV = present.ENVIRONMENT()
	ENV.build_gui(monitor = present.mymon)
	ENV.run_exp(present.base_mseq)
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def rec():
	Stream = record.EEG_STREAM()
	Stream.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

if __name__ == '__main__':
	p1 = multiprocessing.Process(target=view)
	p1.start()
	p2 = multiprocessing.Process(target=rec)
	p2.start()