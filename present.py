from psychopy import visual, core, event, monitors
from pylsl import StreamInfo, StreamOutlet
import os, sys, time, socket
import numpy

# mymon = monitors.Monitor('Eizo', distance=50, width = 52.5)
mymon = monitors.Monitor('zenbook', distance=25, width = 29.5)
mymon.setSizePix([1920, 1080])	
Fullscreen = False

# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state 60 fps
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps
base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,2,2,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps with red bits
# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0] # cvep
# base_mseq = [a for a in base_mseq for b in [0,0]] # extend sequence
base_mseq = numpy.array(base_mseq, dtype = int)



class ENVIRONMENT():
	""" class for visual stimulation during the experiment """
	
	 			  rgb = '#868686', stimrad = 2, stimcolor = 'red', fix_size = 1):
		''' function for creating visual enviroment. Input: various parameters of stimuli, all optional'''
		
		def create_circle():
			''' Call this to create identical circles, posing as stimulis'''
			circ = visual.Circle(self.win,
								radius=stimrad,
								lineColor = None, 
								fillColor = stimcolor,
								units = 'deg',
								autoDraw=True
								)
			return circ

		# Create window
		self.win = visual.Window(fullscr = Fullscreen, 
							rgb = rgb,
							size = size,
							monitor = monitor
							)

		# Crete stimuli
		self.cell1 = create_circle() 
		self.cell2 = create_circle()
		self.cell3 = create_circle()
		self.cell4 = create_circle()

		# Create fixation cross
		self.fixation = visual.ShapeStim(self.win,  							
								vertices=((0, -1*fix_size), (0, fix_size), (0,0), 
										  (-1*fix_size,0), (fix_size, 0)),
								units = 'deg',
								lineWidth=5,
								closeShape=False,
								lineColor='white',
								autoDraw=True
								)

		# position circles over board. units are taken from the create_circle function
		self.cell1.pos = [0, 15]
		self.cell2.pos = [15, 0]
		self.cell3.pos = [0, -15]
		self.cell4.pos = [-15, 0]


	def flipper(self, CELL, bit, n):
		'''function gets circe object, bit of the stimuli sequence, 
		and number of this bit, and decides what to do with circle on the next step'''
		# if n == 2:
		# 	CELL.fillColor = 'red'
		# 	return
		if bit == 2:
			CELL.fillColor = 'red'
			return
		if bit ==1:
			CELL.fillColor = 'white'
			return
		elif bit == 0:
			CELL.fillColor = 'black'
			return

	def run_exp(self, base_mseq):
		'''Core function of the experiment. Defines GUI behavior and marker sending'''
		# create sequences for every cell; should be the same length!
		# seq1, seq2, seq3, seq4 = base_mseq, numpy.roll(base_mseq, 8), numpy.roll(base_mseq, 16), numpy.roll(base_mseq, 24) # CVEP
		steady_state_seqs = create_steady_state_sequences([6, 10, 12, 15])
		seq1, seq2, seq3, seq4 = steady_state_seqs[0], steady_state_seqs[1], steady_state_seqs[2], steady_state_seqs[3]
		pattern = [(self.cell1, seq1), (self.cell2, seq2), (self.cell3, seq3), (self.cell4, seq4)]
		
		LSL = create_lsl_outlet() # create outlet for sync with NIC
		core.wait(1)
		LSL.push_sample([11]) # indicate the begining of trial

		tt = time.time()
		while  1:
			if 'escape' in event.getKeys():
				sys.exit()
			
			LSL.push_sample([111])  # sync with EEG

			for bit_number in range(len(seq4)): # cycle through sequences 
				for pair in pattern: # decide what to do with every circle at the next refresh
					self.flipper(pair[0], pair[1][bit_number], bit_number)
				
				self.win.flip() # refresh screen
			
			print (2 + (tt - time.time()))*1000 # difference between desired and real time
			tt = time.time()


def create_steady_state_sequences(freqs, refresh_rate = 120, limit_pulse_width = None, phase_rotate = None):
	'''Function receives list of frequencies and screen refresh_rate, returns list of sequences, corresponding to this frequencies. 
	   Frequences should be disisors of refresh_rate/2
	   TODO:
	   If limit_pulse_width (int) is specified, number of consequitive 'ones' is limited by this number 
	   If duty_cycles (list) is specified, sequences will have corresponding duty cycles
	   If phase_rotate (list) is specified, sequences will be shifted by corresponding fraction of its period  '''
	ss_seqs = []
	for freq in freqs:
		ss_seq = [a  for a in [1,0]*freq for b in range(refresh_rate/2/freq)]
		ss_seqs.append(ss_seq)
	return ss_seqs

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' Create outlet for Enobio. Requires name of the stream (same as in NIC) and MAC adress of the device. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])'''
	info = StreamInfo(name,'Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet

if __name__ == '__main__':
	
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)