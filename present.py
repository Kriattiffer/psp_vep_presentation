from psychopy import visual, core, event, monitors
from pylsl import StreamInfo, StreamOutlet
import os, sys, time, socket, random
import numpy as np


mymon = monitors.Monitor('Eizo', distance=48, width = 52.5)
# mymon = monitors.Monitor('zenbook', distance=18, width = 29.5)
mymon.setSizePix([1920, 1080])		
	# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state 60 fps
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps
base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,2,2,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps with red bits
# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0] # cvep
# base_mseq = [a for a in base_mseq for b in [0,0]] # extend sequence
base_mseq = np.array(base_mseq, dtype = int)


class ENVIRONMENT():
	""" class for visual stimulation during the experiment """
	def __init__(self):
		self.rgb = '#868686'
		self.stimcolor_p300 = [[self.rgb,self.rgb,self.rgb,self.rgb],['red', 'green', 'blue', 'pink']]
		self.stimcolor = ['white', 'white', 'white', 'white']
		self.Fullscreen = False
		self.window_size = (1000, 400)

		self. time_4_one_letter = 6
		self.number_of_inputs = 12
		self.refresh_rate = 120
		self.LSL = create_lsl_outlet() # create outlet for sync with NIC
		core.wait(1)
		self.LSL.push_sample([11]) # indicate the end of __init__

	def build_gui(self, monitor = mymon,
	 			  rgb = '#868686', stimrad = 2, fix_size = 1):
		''' function for creating visual enviroment. Input: various parameters of stimuli, all optional'''
		
		def create_circle(i):
			''' Call this to create identical circles, posing as stimulis'''
			circ = visual.Circle(self.win,
								radius=stimrad,
								lineColor = None, 
								fillColor = rgb,#self.stimcolor[i],
								units = 'deg',
								autoDraw=True,
								# autoLog=True,
								name=i
								)
			return circ

		# Create window
		self.win = visual.Window(fullscr = self.Fullscreen, 
							rgb = rgb,
							size = self.window_size,	
							monitor = monitor
							)

		# Crete stimuli
		self.cell1 = create_circle(0) 
		self.cell2 = create_circle(1)
		self.cell3 = create_circle(2)
		self.cell4 = create_circle(3)
		# self.cell5 = create_circle(4)
		# self.cell6 = create_circle(5)



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
		# self.cell5.pos = [-15, 0]
		# self.cell6.pos = [-15, 0]



	def flipper(self, N):
		'''function gets circe object, bit of the stimuli sequence (SSVEP or C-VEP), 
		and number of this bit, and decides what to do with circle on the next step'''
		# if n == 2:
		# 	CELL.fillColor = 'red'
		# 	return
		# if n == 3:
			# CELL.fillColor = 'red'
			# return
		if self.pattern[N][1][self.pattern[N][2]] ==1:
			self.pattern[N][0].fillColor = self.stimcolor[self.pattern[N][0].name]
			self.pattern[N][2] +=1
			if self.pattern[N][2] == len(self.pattern[N][1]):
				self.pattern[N][2] = 0
			else:
				pass
			return

		elif self.pattern[N][1][self.pattern[N][2]] ==0:
			self.pattern[N][0].fillColor = 'black'
			self.pattern[N][2] +=1
			if self.pattern[N][2] == len(self.pattern[N][1]):
				self.pattern[N][2] = 0
			else:
				pass
			return
	
	def exit_(self):
		''' exit and kill dependent processes'''
		self.LSL.push_sample([666])
		core.wait(0.5)
		sys.exit()

	def run_exp(self):
		'''Core function of the experiment. Defines GUI behavior and marker sending'''
		# create sequences for every cell; should be the same length!
		# seq1, seq2, seq3, seq4 = base_mseq, numpy.roll(base_mseq, 8), numpy.roll(base_mseq, 16), numpy.roll(base_mseq, 24) # CVEP

		steady_state_seqs = create_steady_state_sequences([6, 10, 12, 15], refresh_rate = self.refresh_rate)
		seq1, seq2, seq3, seq4 = steady_state_seqs[0], steady_state_seqs[1], steady_state_seqs[2], steady_state_seqs[3]
		self.pattern = [[self.cell1, seq1, 0], [self.cell2, seq2, 0], [self.cell3, seq3,0],[self.cell4, seq4,0 ]]

		tt = time.time()
		deltalist = ['','','','','','','','','','']
		
		while 's' not in event.getKeys():
			pass
		for a in range(self.number_of_inputs):
			timer = core.Clock()
			self.LSL.push_sample([222])  # indicate trial start
			flipcount = 0
			while timer.getTime() < self.time_4_one_letter: 
				if 'escape' in event.getKeys():
					self.exit_()
				for pair_num in range(len(self.pattern)):
					self.flipper(pair_num)
				self.win.flip()
				flipcount +=1
				
				if flipcount == self.refresh_rate:
					deltaT = int((1 + (tt - time.time()))*1000)
					deltaT = "{0:.1f}".format(round(deltaT,2))
					deltalist[1:] = deltalist[0:-1]
					deltalist[0] = deltaT
					print 'delta T:	%s ms \r' % str(deltalist),
					flipcount = 0
					tt = time.time()
			self.LSL.push_sample([333])  # indicate trial end

			print 'next letter\n'
			core.wait(2)

		self.exit_()


	def run_P300_exp(self, stim_duration_FRAMES = 6, ISI_FRAMES = 18, repetitions =  12):
		'P300 expreiment. Stimuli duration and interstimuli interval should be supplied as number of frames.'
		seq = [1]*stim_duration_FRAMES + [0]*ISI_FRAMES

		self.cells = [self.cell1,self.cell2,self.cell3,self.cell4]

		on = 0
		p300_markers_on =  [[111], [222],[333],[444]]
		# p300_markers_off =  [[0001], [2221],[3331],[4441]]
		while 's' not in event.getKeys():
			pass
		for letter in range(12):
			self.LSL.push_sample([555]) # input of new letter
			superseq = generate_p300_superseq(repetitions = repetitions)
			core.wait(2)

			deltalist = ['','','','','','','','','','']
			if 'escape' in event.getKeys():
				self.exit_()
			tt = time.time()
			for a in superseq:
				#check for Esc key

				self.cells[a].fillColor = self.stimcolor_p300[1][self.cells[a].name]
				self.win.flip()
				self.LSL.push_sample(p300_markers_on[a])
				for b in seq[1:]:
					self.cells[a].fillColor = self.stimcolor_p300[b][self.cells[a].name]
					self.win.flip()
				#acess timing accuracy
				deltaT = time.time() - tt
				deltaT = "{0:2.2f}".format(round((deltaT*1000)-200,2))
				print deltaT
				# deltalist[1:] = deltalist[0:-1]
				# deltalist[0] = deltaT
				# print 'delta T:%s ms \r' % str(deltalist),
				

				tt = time.time()
								
			print 'next letter'
		self.exit_()

def generate_p300_superseq(numbers = [0,1,2,3], repetitions = 10):
	''' receives IDs of stimuli, and number of repetitions, returns stimuli sequence without repeats'''
	seq = numbers*repetitions
	random.shuffle(seq) # generate random list
	dd_l =  [seq[a] for a in range(len(seq)) if seq[a] != seq[a-1]] #find duplicates
	dup_l =  [seq[a] for a in range(len(seq)) if seq[a] == seq[a-1]]
	for a in dup_l: # deduplicate
		p = [b for b in range(len(dd_l)) if dd_l[b] !=a and dd_l[b-1] !=a]
		dd_l.insert(p[1],a)
	return dd_l

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
	
	# Volosyak sequences 15 12 7.50 6.66
	if refresh_rate == 120:
		ss = ['00001111','0000111111','0000000011111111','000000001111111111']
	elif refresh_rate == 60:
		ss = ['0011','00111','00001111','000011111']

	ss_seqs = [[int(bit) for bit in [a for a in seq] ] for seq in ss]
	print ss_seqs
	return ss_seqs

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' Create outlet for Enobio. Requires name of the stream (same as in NIC) and MAC adress of the device. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])'''
	info = StreamInfo(name,'Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet


def view():
	'''function for multiprocessing'''
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)

if __name__ == '__main__':
	
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)