import os, sys, time, socket, random, datetime, socket
from psychopy import visual, core, event, monitors
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from psychopy.tools.monitorunittools import posToPix

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
	def __init__(self, DEMO = False):
		# self.input_p.close()

		self.rgb = '#868686'
		self.stimcolor_p300 = [[self.rgb,self.rgb,self.rgb,self.rgb,self.rgb,self.rgb],['red', 'green', 'blue', 'pink', 'yellow', 'purple']]
		self.stimcolor = ['white', 'white', 'white', 'white']
		self.Fullscreen = False
		self.window_size = (1920, 1080)
		self.LEARN = True
		self. time_4_one_letter = 6 #steadystate
		self.stimuli_number = 6		
		self.number_of_inputs = 12
		self.refresh_rate = 120

		if DEMO == True:
			self.LSL, self.conn = self.fake_lsl_and_conn()

		elif DEMO == False:
			self.LSL = create_lsl_outlet() # create outlet for sync with NIC
			core.wait(1)		
			sck = socket.socket() # cant make Pipe() work
			sck.bind(('localhost', 22828))
			sck.listen(1)
			self.conn, addr = sck.accept()	
			print 'Classifier socket connected'
		
	def fake_lsl_and_conn(self):
		fakeLSL = create_lsl_outlet() # create outlet for sync with NIC - dosen't need connection
		class Fakeconn(object):
			"""dummy connection class with recv() method"""
			def __init__(self):
				pass
			def recv(self, arg):
				'''returns different strings in different enviroments 
					depending on number of bits input'''
				if arg == 1024:
					return 'answer'
				elif arg == 2048:
					return 'startonlinesession'
		return fakeLSL, Fakeconn()
				


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
							rgb = '#868686',
							size = self.window_size,	
							monitor = monitor
							)

		# Crete stimuli
		self.cell1 = create_circle(0) 
		self.cell2 = create_circle(1)
		self.cell3 = create_circle(2)
		self.cell4 = create_circle(3)
		if self.stimuli_number == 6:
			self.cell5 = create_circle(4)
			self.cell6 = create_circle(5)
		else:
			self.cell5, self.cell6 = False, False
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

		self.cell1.pos = [0, 14]
		self.cell2.pos = [14, 8]
		self.cell3.pos = [0, -14]
		self.cell4.pos = [-14, 8]
		if self.stimuli_number == 6:
			self.cell5.pos = [-14, -8]
			self.cell6.pos = [14, -8]





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
		self.LSL.push_sample([999])
		core.wait(1)
		sys.exit()

	def run_SSVEP_exp(self):
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


	def run_P300_exp(self, stim_duration_FRAMES = 3, ISI_FRAMES = 9, repetitions =  10, waitforS=True):
		'P300 expreiment. Stimuli duration and interstimuli interval should be supplied as number of frames.'
		cycle_ms = (stim_duration_FRAMES +ISI_FRAMES)*1000.0/self.refresh_rate
		print 'P300 cycle is %.2f ms' % cycle_ms
		seq = [1]*stim_duration_FRAMES + [0]*ISI_FRAMES

		# self.cells = [self.cell1,self.cell2,self.cell3,self.cell4]
		self.cells = [self.cell1,self.cell2,self.cell3,self.cell4, self.cell5, self.cell6][0:self.stimuli_number]

		p300_markers_on =  [[11], [22],[33],[44], [55], [66]]
		if self.LEARN == True:
			aims = [int(a)-1 for a in np.genfromtxt('aims_learn.txt')]
			print aims
		elif self.LEARN == False:
			aims = [int(a) -1 for a in np.genfromtxt('aims_play.txt')]
			print aims

		if waitforS == True:
			while 's' not in event.getKeys(): # wait for S key to start
				pass
		for letter, aim in enumerate(aims):
			self.LSL.push_sample([777]) # input of new letter
			superseq = generate_p300_superseq(numbers = range(self.stimuli_number), repetitions = repetitions)

			# aim_stimuli
			print [self.cells[aim].name] 
			self.cells[aim].fillColor = self.stimcolor_p300[1][self.cells[aim].name] # indicate aim stimuli
			self.win.flip()
			core.wait(2)
			self.cells[aim].fillColor = self.stimcolor_p300[0][self.cells[aim].name] # fade to grey
			self.win.flip()
			core.wait(1)

			deltalist = ['','','','','','','','','','']
			if 'escape' in event.getKeys():
				self.exit_()

			self.win.flip() # sycnhronize time.time() with first flip() => otherwise first interval seems longer.
			tt = time.time()
			for a in superseq:
				self.cells[a].fillColor = self.stimcolor_p300[1][self.cells[a].name]
				self.win.flip()
				self.LSL.push_sample(p300_markers_on[a]) # push marker immdiately after first bit of the sequence
				## tried to sync _data and .easy files.
				# dtm =  datetime.datetime.now()
				# dtmint = int(time.mktime(dtm.timetuple())*1000 + dtm.microsecond/1000)
				# dtmint = dtmint - (dtmint/1000000)*1000000 +  p300_markers_on[a][0]*1000000
				# print int(time.mktime(dtm.timetuple())*1000 + dtm.microsecond/1000)
				# self.LSL.push_sample([dtmint]) # push marker immdiately after first bit of the sequence; first digit is the number, other  - tail of Unix Time


				for b in seq[1:]:
					self.cells[a].fillColor = self.stimcolor_p300[b][self.cells[a].name]
					self.win.flip()
				
				#acess timing accuracy
				deltaT = time.time() - tt
				deltaT = "{0:2.0f}".format(round((deltaT*1000)- cycle_ms,2))
				# print deltaT
				deltalist[1:] = deltalist[0:-1]
				deltalist[0] = deltaT
				print 'delta T:%s ms \r' % str(deltalist),
				tt = time.time()

			
			core.wait(1.5) # wait one second after last blink
			self.LSL.push_sample([888]) # end of the trial
			core.wait(0.5)
			if self.LEARN == False:
				while 'answer' not in self.conn.recv(1024):
					pass	
				print 'next letter'
		
		if self.LEARN == True:
			core.wait(1)
			self.LSL.push_sample([888999]) # end of learningsession
			# start online session
			print stim_duration_FRAMES
			self.LEARN = False

			# wait while classifier finishes learning
			while self.conn.recv(2048) != 'startonlinesession':
				pass
			print  'learning session finished, press s to continue'
			while 's' not in event.getKeys(): # wait for S key to start
				pass
			print 'Online session started'

			self.run_P300_exp(stim_duration_FRAMES = stim_duration_FRAMES,
							  ISI_FRAMES = ISI_FRAMES, repetitions =  repetitions,
							  waitforS= waitforS)
		else:
			self.exit_()

def generate_p300_superseq(numbers =[0,1,2,3], repetitions = 10):
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
	
	os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

	ENV = ENVIRONMENT(DEMO = True)
	# ENV.Fullscreen = True
	ENV.refresh_rate = 60
	ENV.build_gui(monitor = mymon, rgb = ENV.rgb)
	ENV.run_P300_exp()