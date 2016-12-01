# -*- coding: utf-8 -*- 

import os, sys, time, socket, random, datetime, socket, ast, math
from psychopy import visual, core, event, monitors, logging
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from psychopy.tools.monitorunittools import posToPix

mymon = monitors.Monitor('Eizo', distance=48, width = 52.5)
# mymon = monitors.Monitor('zenbook', distance=18, width = 29.5)
mymon.setSizePix([1920, 1080])		
# mymon.setSizePix([1024, 768])		


class ENVIRONMENT():
	""" class for visual stimulation during the experiment """
	def __init__(self, DEMO = False, config = './circles.bcicfg'):

		self.background = '#868686'

		self.Fullscreen = False
		self.window_size = (1920, 1080)
		self.LEARN = True
		self.number_of_inputs = 12
	

		try:
			self.config =  ast.literal_eval(open(config).read())
		except Exception, e:
			print 'Problem with config file:'
			print e
			self.exit_()

		self.stimcolor = ['white', 'white', 'white', 'white'] #steadystate
		self. time_4_one_letter = 6 #steadystate

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
				


	def build_gui(self, stimuli_number = 6, 
					monitor = mymon, fix_size = 1, screen  = 1):
		''' function for creating visual enviroment. Input: various parameters of stimuli, all optional'''
		
		self.stimuli_indices = range(stimuli_number)
		
		active_stims = []
		non_active_stims = []
		# Create window
		self.win = visual.Window(fullscr = self.Fullscreen, 
							rgb = self.background,
							size = self.window_size,	
							monitor = monitor,
							# waitBlanking = True,
							# useFBO=True,
							screen = screen # 1- right, 0 - left
							)
		self.win.setRecordFrameIntervals(True)

		self.refresh_rate =  math.ceil(self.win.monitorFramePeriod**-1)
		self.frame_time = self.win.monitorFramePeriod*1000

		# read image from rescources dir and crate ImageStim objects
		stimpics = os.listdir(self.config['stimuli_dir'])
		for pic in stimpics:
			name = int(pic.split('_')[1])
			pic = os.path.join(self.config['stimuli_dir'], pic)
			if name in self.stimuli_indices:
				stim = visual.ImageStim(self.win, image=pic,
										 name = name,
										size = 4, units = 'deg')
				if 'non_active' not in pic:
					active_stims.append(stim)
				else:
					non_active_stims.append(stim)
			else:
				pass
		active_stims  = active_stims
		non_active_stims  = non_active_stims

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
		poslist = self.config['positions']
		for a in active_stims:
			a.pos = poslist[int(a.name)]

		for a in non_active_stims:
			a.pos = poslist[int(a.name)]
			a.autoDraw = True
			
		self.stimlist = [non_active_stims, active_stims]

		self.p300_markers_on =  [[11], [22],[33],[44], [55], [66]]

	def sendTrigger(self, stim):
		'''This function is called with callOnFlip which
		 "call the function just after the flip, before psychopy does a bit of housecleaning. ", 
		 according to some dude on the internets'''
		self.LSL.push_sample([self.stimlist[1][stim].name], pushthrough = True) # push marker immdiately after first bit of the sequence

	def run_P300_exp(self, stim_duration_FRAMES = 3, ISI_FRAMES = 9, repetitions =  10, waitforS=True, stimuli_number = 6):
		'P300 expreiment. Stimuli duration and interstimuli interval should be supplied as number of frames.'
		cycle_ms = (stim_duration_FRAMES + ISI_FRAMES)*1000.0/self.refresh_rate
		print 'P300 cycle is %.2f ms' % cycle_ms
		seq = [1]*stim_duration_FRAMES + [0]*ISI_FRAMES


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
			superseq = generate_p300_superseq(numbers = self.stimuli_indices, repetitions = repetitions)

			# aim_stimuli
			self.stimlist[1][aim].autoDraw = True # indicate aim stimuli
			self.stimlist[0][aim].autoDraw = False 
			self.win.flip()
			
			core.wait(2)
			
			self.stimlist[0][aim].autoDraw = True  # fade back
			self.stimlist[1][aim].autoDraw = False 
			self.win.flip()
			core.wait(1)

			if 'escape' in event.getKeys():
				self.exit_()

			self.win.flip() # just in case

			for a in superseq:
				# first bit of sequence and marker
				self.win.callOnFlip(self.sendTrigger, stim = a)
				self.stimlist[1][a].autoDraw = True
				self.stimlist[0][a].autoDraw = False
				self.win.flip()

				# other bits of sequence
				for b in seq[1:]:
					self.stimlist[b][a].autoDraw = True
					self.stimlist[b==0][a].autoDraw = False
					self.win.flip()

			
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
			self.LSL.push_sample([999888])
			self.run_P300_exp(stim_duration_FRAMES = stim_duration_FRAMES,
							  ISI_FRAMES = ISI_FRAMES, repetitions =  repetitions,
							  waitforS= waitforS)
		else:
			self.exit_()

	def exit_(self):
		''' exit and kill dependent processes'''
		self.LSL.push_sample([999])
		core.wait(1)

		# from matplotlib import pyplot as plt
		# plt.plot(self.win.frameIntervals[2:-3], 'o')
		# plt.show()

		sys.exit()

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

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' Create outlet for Enobio. Requires name of the stream (same as in NIC) and maybe MAC adress of the device. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])'''
	info = StreamInfo(name,'Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet

def view():
	'''function for multiprocessing'''
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)

if __name__ == '__main__':
	print 'done imports'
	os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

	ENV = ENVIRONMENT(DEMO = True, config = './circles.bcicfg')
	# ENV.Fullscreen = True
	# ENV.photocell = True
	ENV.refresh_rate = 60
	ENV.build_gui(monitor = mymon, screen = 0)
	ENV.run_P300_exp(waitforS = False)