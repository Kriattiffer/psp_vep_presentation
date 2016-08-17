from psychopy import visual, core, event, monitors
import os, sys, time
import numpy

# mymon = monitors.Monitor('Eizo', distance=50, width = 52.5)
mymon = monitors.Monitor('zenbook', distance=25, width = 29.5)
mymon.setSizePix([1920, 1080])	

# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state 60 fps
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps
base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,2,2,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0] # cvep 60 fps with red bits
# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state
# base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0] # cvep
# base_mseq = [a for a in base_mseq for b in [0,0]] # extend sequence
base_mseq = numpy.array(base_mseq, dtype = int)


class ENVIRONMENT():
	""" class for visual stimulation during the experiment """

	def build_gui(self, monitor = mymon, size=(1920, 1080),
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
		self.win = visual.Window(fullscr = True, 
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
		tt = time.time()

		seq1, seq2, seq3, seq4 = base_mseq, numpy.roll(base_mseq, 8), numpy.roll(base_mseq, 16), numpy.roll(base_mseq, 24) # create sequences for every cell; should be identical size!
		pattern = [(self.cell1, seq1), (self.cell2, seq2), (self.cell3, seq3), (self.cell4, seq4)]
		while  1:
			if 'escape' in event.getKeys():
				sys.exit()
				
			for bit_number in range(len(seq4)): # cycle through sequences 
				for pair in pattern: # decide what to do with every circle at the next refresh
					self.flipper(pair[0], pair[1][bit_number], bit_number)
				
				self.win.flip() # refresh screen
			
			print (tt - time.time())
			tt = time.time()


if __name__ == '__main__':
	
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)