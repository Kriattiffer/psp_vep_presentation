from psychopy import visual, core, event, monitors

import os, sys, time

# mymon = monitors.Monitor('Eizo', distance=50, width = 52.5)
mymon = monitors.Monitor('zenbook', distance=25, width = 29.5)
mymon.setSizePix([1920, 1080])	


stimrad = 2
stimcolor = 'red'
fix_size = 1
# base_mseq = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # steady-state
base_mseq = [0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0] # cvep
base_mseq = [a for a in base_mseq for b in [0,0]]

win = visual.Window(fullscr = True,
					rgb = '#868686',
					size=(1920, 1080),
					monitor = mymon
					)
cell1 = visual.Circle(win,
						radius=stimrad,
						lineColor = None, 
						fillColor = stimcolor,
						units = 'deg',
						autoDraw=True
						)
cell2 = visual.Circle(win,
						radius=stimrad,
						lineColor = None, 
						fillColor = stimcolor,
						units = 'deg',
						autoDraw=True
						)
cell3 = visual.Circle(win,
						radius=stimrad,
						lineColor = None, 
						fillColor = stimcolor,
						units = 'deg',
						autoDraw=True
						)
cell4 = visual.Circle(win,
						radius=stimrad,
						lineColor = None, 
						fillColor = stimcolor,
						units = 'deg',
						autoDraw=True
						)
fixation = visual.ShapeStim(win, 
						vertices=((0, -1*fix_size), (0, fix_size), (0,0), (-1*fix_size,0), (fix_size, 0)),
						units = 'deg',
						lineWidth=5,
						closeShape=False,
						lineColor='white',
						autoDraw=True
						)

cell1.pos = [0, 12]
cell2.pos = [12, 0]
cell3.pos = [0, -12]
cell4.pos = [-12, 0]



def flipper(n, frame, CELL):
	if 27<=n<=30:
		CELL.fillColor = 'red'
		return
	if frame ==1:
		CELL.fillColor = 'white'
		return
	elif frame == 0:
		CELL.fillColor = 'black'
		return

while  1:

	if 'escape' in event.getKeys():
		sys.exit()
	for n, frame in enumerate(base_mseq):
		flipper(n, frame, cell1)
		flipper(n, frame, cell2)
		flipper(n, frame, cell3)
		flipper(n, frame, cell4)
		
		win.flip()
