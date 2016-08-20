from pylsl import StreamInlet, resolve_stream
import numpy as np
import socket

def create_streams(stream_name_eeg = 'NIC', stream_name_markers = 'CycleStart'):
	''' Opens two LSL streams: one for EEG, another for markers'''
	
	print ("Connecting to NIC stream...")
	streams_eeg = resolve_stream('type', 'EEG')
	inlet_eeg = StreamInlet(streams_eeg[0])   
	try:
		inlet_eeg
		print '...done \n'
	except NameError:
		print ("Error: NIC stream not available\n")


	print ("Connecting to Psychopy stream...")
	sterams_markers = resolve_stream('name', stream_name_markers)	
	inlet_markers = StreamInlet(sterams_markers[0])   
	try:
		inlet_markers
		print '...done \n'

	except NameError:
		print ("Error: Psychopy stream not available\n")

	return inlet_eeg, inlet_markers

def create_array(top_exp_length = 60):
	"Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes"
	record_length = 500*60*top_exp_length*1.2
	array_size = (9, record_length)
	print 'Creating array with dimensions %s...' %str(array_size) 
	a = np.ones((record_length, 9), dtype = 'float')
	a[:] = np.NAN
	print '... done'
	return a

def fill_array(eeg_array, line_counter, data_chunk, timestamp_chunk, datatype = 'EEG'):
	length = len(timestamp_chunk)

	eeg_array[line_counter:line_counter+length, 0] = timestamp_chunk
	eeg_array[line_counter:line_counter+length,1:] = data_chunk
	# print eeg_array


ie, im =  create_streams()
EEG_ARRAY = create_array()

line_counter = 0
while line_counter<800:
    print line_counter
    marker, timestamp1 = im.pull_chunk()
    if timestamp1:
    	pass
    	# line_counter += len(timestamp2)
        # print('MARKER', timestamp1, marker)
	# EEG = ie.pull_chunk()    
    EEG, timestamp2 = ie.pull_chunk()
    if timestamp2:
    	fill_array(EEG_ARRAY, line_counter, EEG, timestamp2, datatype = 'EEG')
    	line_counter += len(timestamp2)

        # print('eeg', timestamp2, EEG)
np.savetxt('data.txt', EEG_ARRAY, fmt= '%.4f')
