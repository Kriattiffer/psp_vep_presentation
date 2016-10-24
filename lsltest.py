from pylsl import StreamInfo, StreamOutlet
import time

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' Create outlet for Enobio. Requires name of the stream (same as in NIC) and MAC adress of the device. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])'''
	info = StreamInfo(name,'sdfsdf','Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet

LSL = create_lsl_outlet() # create outlet for sync with NIC

raw_input()
LSL.push_sample([999])
time.sleep(4)
print 'done'
