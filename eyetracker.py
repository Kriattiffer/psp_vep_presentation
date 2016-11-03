from iViewXAPI import  *            			#iViewX library
from ctypes import *
import time
from pylsl import StreamInlet, resolve_stream

host_ip = '192.168.0.2'
server_ip = '192.168.0.3'


def create_stream(stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
        ''' Opens LSL stream for markers, If error, tries to reconnect several times'''
        if recursion_meter == 0:
            recursion_meter +=1
        elif 0<recursion_meter <max_recursion_depth:
            print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
            recursion_meter +=1
        else:
            print 'exiting'
            sys.exit()
            inlet_markers = []
            
        print ("Eyetracker connecting to markers stream...")
        # inlet for markers
        if stream_name_markers in [stream.name() for stream in resolve_stream()]:
            sterams_markers = resolve_stream('name', stream_name_markers)
            inlet_markers = StreamInlet(sterams_markers[0])   
            try:
                inlet_markers
                print '...done \n'
            except NameError:
                print ("Error: Eyetracker cannot conect to markers stream\n")
                sys.exit()
        else:
            print 'Error: markers stream is not available\n'
            return create_stream(stream_name_markers,recursion_meter)
        return inlet_markers

def send_marker_to_iViewX(marker):
    res = iViewXAPI.iV_SendImageMessage(marker)
    if str(res) !='1':
        print "iV_SendImageMessage " + str(res)

def main():
    im = create_stream()
    for a in range(10):
        marker, timestamp_mark = im.pull_sample()
        IDF_marker =  str([marker, timestamp_mark])
        send_marker_to_iViewX(IDF_marker)

# ---------------------------------------------
#---- connect to iView
# ---------------------------------------------

# res = iViewXAPI.iV_SetLogger(c_int(1), c_char_p("iViewXSDK_Python_GazeContingent_Demo.txt"))
res = iViewXAPI.iV_Connect(c_char_p(host_ip), c_int(4444), c_char_p(server_ip), c_int(5555))
res = iViewXAPI.iV_GetSystemInfo(byref(systemData))

# ---------------------------------------------
#---- configure and start calibration
# ---------------------------------------------

displayDevice = 0
calibrationData = CCalibration(2, 1, displayDevice, 0, 1, 20, 239, 1, 10, b"")

res = iViewXAPI.iV_SetupCalibration(byref(calibrationData))
print "iV_SetupCalibration " + str(res)
res = iViewXAPI.iV_Calibrate()
print   "iV_Calibrate " + str(res)
# res = iViewXAPI.iV_Validate()
# print "iV_Validate " + str(res)
res = iViewXAPI.iV_StartRecording ()
print "iV_record " + str(res)

main()


res = iViewXAPI.iV_StopRecording()
res1 = iViewXAPI.iV_SaveData('test_test', '2', '3', 1)
print "iV_SaveData" + str(res1)
res = iViewXAPI.iV_Disconnect()
