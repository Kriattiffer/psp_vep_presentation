from iViewXAPI import  *            			#iViewX library
from numpy import *                   			#many different maths functions
from numpy.random import *       				#maths randomisation functions
import os                                   	#handy system and path functions
from win32api import GetSystemMetrics
import PIL
from ctypes import *

host_ip = '192.168.0.2'
server_ip = '192.168.0.3'
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)

# ---------------------------------------------
#---- connect to iView
# ---------------------------------------------

# res = iViewXAPI.iV_SetLogger(c_int(1), c_char_p("iViewXSDK_Python_GazeContingent_Demo.txt"))
res = iViewXAPI.iV_Connect(c_char_p(host_ip), c_int(4444), c_char_p(server_ip), c_int(5555))
res = iViewXAPI.iV_GetSystemInfo(byref(systemData))

# ---------------------------------------------
#---- configure and start calibration
# ---------------------------------------------
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream

for stream in resolve_stream():
    print 'stream', stream.name() 

displayDevice = 0
calibrationData = CCalibration(9, 1, displayDevice, 0, 1, 20, 239, 1, 10, b"")

res = iViewXAPI.iV_SetupCalibration(byref(calibrationData))
print "iV_SetupCalibration " + str(res)
res = iViewXAPI.iV_Calibrate()
print "iV_Calibrate " + str(res)
res = iViewXAPI.iV_Validate()
print "iV_Validate " + str(res)

# raw_input()

class CEye(Structure):
    _fields_ = [("gazeX", c_double),
    ("gazeY", c_double),
    ("diam", c_double),
    ("eyePositionX", c_double),
    ("eyePositionY", c_double),
    ("eyePositionZ", c_double)]

class CSample(Structure):
    _fields_ = [("timestamp", c_longlong),
    ("leftEye", CEye),
    ("rightEye", CEye),
    ("planeNumber", c_int)]

leftEye = CEye(0,0,0)
rightEye = CEye(0,0,0)
sampleData = CSample(0,leftEye,rightEye,0)
for a in range(10000):
    css = iViewXAPI.iV_GetSample(byref(sampleData))
    print sampleData.planeNumber, sampleData.rightEye.gazeX

    # print css
iViewXAPI.iV_Disconnect()
# raw_input()