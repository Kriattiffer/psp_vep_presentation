# -*- coding: utf-8 -*- 

from iViewXAPI import  *            			#iViewX library
from ctypes import *
import time, sys
from pylsl import StreamInlet, resolve_stream



def create_stream(stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
        ''' Opens LSL stream for markers, If error, tries to reconnect several times'''
        if recursion_meter == 0:
            recursion_meter +=1
        elif 0<recursion_meter <max_recursion_depth:
            print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
            recursion_meter +=1
        else:
            print 'exiting'
            return None
            
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
                return None
        else:
            print 'Error: markers stream is not available\n'
            return create_stream(stream_name_markers,recursion_meter)
        return inlet_markers


class Eyetracker():
    """docstring for Eyetracker"""
    def __init__(self, debug = False):
        self.im  = create_stream()
        self.host_ip = '192.168.0.2'
        self.server_ip = '192.168.0.3'
        if not self.im:
            print debug
            if debug != True:
                self.exit_()

    def  calibrate(self):
        '''configure and start calibration'''

        numberofPoints = 9
        displayDevice = 0 # 0 - primary, 1- secondary
        pointBrightness = 250
        backgroundBrightnress = 50
        targetFile = b""
        calibrationSpeed = 0 # slow
        autoAccept  = 2 # 0 = auto, 1 = semi-auto, 2 = auto 
        targetShape = 1 # 0 = image, 1 = circle1, 2 = circle2, 3 = cross
        targetSize = 20
        WTF = 1 #do not touch -- preset?

        calibrationData = CCalibration(numberofPoints, WTF, displayDevice, 
                                        calibrationSpeed, autoAccept, pointBrightness,
                                        backgroundBrightnress, targetShape, targetSize, targetFile)

        self.res = iViewXAPI.iV_SetupCalibration(byref(calibrationData))
        print "iV_SetupCalibration " + str(self.res)
        self.res = iViewXAPI.iV_Calibrate()
        print   "iV_Calibrate " + str(self.res)

    
    def validate(self):
        self.res = iViewXAPI.iV_Validate()
        print "iV_Validate " + str(self.res)
    
    def connect_to_iView(self):
        self.res = iViewXAPI.iV_Connect(c_char_p(self.host_ip), c_int(4444), c_char_p(self.server_ip), c_int(5555))
        self.res = iViewXAPI.iV_GetSystemInfo(byref(systemData))
        print "iV_sysinfo " + str(self.res)

    def send_marker_to_iViewX(self, marker):
        res = iViewXAPI.iV_SendImageMessage(marker)
        if str(self.res) !='1':
            print "iV_SendImageMessage " + str(self.res)
    
    def mainloop(self):
        self.res = iViewXAPI.iV_StartRecording ()
        print "iV_record " + str(self.res)
        for a in range(10):
            marker, timestamp_mark = self.im.pull_sample()
            IDF_marker =  str([marker, timestamp_mark])
            self.send_marker_to_iViewX(IDF_marker)

    def main(self):
        self.connect_to_iView()
        self.calibrate()
        self.validate()
        self.mainloop()


    def exit_(self):
        self.res = iViewXAPI.iV_StopRecording()
        self.res = iViewXAPI.iV_SaveData('test_test', '2', '3', 1)
        print "iV_SaveData" + str(self.res)
        self.res = iViewXAPI.iV_Disconnect()
        sys.exit()



if __name__ == '__main__':
    
    RED = Eyetracker(debug = True)
    RED.main()

