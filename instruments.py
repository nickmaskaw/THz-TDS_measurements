# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:49:23 2021

@author: Nicolas Kawahala
"""

import pyvisa as pv
import time as tm
import sys
import clr
from System import String
from System import Decimal
from System.Collections import *

if not r'C:\Program Files\Thorlabs\Kinesis' in sys.path:
    sys.path.append(r'C:\Program Files\Thorlabs\Kinesis')
    
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI

clr.AddReference("Thorlabs.MotionControl.KCube.BrushlessMotorCLI")
from Thorlabs.MotionControl.KCube.BrushlessMotorCLI import KCubeBrushlessMotor



class Multimeter:
    def __init__(self, init_open=True, VISA_ADDRESS='USB0::0x0957::0x0607::MY47027685::INSTR'):
        self.VISA_ADDRESS = VISA_ADDRESS
        self.instr = None
        self.instr_idn = 'no instrument'
        
        if init_open: self.open_()
        
    def __repr__(self):
        return f'This instrument: {self.instr_idn}'
            
    def open_(self):
        rm = pv.ResourceManager()
        
        try:
            self.instr = rm.open_resource(self.VISA_ADDRESS)
            self.instr_idn = self.instr.query('*IDN?')
        except:
            print('ERROR. Check if the intended VISA address is listed below:')
            print(rm.list_resources())
            
    def close(self):
        self.instr.close()
        
    def get_meas(self):
        return self.instr.query('MEAS?')
    
    
    
class KBD101:
    def __init__(self, serial='28250877'):
        self._serial = serial
        self._device = self._build_device(serial)
        
        self.connect()
        tm.sleep(.2)
        self._motor_config = self._load_motor_config(serial)
        
    def __repr__(self):
        info = self.get_info()
        return 'Device: {} (serial no. {})  |  Stage: {} (serial no. {})'.format(*tuple(info.values()))
    
    def _build_device(self, serial):
        if isinstance(serial, str) and serial[:2] == '28':
            DeviceManagerCLI.BuildDeviceList()
            device_list = DeviceManagerCLI.GetDeviceList()
            
            if serial in device_list:
                device = KCubeBrushlessMotor.CreateKCubeBrushlessMotor(serial)
                return device
            else:
                print('Check in the intended serial number is listed below:')
                print(device_list)
                return None
        else:
            print('Check if the intended serial number is a string that begins with "28".')
            return None
        
    def _load_motor_config(self, serial):
        motor_config = self._device.LoadMotorConfiguration(serial)
        return motor_config
        
    def connect(self):
        self._device.Connect(self._serial)
        
    def disconnect(self):
        self._device.Disconnect()
        
    def get_info(self):
        device_info = self._device.GetDeviceInfo()
        stage_info  = self._device.GetStageDefinition()
        useful_info = {'deviceName'   : device_info.Name,
                       'deviceSerial' : device_info.SerialNumber,
                       'stageName'    : stage_info.PartNumber,
                       'stageSerial'  : stage_info.SerialNumber}
        return useful_info
    
    def enable(self):
        self._device.EnableDevice()
        tm.sleep(2)
        
    def disable(self):
        self._device.DisableDevice()
    
    def start_polling(self, rate=50):
        self._device.StartPolling(rate)
        
    def stop_polling(self):
        self._device.StopPolling()
        
    def get_polling_rate(self):
        return self._device.PollingDuration()
        
    def home(self, polling_rate=50, timeout=60000):
        poll = False if self.get_polling_rate() else True
        
        if poll: self.start_polling()
        self._device.Home(timeout)
        if poll: self.stop_polling()
            
    def move_to(self, pos, timeout=60000):
        self._device.MoveTo(Decimal(float(pos)), timeout)
        
    def return_to(self, pos, timeout=60000):
        self.set_vel(100)
        self.move_to(pos, timeout)
        
    def get_pos(self):
        pos = str(self._device.Position).replace(',', '.')
        return float(pos)
    
    def get_vel(self):
        vel = str(self._device.GetVelocityParams().MaxVelocity).replace(',', '.')
        return float(vel)
    
    def set_vel(self, vel, acceleration=999):
        self._device.SetVelocityParams(Decimal(vel), Decimal(acceleration))