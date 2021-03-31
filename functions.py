# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:48:28 2021

@author: nicolas kawahala
"""

import pyvisa as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

class Constants:
    C = 299_792_458*1e3/1e12 #mm/ps (~0.3mm/ps)
    
    
class Convert:
    def ps_to_mm(t):
        return t*Constants.C

    def mm_to_ps(d):
        return d/Constants.C
    
    
class Identity:
    def __init__(self, start, end, sens, tcons, vel, vbias, freq, hum):
        self.start = start  # mm
        self.end   = end    # mm
        self.sens  = sens   # nA
        self.tcons = tcons  # ms
        self.vel   = vel    # mm/s
        self.vbias = vbias  # V
        self.freq  = freq   # Hz
        self.hum   = hum    # %
            
    def __str__(self):
        self.time_stamp = tm.strftime('%Y%m%d-%H%M%S')
        f = '{}__{}to{}mm__{}nA__{}ms__{}mmps__{}V__{}Hz__{}%__.txt'
        args = ('time_stamp', 'start', 'end', 'sens', 
                'tcons', 'vel', 'vbias', 'freq', 'hum')
        
        return f.format(*(self.__dict__[n] for n in args))
    

class Multimeter:
    def __init__(self, VISA_ADDRESS='USB0::0x0957::0x0607::MY47027685::INSTR'):
        self.VISA_ADDRESS = VISA_ADDRESS
        self.instr = None
        self.instr_idn = 'no instrument'
        
    def __str__(self):
        return f'This instrument: {self.instr_idn}'
            
    def open(self):
        rm = pv.ResourceManager()
        
        try:
            self.instr = rm.open_resource(self.VISA_ADDRESS)
            self.instr_idn = self.instr.query('*IDN?')
        except:
            print('ERROR. Check if the intended VISA address is listed below:')
            print(rm.list_resources())
            
    def close(self):
        self.instr.close()
        
        
class Measurement:
    def __init__(self, idn, Imin=-1.1, Imax=2.1, dt=0.1):
        self.idn = idn # A Identity object instance with the measurement info
        
        # Use the Imin, Imax estimatives (in nA) to scale the plotting area:
        self.ymin = Imin*(10/self.idn.sens) # 10V equals 1 sensivity full scale
        self.ymax = Imax*(10/self.idn.sens) # 10V equals 1 sensivity full scale
        
        # Compute the measurement's total time, in seconds:
        self.T    = (self.idn.start - self.idn.end)/self.idn.vel
        
        # Given the dt between measurements (in s), compute the number of points:
        self.dt   = dt
        self.N    = int(self.T/self.dt)

    def create_plot(self):
        # Arrays to store the data:
        self.v = np.full(self.N, np.nan)     # Multimeter reads
        self.t = np.full(self.N, np.nan)     # Instant of measurement
        
        # Init plot
        plt.close('all')
        self.fig   = plt.figure('delay', figsize=[9, 7])
        self.ax    = self.fig.add_subplot(111)
        self.line, = self.ax.plot(self.t, self.v)
        plt.show(block=False)
        self.fig.canvas.draw()
        self.ax.set_xlim([0, self.T])
        self.ax.set_ylim([self.ymin, self.ymax])
        
    def start(self):
        t0 = tm.time()
        
        for i in range(self.N):
            tm.sleep(self.dt - (tm.time()-t0)%self.dt)
            
            self.t[i] = tm.time() - t0
            self.v[i] = 0
            
            self.line.set_xdata(self.t)
            self.line.set_ydata(self.v)
            self.ax.draw_artist(self.ax.patch)
            self.ax.draw_artist(self.line)
            self.fig.canvas.update()
            self.fig.canvas.flush_events()
            
            if self.t[i] > self.T: break
