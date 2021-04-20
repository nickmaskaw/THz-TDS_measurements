# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:48:28 2021

@author: nicolas kawahala
"""

import pyvisa as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fft import fft, fftfreq
import time as tm
import os

class Constants:
    C     = 299_792_458*1e3/1e12  # mm/ps (~0.3mm/ps)
    n_AIR = 1.000_293             # Refractive index of air
       
    
class Convert:
    def ps_to_mm(t):
        return t*Constants.C

    def mm_to_ps(d):
        return d/Constants.C
    
    def v_to_I(v, sensitivity):
        v_fullscale = 10  # V
        return v * (sensitivity / v_fullscale)
    
    def I_to_v(I, sensitivity):
        v_fullscale = 10  # V
        return I * (v_fullscale / sensitivity)
    
    
class Multimeter:
    def __init__(self, VISA_ADDRESS='USB0::0x0957::0x0607::MY47027685::INSTR'):
        self.VISA_ADDRESS = VISA_ADDRESS
        self.instr = None
        self.instr_idn = 'no instrument'
        
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
    
    
class Identity:
    def __init__(self, start, end, sens, tcons, vel,
                 vbias, freq, hum, time_stamp=None):
        self.time_stamp = time_stamp
        
        self.start = start  # mm
        self.end   = end    # mm
        self.sens  = sens   # nA
        self.tcons = tcons  # ms
        self.vel   = vel    # mm/s
        self.vbias = vbias  # V
        self.freq  = freq   # Hz
        self.hum   = hum    # %
        
    def update_time_stamp(self):
        self.time_stamp = tm.strftime('%Y%m%d-%H%M%S')
        
    def frame(self):
        try:
            tist = tm.strptime(self.time_stamp, '%Y%m%d-%H%M%S')
            date = tm.strftime('%d/%m/%Y', tist)
            time = tm.strftime('%H:%M:%S', tist)
        except:
            date = None
            time = None
        
        cols = ('date', 'time', 'start', 'end', 'sens',
                'tcons', 'vel', 'vbias', 'freq', 'hum')
        vals = (date, time, self.start, self.end, self.sens,
                self.tcons, self.vel, self.vbias, self.freq, self.hum)
        
        df = pd.DataFrame(dict(zip(cols, vals)), index=[0])
        
        return df
            
    def __str__(self):
        if not self.time_stamp: self.update_time_stamp()
        
        f = '{}__{}to{}mm__{}nA__{}ms__{}mmps__{}V__{}Hz__{}%__.txt'
        args = ('time_stamp', 'start', 'end', 'sens', 
                'tcons', 'vel', 'vbias', 'freq', 'hum')
        
        return f.format(*(self.__dict__[n] for n in args))
    
    def __repr__(self):
        return str(self.__dict__)
    
    @classmethod
    def decode(cls, file_name):
        p = file_name.split('__')
        time_stamp = p[0]
        
        start = p[1].split('to')[0]
        end   = p[1].split('to')[1][:-2]
        sens  = p[2].split('nA')[0]
        tcons = p[3].split('ms')[0]
        vel   = p[4].split('mmps')[0]
        vbias = p[5].split('V')[0]
        freq  = p[6].split('Hz')[0]
        hum   = p[7].split('%')[0]
        
        idn = Identity(*cls.int_or_float(start, end, sens, tcons, vel,
                                         vbias, freq, hum), time_stamp)
        return idn
        
    @classmethod
    def int_or_float(cls, *args):
        out = []
        for s in args:
            try:
                out.append(float(s)) if ('.' in s) else out.append(int(s))
            except:
                out.append(s)
        
        return out
        
        
class Measurement:
    folder = './output' 
    
    def __init__(self, idn, Imin=-1.1, Imax=2.1, dt=0.1):
        if not os.path.exists(Measurement.folder): os.makedirs(Measurement.folder)
        
        # Identity object with the measurement info
        self.idn = idn
        
        # Use the Imin, Imax estimatives (in nA) to scale the plotting area:
        self.ymin = Convert.I_to_v(Imin, idn.sens)
        self.ymax = Convert.I_to_v(Imax, idn.sens)
        
        # Compute the measurement's total time, in seconds:
        self.T    = (idn.start - idn.end) / idn.vel
        
        # Given the dt between measurements (in s), compute the number of points:
        self.dt   = dt
        self.N    = int(self.T / self.dt)
        
    def create_plot(self):
        plt.close('all')
        self.fig   = plt.figure('delay', figsize=[9, 7])
        self.ax    = self.fig.add_subplot(111)
        self.line, = self.ax.plot(np.nan, np.nan)
        plt.show(block=False)
        self.fig.canvas.draw()
        self.ax.set_xlim([0, self.T])
        self.ax.set_ylim([self.ymin, self.ymax])
        
    def update_plot(self, x_data, y_data):
        self.line.set_xdata(x_data)
        self.line.set_ydata(y_data)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        
    def final_plot(self, x_data, y_data):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(x_data, y_data)
        
    def save_data(self, d, I):
        self.idn.update_time_stamp()
        data = pd.DataFrame({'t': Convert.mm_to_ps(2*d), 'I': I, 'd': d})
        data.to_csv(f'{Measurement.folder}/{self.idn}', sep='\t', index=False)
        
    def start(self, multimeter):
        #if not multimeter: sample = pd.read_csv('sample.txt', sep='\t', decimal=',')
        
        t0 = tm.time()
        
        # Arrays to store the data:
        v = np.full(self.N, np.nan)     # Multimeter reads
        t = np.full(self.N, np.nan)     # Instant of measurement
        
        for i in range(self.N):
            tm.sleep(self.dt - (tm.time()-t0)%self.dt)
            
            t[i] = tm.time() - t0
            v[i] = multimeter.get_meas()  # sample.v[i] if not multimeter else multimeter.get_meas()
            self.update_plot(t, v)

            if t[i] > self.T: break
        
        d = self.idn.vel * t  # Delay line displacements
        I = Convert.v_to_I(v, self.idn.sens)
        
        self.final_plot(d, I)
        self.save_data(d, I)
        
    def get_filename(self):
        return str(self.idn)


class Data:
    folder = './output'
    
    def __init__(self, file):
        self.file = file
        self.time = self.read_time_domain_data(file)
        self.freq = self.compute_fft()
        
    def __repr__(self):
        return f'Data from file {self.file}'
        
    def read_time_domain_data(self, file):
        return pd.read_table(f'{Data.folder}/{file}', usecols=['t', 'I'])
    
    def compute_fft(self):
        dt = 0.01
        t  = self.time.t
        I  = self.time.I
        
        ti = np.arange(np.min(t), np.max(t), dt)
        Ii = np.interp(ti, t, I)
        
        N = len(ti)
        
        It = fft(Ii)[:N//2]
        amp = (2/N) * np.abs(It)
        phs = np.angle(It)
        frq = fftfreq(N, dt)[:N//2]
        
        return pd.DataFrame({'frq': frq, 'amp': amp, 'phs': phs, 'fft': It})
    
    def plot(self):
        fig = plt.figure()
        
        axt = fig.add_subplot(211)
        axt.plot(self.time.t, self.time.I)
        axt.set_xlabel('time (ps)')
        axt.set_ylabel('Photocurrent (nA)')
        
        axf = fig.add_subplot(212)
        axf.plot(self.freq.frq, self.freq.amp)
        axf.set_xlabel('frequency (THz)')
        axf.set_ylabel('Amplitude (a. u.)')
        axf.set_xlim([-0.2, 5.2])
        axf.set_ylim([1e-5, 1])
        axf.set_yscale('log')
        
    @classmethod
    def data_list(cls, file_list, *indices):
        data_list_ = list(Data(file_list.get_file(i)) for i in indices)
        return data_list_
    
    
class Plot:
    folder = './saved_figures'
    
    @classmethod
    def set_folder(cls, folder):
        cls.folder = folder
    
    @classmethod
    def save_fig(cls, file_name, dpi_val=300):
        if not os.path.exists(Plot.folder): os.makedirs(Plot.folder)
        plt.savefig(f'{Plot.folder}/{file_name}', dpi=dpi_val)
        
    @classmethod
    def get_label(cls, data, label):
        idn = Identity.decode(data.file)
        idn_frame = idn.frame()        
        return f'{idn_frame[label][0]}' if label in idn_frame else None
    
    @classmethod
    def new_figure(cls, nrows=1, ncolumns=1, title=None, fig_size=[5, 5],
                   font_size=16):
        with plt.rc_context({'font.size': font_size}):
            fig, ax = plt.subplots(nrows, ncolumns, figsize=fig_size)
        fig.suptitle(title)
        return fig, ax
        
    @classmethod
    def time_domain(cls, ax, *data_list, label=None, leg_title=None):
        colors = cm.rainbow(np.linspace(0, 1, len(data_list)))
        
        for data, c in zip(data_list, colors):
            l = cls.get_label(data, label)
            ax.plot(data.time.t, data.time.I, color=c, label=l)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Photocurrent (nA)')
        plt.tight_layout()

            
    @classmethod
    def freq_domain(cls, ax, *data_list, label=None, leg_title=None):
        colors = cm.rainbow(np.linspace(0, 1, len(data_list)))
            
        for data, c in zip(data_list, colors):
            l = cls.get_label(data, label)
            ax.plot(data.freq.frq, data.freq.amp, color=c, label=l)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_yscale('log')
        ax.set_xlim([-0.2, 5.2])
        ax.set_ylim([1e-5, 1])
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Amplitude (a. u.)')
        plt.tight_layout()
            
            
class FileList:
    def __init__(self, folder):
        self.folder = folder
        self.time   = tm.strftime('%d/%m/%Y @ %H:%M:%S')
        self.table  = self.frame(folder, *os.listdir(folder))
        
    def __repr__(self):
        return f'File list of "{self.folder}" generated on {self.time}'
    
    def get_file(self, index):
        return self.table.file[index]
    
    def get_table(self, date=None, hide_file=True):
        df = self.table if not hide_file else self.table.drop('file', axis=1)
        return df if not date else df[df['date'] == date]
    
    @classmethod
    def frame(cls, folder, *files):
        cols = ['date', 'time', 'start', 'end', 'sens', 
                'tcons', 'vel', 'vbias', 'freq', 'hum', 'file']
        df = pd.DataFrame(columns=cols)
        
        for file in files:
            idn = Identity.decode(file)
            idn_frame = idn.frame()
            idn_frame['file'] = file
            
            df = df.append(idn_frame, ignore_index=True)
        
        return df            
            