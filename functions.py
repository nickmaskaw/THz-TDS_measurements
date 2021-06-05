# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:48:28 2021

@author: nicolas kawahala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fft import fft, fftfreq
import time as tm
import os


class Constants:
    c     = 299_792_458*1e3/1e12  # mm/ps (~0.3mm/ps)
    n_AIR = 1.000_293             # Refractive index of air


class Convert:
    def ps_to_mm(t):
        return t*Constants.c

    def mm_to_ps(d):
        return d/Constants.c
    
    def v_to_I(v, sensitivity):
        v_fullscale = 10  # V
        return v * (sensitivity / v_fullscale)
    
    def I_to_v(I, sensitivity):
        v_fullscale = 10  # V
        return I * (v_fullscale / sensitivity)


class Identity:
    def __init__(self, start, end, sens, tcons, vel,
                 vbias, freq, hum, obs='', time_stamp=None):
        self.time_stamp = time_stamp
        
        self.start = start  # mm
        self.end   = end    # mm
        self.sens  = sens   # nA
        self.tcons = tcons  # ms
        self.vel   = vel    # mm/s
        self.vbias = vbias  # V
        self.freq  = freq   # Hz
        self.hum   = hum    # %
        self.obs   = obs
        
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
                'tcons', 'vel', 'vbias', 'freq', 'hum', 'obs')
        vals = (date, time, self.start, self.end, self.sens, self.tcons,
                self.vel, self.vbias, self.freq, self.hum, self.obs)
        
        df = pd.DataFrame(dict(zip(cols, vals)), index=[0])
        
        return df
            
    def __str__(self):
        if not self.time_stamp: self.update_time_stamp()
        
        f = '{}__{}to{}mm__{}nA__{}ms__{}mmps__{}V__{}Hz__{}%__{}.txt'
        args = ('time_stamp', 'start', 'end', 'sens', 
                'tcons', 'vel', 'vbias', 'freq', 'hum', 'obs')
        
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
        obs   = p[8].split('.txt')[0]
        
        idn = Identity(*cls.int_or_float(start, end, sens, tcons, vel, vbias,
                                         freq, hum), obs, time_stamp)
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
    
    def __init__(self, idn, Imin=-6, Imax=6.5, dt=0.1, step=None):
        if not os.path.exists(Measurement.folder): os.makedirs(Measurement.folder)
        
        # Identity object with the measurement info
        self.idn = idn
        
        # Use the Imin, Imax estimatives (in nA) to scale the plotting area:
        self.ymin = Convert.I_to_v(Imin, idn.sens)
        self.ymax = Convert.I_to_v(Imax, idn.sens)
        
        # Given the dt between measurements (in s), compute the number of points:
        self.dt   = dt
        
    def create_plot(self):
        plt.close('all')
        self.fig   = plt.figure('delay', figsize=[9, 7])
        self.ax    = self.fig.add_subplot(111)
        self.line, = self.ax.plot(np.nan, np.nan)
        plt.show(block=False)
        self.fig.canvas.draw()
        self.ax.set_xlim([self.idn.start, self.idn.end])
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
        self.ax.set_xlim([self.idn.start, self.idn.end])
        
    def save_data(self, d, I):
        self.idn.update_time_stamp()
        data = pd.DataFrame({'t': Convert.mm_to_ps(2*d), 'I': I, 'd': d})
        data.to_csv(f'{Measurement.folder}/{self.idn}', sep='\t', index=False)
        
    def start(self, multimeter, delayline, step=None):
        self.create_plot()
        
        start = self.idn.start
        end   = self.idn.end
        vel   = self.idn.vel
        tcons = self.idn.tcons * 1e-3  # integration time in seconds
        sens  = self.idn.sens
        dt    = self.dt
        
        delayline.start_polling(10)
        delayline.set_vel(vel)
        
        t0 = tm.time()
        if not step:
            T = (start - end) / vel
            N = int(T / dt)
            
            v = np.full(N, np.nan)     # Multimeter reads
            d = np.full(N, np.nan)     # Delayline position
        
            delayline.move_to(end, timeout=0)
            for i in range(N):
                tm.sleep(dt - (tm.time()-t0)%dt)
            
                d[i] = delayline.get_pos()
                v[i] = multimeter.get_meas()
                self.update_plot(d, v)
                
                if d[i] <= end: break
        
        else:  # if step:            
            N   = int(abs(start - end) / step)
            pos = np.arange(start, end, -step)
            
            v = np.full(N, np.nan)     # Multimeter reads
            d = np.full(N, np.nan)     # Delayline position
            
            for i in range(N):
                delayline.move_to(pos[i])
                tm.sleep(3*tcons)
                
                d[i] = delayline.get_pos()
                v[i] = multimeter.get_meas()
                self.update_plot(d, v)
                
        delayline.stop_polling()
            
        I = Convert.v_to_I(v, sens)
        
        self.final_plot(d, I)
        self.save_data(start-d, I)
        
    def get_filename(self):
        return str(self.idn)


class Data:
    folder = './output'
    
    def __init__(self, file, fft_dt=0.01):
        self._file = file
        self._idn  = Identity.decode(file)
        self._time = self._read_time_domain_data(file)
        self._freq = self.compute_fft(fft_dt)
        
    @property
    def file(self): return self._file
    @property
    def idn(self): return self._idn    
    @property
    def time(self): return self._time
    @property
    def freq(self): return self._freq
        
    def __repr__(self):
        return f'Data from file {self.file}'
        
    def _read_time_domain_data(self, file):
        return pd.read_table(f'{Data.folder}/{file}', usecols=['t', 'I'])
    
    def compute_fft(self, dt):
        t = self.time.t
        I = self.time.I
        
        ti = np.arange(np.min(t), np.max(t), dt)
        Ii = np.interp(ti, t, I)
        
        N = len(ti)
        
        It = fft(Ii)[:N//2]
        amp = (2/N) * np.abs(It)
        phs = np.angle(It)
        frq = fftfreq(N, dt)[:N//2]
        
        return pd.DataFrame({'frq': frq, 'amp': amp, 'phs': phs, 'fft': It})
    
    def plot(self):
        fig, ax = Plot.new_figure(nrows=1, ncolumns=2, fig_size=[9, 5], font_size=12)
        Plot.time_domain(ax[0], self)
        Plot.freq_domain(ax[1], self)

    
    @classmethod
    def data_list(cls, file_list, *indices):
        data_list_ = []
        for i in indices:
            if isinstance(i, int):
                data_list_.append(Data(file_list.get_file(i)))
            elif isinstance(i, list) and len(i)==2:
                i_list = list(range(i[0], i[-1]+1))
                for j in i_list:
                    data_list_.append(Data(file_list.get_file(j)))
            else:
                print(f'Warning: {i} is not an integer, nor a list of the type [imin, imax]')
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
    def get_label(cls, data, label, index):
        if isinstance(label, list):
            return f'{label[index]}'
        else:
            idn_frame = data.idn.frame()        
            return f'{idn_frame[label][0]}' if label in idn_frame else None
    
    @classmethod
    def new_figure(cls, nrows=1, ncolumns=1, title=None, fig_size=[5, 5],
                   font_size=16):
        with plt.rc_context({'font.size': font_size}):
            fig, ax = plt.subplots(nrows, ncolumns, figsize=fig_size)
        fig.suptitle(title)
        return fig, ax
        
    @classmethod
    def time_domain(cls, ax, *data_list, label=None, leg_title=None, colormap='rainbow', delayline_zero=None):
        colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(data_list)))
        
        for i, data in enumerate(data_list):
            if delayline_zero:
                D = delayline_zero - data.idn.start
                time_shift = Convert.mm_to_ps(2*D)
            else:
                time_shift = 0
            
            l = cls.get_label(data, label, i)
            ax.plot(data.time.t + time_shift, data.time.I, color=colors[i], label=l)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Photocurrent (nA)')
        plt.tight_layout()

            
    @classmethod
    def freq_domain(cls, ax, *data_list, y='amp', yscale='log', ymin=1e-4,
                    ymax=1, label=None, leg_title=None, colormap='rainbow'):
        colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(data_list)))
            
        for i, data in enumerate(data_list):
            l = cls.get_label(data, label, i)
            ax.plot(data.freq.frq, np.abs(data.freq[y]), color=colors[i], label=l)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_yscale(yscale)
        ax.set_xlim([-0.2, 5.2])
        ax.set_ylim([ymin, ymax])
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
    
    def get_table(self, date=None, hide_file=True, max_rows=60):
        pd.set_option('display.max_rows', max_rows)
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