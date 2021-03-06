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
from scipy.optimize import curve_fit as fit
import time as tm
import os


class Constants:
    c     = 299_792_458e3/1e12    # mm/ps (~0.3mm/ps)
    n_AIR = 1.000_293             # Refractive index of air
    hbar  = 6.582_119_569e-16     # eVs
    kB    = 8.617_333_262_145e-5  # eV/K


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
    
    def __init__(self, file, dt=0.033, time_range=(None, None), delayline_zero=None):        
        self._file       = file
        self._dt         = dt
        self._time_range = None
        self._idn        = Identity.decode(file)
        self._raw_data   = self._read_data_from_file(file)
        self._time       = self._fix_time_domain_data(dt, time_range, delayline_zero)
        self._freq       = self._compute_fft()
        
    @property
    def file(self): return self._file
    @property
    def dt(self): return self._dt
    @property
    def time_range(self): return self._time_range
    @property
    def idn(self): return self._idn   
    @property
    def raw_data(self): return self._raw_data 
    @property
    def time(self): return self._time
    @property
    def freq(self): return self._freq
        
    def __repr__(self):
        cut = f' | range={self.time_range}ps' if self.time_range else ''
        return f'Data from file {self.file}{cut}'
    
    def _read_data_from_file(self, file):
        data = pd.read_table(f'{Data.folder}/{file}').dropna()
        return data
        
    def _fix_time_domain_data(self, dt, time_range, delayline_zero):
        if delayline_zero:
            D = delayline_zero - self.idn.start
            time_shift = Convert.mm_to_ps(2*D)
        else:
            time_shift = 0
        
        raw_t = self.raw_data.t.values + time_shift
        raw_I = self.raw_data.I.values
        
        tmin, tmax = np.min(raw_t), np.max(raw_t)
        if isinstance(time_range, (list, tuple)) and len(time_range)==2:
            if time_range != (None, None):
                self._time_range = time_range
                tmin, tmax = time_range
        else:
            print(f'{time_range} is not a valid time_range of the type (tmin, tmax)')
            
        if tmin < raw_t[0]  : raw_I[0]   = 0
        if tmax > raw_t[-1]: raw_I[-1] = 0
            
        t = np.arange(tmin, tmax+dt, dt)
        I = np.interp(t, raw_t, raw_I)
        
        return pd.DataFrame({'t':t, 'I':I}).dropna()
    
    def _compute_fft(self):
        t  = self.time.t.values
        I  = self.time.I.values
        dt = self.dt        
        
        N = len(t)
        
        It  = np.conj(fft(I)[:N//2])  ### Changed to complex conjugate to fix analysis
        amp = (2/N) * np.abs(It)
        phs = np.angle(It)
        frq = fftfreq(N, dt)[:N//2]
        
        return pd.DataFrame({'frq': frq, 'amp': amp, 'phs': phs, 'fft': It})
    
    def plot(self):
        fig, ax = Plot.new_figure(nrows=1, ncolumns=2, fig_size=[9, 5], font_size=12)
        Plot.time_domain(ax[0], self)
        Plot.freq_domain(ax[1], self)
    
    @classmethod
    def data_list(cls, file_list, *indices, dt=0.01, time_range=(None, None), delayline_zero=None):
        data_list_ = []
        for i in indices:
            if isinstance(i, int):
                data_list_.append(Data(file_list.get_file(i), dt, time_range, delayline_zero))
            elif isinstance(i, (list, tuple)) and len(i)==2:
                i_list = list(range(i[0], i[-1]+1))
                for j in i_list:
                    data_list_.append(Data(file_list.get_file(j), dt, time_range, delayline_zero))
            else:
                print(f'Warning: {i} is not an integer, nor a list of the type [imin, imax]')
        return data_list_
                

class Transmittance:
    def __init__(self, data, ref):
        self._data, self._ref = self._verify_data(data, ref)
        self._T    = self._compute_T()
        self._func = None
        
    @property
    def data(self): return self._data
    @property
    def ref(self): return self._ref
    @property
    def T(self): return self._T
    @property
    def func(self): return self._func
    
    def _verify_data(self, data, ref):
        if not np.array_equal(data.freq.frq, ref.freq.frq):
            print('Data and reference frequency points are not equivalent')
            return None, None
        else:
            return data, ref
    
    def _compute_T(self):
        if self.data and self.ref:
            freq = self.data.freq.frq
            Es   = self.data.freq.fft
            Eref = self.ref.freq.fft
            T    = Es/Eref
            
            return pd.DataFrame({'freq': freq, 'complex': T, 'real': np.real(T),
                                 'imag': np.imag(T), 'ampl': np.abs(T), 'phase': np.angle(T)})
        
    def fit(self, func, p0, freq_range=(0, 1.5), bounds=(-np.inf, np.inf)):
        self._func = func
        
        T = self.T.loc[self.T.freq.between(*freq_range)]
        
        real_func = lambda *args: np.real(func(*args))
        imag_func = lambda *args: np.imag(func(*args))
        
        ropt, rcov = fit(real_func, T.freq, T.real, p0=p0, bounds=bounds)
        iopt, icov = fit(imag_func, T.freq, T.imag, p0=p0, bounds=bounds)
        rerr = np.sqrt(np.diag(rcov))
        ierr = np.sqrt(np.diag(icov))
        
        return pd.DataFrame({'r': ropt, 'rerr': rerr, 'i': iopt, 'ierr': ierr})


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
                   font_size=16, dpi=100):
        with plt.rc_context({'font.size': font_size}):
            fig, ax = plt.subplots(nrows, ncolumns, figsize=fig_size, dpi=dpi)
        fig.suptitle(title)
        return fig, ax
        
    @classmethod
    def time_domain(cls, ax, *data_list, label=None, leg_title=None, colormap='rainbow',
                    xlabel='Time (ps)', ylabel='Photocurrent (nA)', linewidth=1):
        colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(data_list)))
        
        for i, data in enumerate(data_list):
            l = cls.get_label(data, label, i)
            ax.plot(data.time.t, data.time.I, color=colors[i], label=l, linewidth=linewidth)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()

            
    @classmethod
    def freq_domain(cls, ax, *data_list, y='amp', yscale='log', ymin=1e-4,
                    ymax=1, label=None, leg_title=None, colormap='rainbow',
                    xlabel='Frequency (THz)', ylabel='Amplitude (a. u.)', linewidth=1):
        colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(data_list)))
            
        for i, data in enumerate(data_list):
            l = cls.get_label(data, label, i)
            ax.plot(data.freq.frq, np.abs(data.freq[y]), color=colors[i], label=l, linewidth=linewidth)
            
        if leg_title: ax.legend(title=leg_title)
        ax.set_yscale(yscale)
        ax.set_xlim([-0.2, 5.2])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        
    
    @classmethod
    def T(cls, ax, Tdata, col='ampl', xlim=[-.1, 2], ylim=[4, -4], fit=False,
          custom_func=None, custom_params=None):
        
        ax.plot(Tdata.T.freq, Tdata.T[col], 'k.')
        if fit:
            opt = T.fit()
            

class FileList:
    def __init__(self, folder):
        self._folder = folder
        self._time   = tm.strftime('%d/%m/%Y @ %H:%M:%S')
        self._table  = self.frame(folder, *os.listdir(folder))
        
    def __repr__(self):
        return f'File list of "{self._folder}" generated on {self._time}'
    
    def get_file(self, index):
        return self._table.file[index]
    
    def get_table(self, date=None, hide_file=True, max_rows=60):
        pd.set_option('display.max_rows', max_rows)
        df = self._table if not hide_file else self._table.drop('file', axis=1)
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