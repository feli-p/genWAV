"""
Sintetizador digital de sonidos.

Hecho por Losa lucines GPT.
"""

import numpy as np
import scipy.io.wavfile as wav
# from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100 # Hz
SAMPLE_LENGTH = 2 # Sec
N = SAMPLE_LENGTH * SAMPLE_RATE
time = np.linspace(0, SAMPLE_LENGTH, N) # vector de tiempo [0, SAMPLE_LENGTH]

# Señales
def sawtooth(x):
    return (x + np.pi) / np.pi % 2 -1

def square(x):
    return 2*(2*np.floor(0.16*x) - np.floor(2*0.16*x)) + 1

class Envelope:
    # Envolvente lineal de la señal (fade in/out), modulación de amplitud (volumen) de la señal
    def __init__(self):
        self.name = ""
        self.attack = 0.01 # sec
        self.peak = 1 # % [0 - 1] 
        self.decay = 0.5 # sec
        self.sustain = 0.5 # % [0 - 1]
        self.release = 1 # sec
        self.sustain_length = 3 # sec
    
    def wave(self):
        output = np.zeros((N,))
        # obtener índices correspondientes
        attack_indx = self.get_time_index(self.attack)
        decay_indx = self.get_time_index(self.attack+self.decay)
        sustain_indx = self.get_time_index(self.attack+self.decay+self.sustain_length)
        # release_indx = self.get_time_index(self.attack+self.decay+self.sustain_length+self.release)
        
        # Definir el envolvente.
        CA = self.peak / self.attack
        # CD = (self.sustain - self.peak) / self.decay
        CR_1 = self.attack + self.decay + self.sustain_length
        # CR_2 = -self.sustain / self.release
        
        a = self.peak-self.sustain
        l1 = -7/(self.decay)
        l2 = -3/self.release

        output[decay_indx+1:sustain_indx+1] = self.sustain
        for n in range(N):
            if (n >= 0 and n <= attack_indx):
                output[n] = time[n] * CA
            elif (n > attack_indx and n <= decay_indx):
                output[n] = a*np.exp(l1*(time[n]-self.attack)) + self.sustain
            elif (n > sustain_indx):
                output[n] = self.sustain*np.exp(l2*(time[n] - CR_1))
            else:
                pass

        # plt.plot(time, output)
        # plt.show()
        
        return output
    
    def get_time_index(self, mark):
        n = 0
        while (n < N and time[n] < mark):
            n += 1
        return n

class LFO:
    # "Low Frequency Oscilator", modulación de frecuencia de la señal
    def __init__(self):
        self.name = ""
        self.frequency = 0 # Hz
        self.ammount = 0 # % (0 - 1) 
        self.wave_from = 0 # Int ([0] -> sin, [1] -> sawtooth, [2] -> square)
    
    def wave(self):
        if (self.wave_form == 0):
            waveform = np.sin
        elif (self.wave_form == 1):
            waveform = sawtooth
        elif (self.wave_form == 2):
            waveform = square
        else:
            return -1
        
        output = np.zeros((N,))
        
        w = 2*np.pi*self.frequency
        for n in range(N):
            output[n] = waveform(w*time[n])
        
        amplitude = 10 ** (0 / 2)
        # output *= self.ammount
        output *= amplitude

        return output

class Filter:
    # Filter
    def __init__(self):
        self.name = ""
        self.cutoff_freq = 150 # Hz
        self.order = 5 # 
        self.type = 0 # [0] -> lowpass, [1] -> highpass
    
    def butter_filter(self, signal):
        if (self.type == 0):
            type_str = "low"
        elif (self.type == 1):
            type_str = "high"
        else:
            return -1
        b, a = butter(self.order, self.cutoff_freq, fs=SAMPLE_RATE, btype=type_str, analog=False)
        output = lfilter(b,a,signal)
        return output

class OSC:
    # Oscilador, señal principal o que modula en frecuencia a otra señal
    def __init__(self):
        self.name = ""
        self.frequency = 440 # Hz
        self.gain = -3 # dB
        self.wave_form = 0 # Int ([0] -> sin, [1] -> sawtooth, [2] -> square)
        self.beta = 5 # Modulating coefficient
        self.freq_modulator = np.zeros((N,))  # OSC
        self.envelope = Envelope() # Envelope
        self.LFO = LFO() # LFO
    
    def wave(self):
        if (self.wave_form == 0):
            waveform = np.sin
        elif (self.wave_form == 1):
            waveform = sawtooth
        elif (self.wave_form == 2):
            waveform = square
        else:
            return -1
        
        output = np.zeros((N,))
        LFO_signal = self.LFO.wave()
        
        w = 2*np.pi*self.frequency
        for n in range(N):
            output[n] = waveform(w*(time[n]) + self.beta*self.freq_modulator[n] + self.LFO.ammount*LFO_signal[n])
        
        amplitude = 10 ** (self.gain / 2)
        output *= amplitude
        
        # Envelope
        output *= self.envelope.wave()

        return output
    
class Synth:
    def __init__(self):
        self.name = ""
        self.osc_A = OSC()
        self.osc_A.name = "OSC A"
        self.osc_B = OSC()
        self.osc_B.name = "OSC B"
        self.filter = Filter() # Filter
        self.LFO = LFO() # LFO
        self.algorithm = 0 # Int ([0] B -> A, [1] A -> B, [2] A + B)
        self.sample_time = time # vector de tiempo [0, SAMPLE_LENGTH]
    
    def wave(self):
        # crear un archivo .wav con la señal final
        self.osc_A.LFO = self.LFO
        self.osc_B.LFO = self.LFO
        output = np.zeros((SAMPLE_LENGTH * SAMPLE_RATE,))
        if (self.algorithm == 0):
            # B modula A
            self.osc_A.freq_modulator = self.osc_B.wave()
            output = self.osc_A.wave()
        elif (self.algorithm == 1):
            # A modula B
            self.osc_B.freq_modulator = self.osc_A.wave()
            output = self.osc_B.wave()
        else:
            output = self.osc_A.wave() + self.osc_B.wave()
        
        f_output = self.filter.butter_filter(output) 
        
        return f_output
        
    def write_wav(self, file_name):
        output = self.wave()
        file_name += ".wav"
        wav.write(file_name, SAMPLE_RATE, output.astype(np.float32))


    def update_param(self, param):
        self.filter.type = param[0] # ([0] -> lowpass, [1] -> highpass)
        self.algorithm = param[1]
        self.LFO.wave_form = param[2] # Forma de la señal ([0] -> sin, [1] -> sawtooth, [2] -> square)
        self.osc_A.wave_form = param[3] # ([0] -> sin, [1] -> sawtooth, [2] -> square)
        self.osc_B.wave_form = param[4] # ([0] -> sin, [1] -> sawtooth, [2] -> square)
        self.LFO.frequency = param[5] # Hz
        self.LFO.ammount = param[6] # % [0 - 1] indica que tanto modula el LFO (0 lo desactiva)
        self.filter.cutoff_freq = param[7] # Hz
        self.osc_A.frequency = param[8] # Hz
        self.osc_A.gain = param[9] # dB
        self.osc_A.beta = param[10] # Constante para la modulación (0 desactiva la modulación)
        self.osc_A.envelope.attack = param[11] # sec 
        self.osc_A.envelope.peak = param[12] # %
        self.osc_A.envelope.decay = param[13] # sec
        self.osc_A.envelope.sustain = param[14] # %
        self.osc_A.envelope.sustain_length = param[15] # sec
        self.osc_A.envelope.release = param[16] # sec
        self.osc_B.frequency = param[17] # Hz
        self.osc_B.gain = param[18] # dB
        self.osc_B.beta = param[19] # Constante para la modulación
        self.osc_B.envelope.attack = param[20] # sec 
        self.osc_B.envelope.peak = param[21] # %
        self.osc_B.envelope.decay = param[22] # sec
        self.osc_B.envelope.sustain = param[23] # %
        self.osc_B.envelope.sustain_length = param[24] # sec
        self.osc_B.envelope.release = param[25] # sec
    
def main():
    
    # Instancia de un sintetizador
    synth1 = Synth()
    synth1.name = "Synth 1"
    
    # PARÁMETROS
    
    # Algoritmo para la modulación o adición ([0] A -> B , [1] B -> A , [2] A + B)
    synth1.algorithm = 0 
    # Parámetros del LFO
    synth1.LFO.frequency = 4 # Hz
    synth1.LFO.wave_form = 0 # Forma de la señal ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.LFO.ammount = 1.0 # % [0 - 1] indica que tanto modula el LFO (0 lo desactiva)
    # Parámetros del filtro
    synth1.filter.cutoff_freq = 500 # Hz
    synth1.filter.type = 0 # ([0] -> lowpass, [1] -> highpass)
    # Parámetros del Oscilador A
    synth1.osc_A.frequency = 391.995/2 # Hz
    synth1.osc_A.gain = 0 # dB
    synth1.osc_A.wave_form = 0 # ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.osc_A.beta = 12 # Constante para la modulación (0 desactiva la modulación)
    # Parámetros de la envolvente del Oscilador A
    synth1.osc_A.envelope.attack = 0.01 # sec 
    synth1.osc_A.envelope.peak = 1.0 # %
    synth1.osc_A.envelope.decay = 0.5 # sec
    synth1.osc_A.envelope.sustain = 0.1 # %
    synth1.osc_A.envelope.sustain_length = 0.5 # sec
    synth1.osc_A.envelope.release = 1.0 # sec
    # Parámetros del Oscilador B
    synth1.osc_B.frequency = 293.665 # Hz
    synth1.osc_B.gain = -0.2 # dB
    synth1.osc_B.wave_form = 1 # ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.osc_B.beta = 0 # Constante para la modulación
    # Parámetros de la envolvente del Oscilador B
    synth1.osc_B.envelope.attack = 0.01 # sec 
    synth1.osc_B.envelope.peak = 0.5 # %
    synth1.osc_B.envelope.decay = 0.5 # sec
    synth1.osc_B.envelope.sustain = 0.1 # %
    synth1.osc_B.envelope.sustain_length = 0.5 # sec
    synth1.osc_B.envelope.release = 1.0 # sec
    
    # Escribir archivo .wav
    synth1.write_wav("prueba")
    
    # Gráfica
    plt.plot(time, synth1.wave())
    plt.show()
 
if __name__ == '__main__':
    main()