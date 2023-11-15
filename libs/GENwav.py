"""
GENwav replica el sonido de un osciloscopio usando algoritmos evolutivos.

Hecho por Losa lucines GPT.
"""
#%%

import numpy as np
import scipy.io.wavfile as wav
# from scipy import signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100 # Hz
SAMPLE_LENGTH = 4 # Sec
N = SAMPLE_LENGTH * SAMPLE_RATE
time = np.linspace(0,SAMPLE_LENGTH,N) # vector de tiempo

# Señales
def sawtooth(x):
    return (x + np.pi) / np.pi % 2 - 1

def square(x):
    return 2*(2*np.floor(0.16*x) - np.floor(2*0.16*x)) + 1

class Envelope:
    # Envolvente lineal de la señal (fade in/out), modulación de amplitud (volumen) de la señal
    def __init__(self):
        self.name = ""
        self.attack = 0.2 # sec
        self.peak = 1 # % [0 - 1] 
        self.decay = 0.5 # sec
        self.sustain = 0.75 # % [0 - 1]
        self.release = 1 # sec
        self.sustain_length = 3 # sec
    
    def wave(self):
        output = np.zeros((N,))
        # obtener índices correspondientes
        attack_indx = self.get_time_index(self.attack)
        decay_indx = self.get_time_index(self.attack+self.decay)
        sustain_indx = self.get_time_index(self.attack+self.decay+self.sustain_length)
        release_indx = self.get_time_index(self.attack+self.decay+self.sustain_length+self.release)
        
        # Definir el envolvente
        CA = self.peak / self.attack
        CD = (self.sustain - self.peak) / self.decay
        CR_1 = self.attack + self.decay + self.sustain_length
        CR_2 = -self.sustain / self.release
        
        a = self.peak-self.sustain
        l1 = -7/(self.decay)
        l2 = -5/self.release

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

        plt.plot(time, output)
        plt.show()
        
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
        self.ammount = 0 # (0 - 1) 
        self.wave_form = "sine" # String (sin, sawtooth, square)
    
    def wave(self):
        if (self.wave_form == "sine"):
            waveform = np.sin
        elif (self.wave_form == "sawtooth"):
            waveform = sawtooth
        elif (self.wave_form == "square"):
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
        self.type = "lowpass" # String (lowpass, highpass)

class OSC:
    # Oscilador, señal principal o que modula en frecuencia a otra señal
    def __init__(self):
        self.name = ""
        self.frequency = 440 # Hz
        self.gain = -3 # dB
        self.wave_form = "sine" # String (sin, sawtooth, square)
        self.beta = 5 # Modulating coefficient
        self.freq_modulator = np.zeros((N,))  # OSC
        self.envelope = Envelope() # Envelope
        self.LFO = LFO() # LFO
    
    def wave(self):
        if (self.wave_form == "sine"):
            waveform = np.sin
        elif (self.wave_form == "sawtooth"):
            waveform = sawtooth
        elif (self.wave_form == "square"):
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
        
        # Filter
        
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
        self.algorithm = 1 # Int ([1] B -> A, [2] A -> B, [3] A + B)
    
    def wave(self):
        # crear un archivo .wav con la señal final
        self.osc_A.LFO = self.LFO
        self.osc_B.LFO = self.LFO
        output = np.zeros((SAMPLE_LENGTH * SAMPLE_RATE,))
        if (self.algorithm == 1):
            # B modula A
            self.osc_A.freq_modulator = self.osc_B.wave()
            output = self.osc_A.wave()
        elif (self.algorithm == 2):
            # A modula B
            self.osc_B.freq_modulator = self.osc_A.wave()
            output = self.osc_B.wave()
        else:
            output = self.osc_A.wave() + self.osc_B.wave()
        return output
        
    def write_wav(self, file_name):
        output = self.wave()
        file_name += ".wav"
        wav.write(file_name, SAMPLE_RATE, output.astype(np.float32))
