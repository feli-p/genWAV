"""
Algoritmo genético para replicar el sonido de osciloscopios específicos.

Hecho por Losa lucines GPT.
"""
import sys

sys.dont_write_bytecode = True

from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
from libs.aluSynth import Synth
from scipy.io import wavfile

# Lectura de un archivo de audio
def read_wav(path):
    _, data = wavfile.read(path)
    return data

# Comparación de dos sonidos V1
def compare_sounds_1(soundRef, soundAprox):
    error = (abs(soundRef) - abs(soundAprox)) ** 2
    error_sum = sum(error)
    return error_sum

# Comparación de dos sonidos V2
def compare_sounds_2(soundRef, soundAprox):
    spectrum1, _, _, _ = plt.specgram(soundRef, NFFT=256, Fs=44100, noverlap=120, scale='dB', cmap='jet_r')
    spectrum2, _, _, _ = plt.specgram(soundAprox, NFFT=256, Fs=44100, noverlap=120, scale='dB', cmap='jet_r')
    error = (abs(spectrum1) - abs(spectrum2)) ** 2 
    error_sum1 = sum(error)
    error_sum = sum(error_sum1)
    return error_sum

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define range for input
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# perform the genetic algorithm search
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))

if __name__ == '__main__':
	
    # Instancia de un sintetizador
    synth1 = Synth()
    synth1.name = "Synth 1"
    
    # PARÁMETROS
    
    # Algoritmo para la modulación o adición ([0] A -> B , [1] B -> A , [2] A + B)
    synth1.algorithm = 0 
    # Parámetros del LFO
    synth1.LFO.frequency = 4 # Hz
    synth1.LFO.wave_form = 0 # ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.LFO.ammount = 1.0 # % [0 - 1] indica que tanto modula el LFO (0 lo desactiva)
    # Parámetros del filtro
    synth1.filter.cutoff_freq = 200 # Hz
    synth1.filter.type = 1 # [1] -> lowpass, [2] -> highpass
    # Parámetros del Oscilador A
    synth1.osc_A.frequency = 391.995/2 # Hz
    synth1.osc_A.gain = 0 # dB
    synth1.osc_A.wave_form = 0 # ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.osc_A.beta = 12 # Constante para la modulación (0 desactiva la modulación)
    # Parámetros de la envolvente del Oscilador A
    synth1.osc_A.envelope.attack = 0.001 # sec 
    synth1.osc_A.envelope.peak = 1.0 # %
    synth1.osc_A.envelope.decay = 2.0 # sec
    synth1.osc_A.envelope.sustain = 0.1 # %
    synth1.osc_A.envelope.sustain_length = 1.0 # sec
    synth1.osc_A.envelope.release = 0.4 # sec
    # Parámetros del Oscilador B
    synth1.osc_B.frequency = 293.665 # Hz
    synth1.osc_B.gain = -0.2 # dB
    synth1.osc_B.wave_form = 1 # ([0] -> sin, [1] -> sawtooth, [2] -> square)
    synth1.osc_B.beta = 0 # Constante para la modulación
    # Parámetros de la envolvente del Oscilador B
    synth1.osc_B.envelope.attack = 0.01 # sec 
    synth1.osc_B.envelope.peak = 0.5 # %
    synth1.osc_B.envelope.decay = 2.0 # sec
    synth1.osc_B.envelope.sustain = 0.1 # %
    synth1.osc_B.envelope.sustain_length = 1.0 # sec
    synth1.osc_B.envelope.release = 0.4 # sec
    
    # Escribir archivo .wav
    synth1.write_wav("prueba")
    
    # Gráfica
    plt.plot(synth1.sample_time, synth1.wave())
    plt.show()
    synth1.osc_A.envelope.wave()
