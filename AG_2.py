import numpy as np
import struct
import time
from scipy.stats import bernoulli
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
from libs.aluSynth import Synth
from scipy.io import wavfile
from progressbar import progressbar

class Individuo():
    def __init__(self, bits, fitness):
        self.bits = bits
        self.fitness = fitness

class AG:
    def __init__(self, path):
        self.n_pop = 10
        self.n_crossover = int(self.n_pop/2)
        self.len_ind = 681
        self.exponent_len = 4
        self.mut_prop = 0.5
        self.bits_ha_mutar = 10
        self.poblacion = []
        self.soundRef = self.read_wav(path)
        self.synth = Synth()


    def getBest(self):
        resp = self.obtenerFenotipo2(self.poblacion[0].bits)
        return resp

    
    def generar_poblacion_inicial(self):
        for i in range(self.n_pop):
            bits = ""
            
            for i in range(self.len_ind):
                x = np.random.choice(["0","1"])
                bits += x
                
            individuo = Individuo(bits, self.fitness(bits))
            self.poblacion.append(individuo)

            
    def selectParents(self):
        parents = []

        p1 = self.poblacion[0]
        p2 = self.poblacion[1]

        parents.append((p1,p2))
        
        for _ in range(1,self.n_crossover):
            index = list(range(len(self.poblacion)))
            index_p1 = np.sort(np.random.choice(index, size=len(self.poblacion)//3))[0]
            index_p2 = np.sort(np.random.choice(index, size=len(self.poblacion)//3))[0]
            
            p1 = self.poblacion[index_p1]
            p2 = self.poblacion[index_p2]
            
            parents.append((p1,p2))

        return parents
            

    def single_point_crossover(self, p1, p2, punto_de_corte):
        offspring1 = self.mutation(p1[:punto_de_corte] + p2[punto_de_corte:])
        offspring2 = self.mutation(p2[:punto_de_corte] + p1[punto_de_corte:])

        return offspring1, offspring2

    
    def crossover(self):
        padres = self.selectParents()
        for p1, p2 in padres:
            os1, os2 = self.single_point_crossover(p1.bits, p2.bits, 10)
            self.poblacion.append(Individuo(os1, self.fitness(os1)))
            self.poblacion.append(Individuo(os2, self.fitness(os2)))

    
    def mutation(self, individuo):
        if bernoulli.rvs(self.mut_prop) == 1:
            bits_a_cambiar = np.random.choice(range(self.len_ind), size=(self.bits_ha_mutar))
            for bit in bits_a_cambiar:
                if individuo[bit] == "0":
                    aux = "1"
                else:
                    aux = "0"
                individuo = individuo[:bit] + aux + individuo[bit+1:]
            
        return individuo


    def bitToFloat(self, individuo):
        try:
            f = int(individuo.replace(' ', ''), 2)  # Remove spaces
            resp = struct.unpack('f', struct.pack('I', f))[0]
        except:
            resp = 1e10
        return resp

    
    def bitToInt(self, individuo):
        try:
            resp = int(individuo.replace(' ', ''), 2)
        except:
            resp = 1e10
        return resp


    def obtenerFenotipo(self, individuo):
        parametros = []
        parametros.append(self.bitToInt(individuo[0]))
        index = 1
        
        for i in range(4):
            parametros.append(self.bitToInt(individuo[index:index+2]))
            index += 2
    
        for i in range(21):
            parametros.append(self.bitToFloat(individuo[index:index+32]))
            index += 32

        return parametros


    def obtenerFenotipo2(self, individuo):
        param_vec = [1, 0, 0, 0, 0, 1, 1, 1000, 880, 0, 12, 0.1, 1, 0.2, 0.1, 0.5, 0.5, 220, 0, 0, 0.1, 1, 0.2, 0.1, 0.5, 0.5]
        param_vec[8] = self.bitToFloat(individuo)
        return param_vec

    
    def fitness(self, individuo):
        parametros = self.obtenerFenotipo2(individuo)
        band = True
        valor = 0

        if parametros[1] > 2:
            valor += 1e5
            band = False

        if parametros[2] > 2:
            valor += 1e5
            band = False

        if parametros[3] > 2:
            valor += 1e5
            band = False

        if parametros[4] > 2:
            valor += 1e5
            band = False

        if parametros[5] < 0 or parametros[5] > 20:
            valor += 1e5
            band = False

        if parametros[6] < 0 or parametros[6] > 1:
            valor += 1e5
            band = False

        if parametros[7] < 20 or parametros[7] > 20000:
            valor += 1e5
            band = False

        if parametros[8] < 20 or parametros[8] > 20000:
            valor += 1e5
            band = False

        if parametros[9] > 0 or parametros[9] < -3:
            valor += 1e5
            band = False

        if parametros[10] < 0 and parametros[10] > 20:
            valor += 1e5
            band = False

        if parametros[11] < 0 or parametros[11] > 0.5:
            valor += 1e5
            band = False

        if parametros[12] < 0 or parametros[12] > 1:
            valor += 1e5
            band = False

        if parametros[13] < 0 or parametros[13] > 0.5:
            valor += 1e5
            band = False

        if parametros[14] < 0 or parametros[14] > 0.9:
            valor += 1e5
            band = False

        if parametros[15] < 0 or parametros[15] > 0.5:
            valor += 1e5
            band = False

        if parametros[16] < 0 or parametros[16] > 0.5:
            valor += 1e5
            band = False

        if parametros[17] < 20 or parametros[17] > 20000:
            valor += 1e5
            band = False

        if parametros[18] > 0 or parametros[18] < -3:
            valor += 1e5
            band = False

        if parametros[19] < 0 or parametros[19] > 20:
            valor += 1e5
            band = False

        if parametros[20] < 0 or parametros[20] > 0.5:
            valor += 1e5
            band = False

        if parametros[21] < 0 or parametros[21] > 1:
            valor += 1e5
            band = False

        if parametros[22] < 0 or parametros[22] > 0.5:
            valor += 1e5
            band = False

        if parametros[23] < 0 or parametros[23] > 0.9:
            valor += 1e5
            band = False

        if parametros[24] < 0 or parametros[24] > 0.5:
            valor += 1e5
            band = False

        if parametros[25] < 0 or parametros[25] > 0.5:
            valor += 1e5
            band = False

        valor += np.random.randint(0,100)
            
        if band:
            #print(f"PARAMETROS: {parametros}")
            self.synth.update_param(parametros)
            error = self.compare_sounds_1(self.synth.wave())
            valor += error
            #print(f"ERROR: {valor}")

        return valor
            

    def order_and_select(self):
        self.mergeSort(self.poblacion, 0, len(self.poblacion)-1)
        if len(self.poblacion) > 5*self.n_pop:
            self.poblacion = self.poblacion[:self.n_pop]


    def merge(self, arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m

	# create temp arrays
        L = [0] * (n1)
        R = [0] * (n2)

	# Copy data to temp arrays L[] and R[]
        for i in range(0, n1):
            L[i] = arr[l + i]

        for j in range(0, n2):
            R[j] = arr[m + 1 + j]

	# Merge the temp arrays back into arr[l..r]
        i = 0	 # Initial index of first subarray
        j = 0	 # Initial index of second subarray
        k = l	 # Initial index of merged subarray

        while i < n1 and j < n2:
            if L[i].fitness <= R[j].fitness:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

	# Copy the remaining elements of L[], if there are any
        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1

	# Copy the remaining elements of R[], if there are any
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1

    # l is for left index and r is right index of the sub-array of arr to be sorted
    def mergeSort(self, arr, l, r):
        if l < r:
	    # Same as (l+r)//2, but avoids overflow for large l and h
            m = l+(r-l)//2
            
	    # Sort first and second halves
            self.mergeSort(arr, l, m)
            self.mergeSort(arr, m+1, r)
            self.merge(arr, l, m, r)


    # Lectura de un archivo de audio
    def read_wav(self, path):
        _, data = wavfile.read(path)
        return data

    # Comparación de dos sonidos V1
    def compare_sounds_1(self, soundAprox):
        error = (abs(self.soundRef) - abs(soundAprox)) ** 2
        error_sum = sum(error)
        return error_sum

    def imprimePoblacion(self):
        for i,p in enumerate(self.poblacion):
            print(f"p{i}: {p.bits}, {self.bitToFloat(p.bits)},{p.fitness}")
        input()

if __name__ == '__main__':
    numero_generaciones = 500
    ag = AG('Samples/prueba_genotipo.wav')
    ag.len_ind = 32
    ag.n_pop = 100
    ag.mut_prop = 1
    ag.bits_ha_mutar = 32
    errores = []

    print("Inicio del AG.")
    print(f"Queremos aproximar: {ag.soundRef}")

    ag.generar_poblacion_inicial()

    #print(f"Generacion inicial: {ag.poblacion[0].bits}")
    
    for i in progressbar(range(1,numero_generaciones+1)):
        inicio = time.time()
        #ag.imprimePoblacion()
        ag.crossover()
        ag.order_and_select()
        if i%2 == 0 and ag.bits_ha_mutar>5:
            ag.mut_prop -= 0.01
            ag.bits_ha_mutar -= 1
            if ag.mut_prop < 0.4:
                ag.mut_prop = 0.4
        aux = ag.poblacion[0].fitness
        errores.append(aux)
        fin = time.time()
        if aux < 1:
            break
        #print(f"Generación: {i},   Error: {aux},   Probabilidad de mutación: {ag.mut_prop},   Bits a mutar: {ag.bits_ha_mutar}, tiempo: {fin-inicio}")
        #if i%100 == 0:
        #print(f"Best: {ag.poblacion[0].bits}")
        #print(ag.poblacion)

    print(ag.getBest())

    plt.plot(errores)
    plt.show()
    
    synth = Synth()
    synth.update_param(ag.getBest())
    synth.write_wav('resultado')
        
