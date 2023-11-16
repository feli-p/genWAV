# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:00:17 2023

@author: lilsa
"""
import struct

def float_to_binary(input: float) -> str:
    return ''.join(format(c, '08b') for c in struct.pack('!f', input))

def main():
    filter_type = '{0:01b}'.format(1)
    algorithm = '{0:02b}'.format(0)
    LFO_waveForm = '{0:02b}'.format(0)
    A_waveForm = '{0:02b}'.format(0)
    B_waveForm = '{0:02b}'.format(0)
    LFO_freq = float_to_binary(1)
    LFO_ammount = float_to_binary(1)
    filter_cuttof = float_to_binary(1000)
    A_freq = float_to_binary(440)
    A_gain = float_to_binary(0)
    A_beta = float_to_binary(12)
    A_envAttk = float_to_binary(0.1)
    A_envPeak = float_to_binary(1)
    A_envDecay = float_to_binary(0.2)
    A_envSustain = float_to_binary(0.1)
    A_envSusLen = float_to_binary(0.5)
    A_envRelease = float_to_binary(0.5)
    B_freq = float_to_binary(220)
    B_gain = float_to_binary(0)
    B_beta = float_to_binary(0)
    B_envAttk = float_to_binary(0.1)
    B_envPeak = float_to_binary(1)
    B_envDecay = float_to_binary(0.2)
    B_envSustain = float_to_binary(0.1)
    B_envSusLen = float_to_binary(0.5)
    B_envRelease = float_to_binary(0.5)
    

    chain = filter_type 
    chain += algorithm 
    chain += LFO_waveForm 
    chain += A_waveForm 
    chain += B_waveForm 
    chain += LFO_freq 
    chain += LFO_ammount 
    chain += filter_cuttof 
    chain += A_freq 
    chain += A_gain 
    chain += A_beta 
    chain += A_envAttk 
    chain += A_envPeak 
    chain += A_envDecay 
    chain += A_envSustain
    chain += A_envSusLen
    chain += A_envRelease
    chain += B_freq 
    chain += B_gain 
    chain += B_beta 
    chain += B_envAttk 
    chain += B_envPeak 
    chain += B_envDecay 
    chain += B_envSustain
    chain += B_envSusLen
    chain += B_envRelease
    print(chain)
    print(len(chain))

if __name__ == '__main__':
    main()