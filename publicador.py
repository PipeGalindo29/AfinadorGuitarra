from time import time

import matplotlib
import pyaudio as pa
import struct

import numpy as np
import scipy.fftpack as fourier
import matplotlib.pyplot as plt

NOTE_MIN = 40       # E2  6ta cuerda entre E2 y E4 hay 24 semitonos
NOTE_MAX = 64       # E4  1
FSAMP = 22050       # Sampling frequency in Hz  -- T=0,000045351
FRAME_SIZE = 2048   # samples per frame         -- T=0,000488281
FRAMES_PER_FFT = 16  # FFT takes average across how many frames?   -- cantidad de frames analizads por la FFT (transformada rapida de fourier)

#variables globales para compartir con otros files.py
varPub = 10
numeroMIDI, frecHz, notaProxima, distNotaProxima = 0, 0, 0, 0

SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT
FREQ_STEP = float(FSAMP) / SAMPLES_PER_FFT


NOTE_NAMES = 'Mi Fa Fa# Sol Sol# La La# Si Do Do# Re Re#'.split()



def freq_to_number(f): return 64 + 12 * np.log2(f / 329.63)


def number_to_freq(n): return 329.63 * 2.0**((n - 64) / 12.0)


def note_name(n):
    return NOTE_NAMES[n % NOTE_MIN % len(NOTE_NAMES)] + str(int(n / 12 - 1))



def note_to_fftbin(n): return number_to_freq(n) / FREQ_STEP


imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN - 1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX + 1))))

# Allocate space to run an FFT.
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

# Initialize audio
stream = pa.PyAudio().open(format=pa.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)  #PEND: este de la frecuencia de transmision??????

stream.start_stream()

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
print()


def PublicaNota():	
    while stream.is_active():   #T0: ojo: Hemos calculado que el tiempo de cada iteracion es 0.1431541443 seg  osea 6.98 Hz de frec 
          
        start_time = time()    #T0: para calcular tiempo de cada iteracion

	    # Shift the buffer down and new data in
        buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
        buf[-FRAME_SIZE:] = np.frombuffer(stream.read(FRAME_SIZE), np.int16)


        fft = np.fft.rfft(buf * window)


        freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP


        n = freq_to_number(freq)
        n0 = int(round(n))

        global num_frames
        num_frames += 1


        global numeroMIDI, frecHz, notaProxima, distNotaProxima
        numeroMIDI, frecHz, notaProxima, distNotaProxima =  n, freq, note_name(n0), (n-n0)

        print('Num MIDI {:7.2f} Frec: {:7.2f} Hz     Nota prox: {:>3s} {:+.2f}'.format( numeroMIDI, frecHz, notaProxima, distNotaProxima ))
	
        global varPub
        varPub = numeroMIDI


        elapsed_time = time() - start_time
        print("Elapsed time: %0.10f seconds." % elapsed_time)

if __name__ == '__main__':
        PublicaNota()





