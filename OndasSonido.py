import matplotlib
import pyaudio as pa
import struct

import numpy as np
import scipy.fftpack as fourier
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# %matplotlib notebook

FRAMES = 1024 * 8  # Tamaño del paquete a procesar
FORMAT = pa.paInt16  # Formato de lectura INT 16 bits
CHANNELS = 1
Fs = 22050    # Frecuencia de muestreo típica para audio
NOTE_MIN = 40  # E2  6ta cuerda entre E2 y E4 hay 24 semitonos  PEND: hacer video sobre eso
NOTE_MAX = 64  # E4  1

FRAMES_PER_FFT = 16  # FFT takes average across how many frames?   -- cantidad de frames analizads por la FFT (transformada rapida de fourier)

#variables globales para compartir con otros files.py
varPub = 10
numeroMIDI, frecHz, notaProxima, distNotaProxima = 0, 0, 0, 0
#ResMidi, ResFrec, ResNota, ResDist = 1 , 2 , 3 , 4

######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (sof
# resolution increases); however, it will incur more delay to process
# new sounds.

SAMPLES_PER_FFT = FRAMES * FRAMES_PER_FFT
FREQ_STEP = float(Fs) / SAMPLES_PER_FFT

######################################################################
NOTE_NAMES = 'Mi Fa Fa# Sol Sol# La La# Si Do Do# Re Re#'.split()

p = pa.PyAudio()


stream = p.open(  # Abrimos el canal de audio con los parámeteros de configuración
    format=FORMAT,
    channels=CHANNELS,
    rate=Fs,
    input=True,
    output=True,
    frames_per_buffer=FRAMES
)

## Creamos una gráfica con 2 subplots y configuramos los ejes

fig, (ax, ax1) = plt.subplots(2)

x_audio = np.arange(0, FRAMES, 1)
x_fft = np.linspace(0, Fs, FRAMES)

line, = ax.plot(x_audio, np.random.rand(FRAMES), 'r')
line_fft, = ax1.semilogx(x_fft, np.random.rand(FRAMES), 'b')

ax.set_ylim(-32500, 32500)
ax.ser_xlim = (0, FRAMES)

ax1.set_xlim(NOTE_MIN, NOTE_MAX)

fig.show()

F = (Fs / FRAMES) * np.arange(0, FRAMES // 2)  # Creamos el vector de frecuencia para encontrar la frecuencia dominante

def freq_to_number(f): return 64 + 12 * np.log2(f / 329.63)


def number_to_freq(n): return 329.63 * 2.0**((n - 64) / 12.0)


def note_name(n):
    return NOTE_NAMES[n % NOTE_MIN % len(NOTE_NAMES)] + str(int(n / 12 - 1))

def note_to_fftbin(n): return number_to_freq(n) / FREQ_STEP

def PublicaNota():
    while True:

        data = stream.read(FRAMES)  # Leemos paquetes de longitud FRAMES
        dataInt = struct.unpack(str(FRAMES) + 'h', data)  # Convertimos los datos que se encuentran empaquetados en bytes

        line.set_ydata(dataInt)  # Asignamos los datos a la curva de la variación temporal

        M_gk = abs(fourier.fft(dataInt) / FRAMES)  # Calculamos la FFT y la Magnitud de la FFT del paqute de datos

        ax1.set_ylim(0, np.max(M_gk + 10))
        line_fft.set_ydata(M_gk)  # Asigmanos la Magnitud de la FFT a la curva del espectro

        M_gk = M_gk[0:FRAMES // 2]  # Tomamos la mitad del espectro para encontrar la Frecuencia Dominante
        Posm = np.where(M_gk == np.max(M_gk))
        F_fund = F[Posm]  # Encontramos la frecuencia que corresponde con el máximo de M_gk

        print(int(F_fund))  # Imprimimos el valor de la frecuencia dominante

        fig.canvas.draw()
        fig.canvas.flush_events()

if __name__ == '__main__':   #se ejecuta solo se llama desde este file
        PublicaNota()