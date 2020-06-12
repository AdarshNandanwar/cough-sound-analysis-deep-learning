# -*- coding: utf-8 -*-

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import sklearn

FIG_SIZE = (15,10)

# file = "dataset/train/cough/1745-9974-2-1-S1.mp3"
file = "dataset/train/cough/1745-9974-2-1-S9.mp3"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=22050)
librosa.get_duration(y=signal, sr=sample_rate)

# WAVEFORM
# display waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")


# FFT -> power spectrum
# perform Fourier transform
fft = np.fft.fft(signal)

# calculate abs values on complex numbers to get magnitude
# magnitude
spectrum = np.abs(fft)

# create frequency variable
# creates len(spectrum) number of equally spaced numbers from 0 to sample_rate (inclusive)
frequency = np.linspace(0, sample_rate, len(spectrum))

# plot spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(frequency, spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Full Power spectrum")

# take half of the spectrum and frequency
# since the plot is symetric
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_frequency = frequency[:int(len(spectrum)/2)]

# plot spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_frequency, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Half Power spectrum")


# STFT -> spectrogram
hop_length = 512 # in num. of samples

# setting n_ftt as a power of 2 optimizes the speed of fft
n_fft = 2048 # window in num. of samples

# the intervals created using hop_length and n_fft may overlap
# which is desirable in certain cases


# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))
print("STFT window overlap duration is: {}s".format(n_fft_duration-hop_length_duration))

# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")






# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

# show plots
plt.show()






# Normalising function for data visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)





# Spectral Centeriod
spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0]
print(spectral_centroids.shape)
# Computing the time variable for visualization
plt.figure(figsize=FIG_SIZE)
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(signal, sr=sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
plt.title("Spectral Centeriod")





# Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(signal+0.01, sr=sample_rate)[0]
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr=sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title("Spectral Rolloff")





# Spectral Bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=4)[0]
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr=sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))
plt.title("Spectral Bandwidth")





# Zero-Crossing Rate
#Plot the signal:
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr=sample_rate)
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=FIG_SIZE)
plt.plot(signal[n0:n1])
plt.grid()
zero_crossings = librosa.zero_crossings(signal[n0:n1], pad=False)
print(sum(zero_crossings))
plt.title("Zero-Crossing Rate")





# Chroma Features
chromagram = librosa.feature.chroma_stft(signal, sr=sample_rate, hop_length=hop_length)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.title("Chroma Features")