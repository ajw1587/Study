import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = '../data/audio/Music_Data/Park_Hyo_Shin/Park Hyo Shin-Wild Flower_first.wav'
amp, sr = librosa.load(audio_path)
FIG_SIZE = (10, 15)

print(amp.shape)
print(type(amp))
print(sr)
print(type(sr))

# plt.figure()
# librosa.display.waveplot(amp, sr, alpha = 0.5)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Waveform")
# plt.show()

# STFT 시간 정보를 보존한 FFT
stft_result = librosa.stft(amp, n_fft = 4096, win_length = 4096, hop_length = 1024)
D = np.abs(stft_result)
S_dB = librosa.power_to_db(D, ref = np.max)
librosa.display.specshow(S_dB, sr = sr, hop_length = 1024,
                         y_axis = 'linear', x_axis = 'time')
plt.colorbar(format = '%2.0f dB')
plt.show()