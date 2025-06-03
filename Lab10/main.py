from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
from helpers import *  # предполагается, что здесь есть change_sample_rate


def robust_spectrogram(samples, sample_rate):
    n = len(samples)

    min_segment = 32
    target_segment = 256
    overlap_ratio = 0.75

    if n < min_segment:
        nperseg = max(8, n)
        noverlap = nperseg // 2
    elif n < target_segment:
        nperseg = min(n, max(min_segment, n // 2))
        noverlap = int(nperseg * overlap_ratio)
    else:
        nperseg = target_segment
        noverlap = int(target_segment * overlap_ratio)

    nperseg = min(nperseg, n)
    noverlap = min(noverlap, nperseg - 1)

    if noverlap >= nperseg:
        raise ValueError(f"Invalid parameters: noverlap ({noverlap}) must be less than nperseg ({nperseg})")

    print(f"DEBUG: nperseg = {nperseg}, noverlap = {noverlap}, len(samples) = {n}")

    return signal.spectrogram(
        samples,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum',
        mode='magnitude'
    )


def analyze_audio(filepath, formants=None, label=""):
    try:
        print(f"\nАнализ файла {filepath}...")

        if not os.path.exists(filepath):
            raise FileNotFoundError("Файл не найден")

        change_sample_rate(filepath)
        wav_path = f"results/wavs/{os.path.basename(filepath)}"

        sample_rate, samples = wavfile.read(wav_path)
        duration = len(samples) / sample_rate
        print(f"Длительность: {duration:.2f} сек, Частота: {sample_rate} Гц")
        print(f"Количество семплов: {len(samples)}")

        if samples.ndim > 1:
            samples = samples[:, 0]

        try:
            frequencies, times, spectro = robust_spectrogram(samples, sample_rate)
        except Exception as e:
            print(f"Ошибка при построении спектрограммы: {e}")
            return False

        spectro = np.maximum(spectro, 1e-10)
        log_spectro = np.log10(spectro)

        # Находим максимальную частоту с максимальной амплитудой
        max_amplitude_per_freq = np.max(spectro, axis=1)
        max_freq_idx = np.argmax(max_amplitude_per_freq)
        max_frequency = frequencies[max_freq_idx]
        print(f"Максимальная частота (с максимальной амплитудой): {max_frequency:.2f} Гц")

        # Вывод формант
        if formants:
            formants_rounded = [round(f) for f in formants]
            print(f"Форманты: {formants_rounded}")
        else:
            print("Форманты: отсутствуют")

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, log_spectro,
                       shading='gouraud',
                       vmin=log_spectro.min(),
                       vmax=log_spectro.max())

        if formants:
            for f in formants:
                plt.axhline(y=f, color='r', linestyle='-', lw=0.5)
            plt.legend(["Форманты"])

        plt.ylim(top=3000)
        plt.yticks(np.arange(0, 3001, 500))
        plt.ylabel('Частота [Гц]')
        plt.xlabel('Время [с]')
        plt.title(f"Спектрограмма звука {label} ({duration:.2f} сек)")

        output_path = f"results/spectrogram_{os.path.basename(filepath)[:-4]}.png"
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Спектрограмма успешно сохранена: {output_path}")

        return True

    except Exception as e:
        print(f"Ошибка при анализе файла: {e}")
        plt.close()
        return False


if __name__ == '__main__':
    os.makedirs("results/wavs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    files_to_analyze = [
        ("src/voice_a.wav", [344, 602, 861, 1119], "А"),
        ("src/voice_i.wav", [344, 602, 1636, 1894], "И"),
        ("src/voice_gav.wav", [], "ГАВ")
    ]

    for file_info in files_to_analyze:
        analyze_audio(*file_info)

    print("\nАнализ завершен")
