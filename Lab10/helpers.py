import numpy as np
from scipy.io import wavfile
from scipy import signal
import os


def change_sample_rate(path, new_sample_rate=44100):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Исходный файл {path} не найден")

        old_sample_rate, old_audio = wavfile.read(path)

        # Ресемплинг
        new_audio = signal.resample(
            old_audio,
            int(len(old_audio) * new_sample_rate / old_sample_rate)
        )

        # Создаем директории, если их нет
        os.makedirs("results/wavs", exist_ok=True)

        # Сохраняем результат
        output_path = f"results/wavs/{os.path.basename(path)}"
        wavfile.write(output_path, new_sample_rate, np.round(new_audio).astype(old_audio.dtype))

        return output_path

    except Exception as e:
        print(f"Ошибка в change_sample_rate: {str(e)}")
        raise

# Другие функции helpers.py...