import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

input_dir = "../Photo/alphabet"
output_dir = "../Photo/alphabet_phofs"
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".bmp"):
        filepath = os.path.join(input_dir, filename)

        img = Image.open(filepath).convert("1")
        binary = np.array(img, dtype=np.uint8)
        binary = 1 - binary  # чёрные пиксели = 1

        h_profile = np.sum(binary, axis=1)  # по строкам (Y)
        v_profile = np.sum(binary, axis=0)  # по столбцам (X)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Горизонтальный профиль: вдоль строки, направо (ось X)
        axs[0].barh(range(len(h_profile)), h_profile, color='black', height=1.0)
        axs[0].set_title("Горизонтальный профиль")
        axs[0].set_xlabel("Чёрные пиксели")
        axs[0].set_ylabel("Номер строки")
        axs[0].invert_yaxis()  # Верх строки — вверху графика
        axs[0].grid(True)

        # Вертикальный профиль: вдоль столбцов, вверх (ось Y)
        axs[1].bar(range(len(v_profile)), v_profile, color='black', width=1.0)
        axs[1].set_title("Вертикальный профиль")
        axs[1].set_xlabel("Номер столбца")
        axs[1].set_ylabel("Чёрные пиксели")
        axs[1].grid(True)

        plt.suptitle(filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{filename[:-4]}_profile.png"))
        plt.close()

print("Профили для всех символов сохранены в папке 'letter_profiles'.")
