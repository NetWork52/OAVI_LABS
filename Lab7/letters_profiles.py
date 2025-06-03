import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Папка с вырезанными символами
input_dir = "symbols"
output_dir = "letter_profiles"
os.makedirs(output_dir, exist_ok=True)

# Перебираем все BMP-файлы в папке
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".bmp"):
        filepath = os.path.join(input_dir, filename)

        # Загружаем и преобразуем в бинарное изображение
        img = Image.open(filepath).convert("1")
        img_np = np.array(img)
        binary = 1 - img_np // 255  # чёрное = 1, белое = 0

        # Горизонтальный профиль
        h_profile = np.sum(binary, axis=1)
        # Вертикальный профиль
        v_profile = np.sum(binary, axis=0)

        # Построение и сохранение графиков
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].bar(range(len(h_profile)), h_profile, color='black')
        axs[0].set_title("Горизонтальный профиль")
        axs[0].set_xlabel("Строка")
        axs[0].set_ylabel("Чёрные пиксели")

        axs[1].bar(range(len(v_profile)), v_profile, color='black')
        axs[1].set_title("Вертикальный профиль")
        axs[1].set_xlabel("Столбец")
        axs[1].set_ylabel("Чёрные пиксели")

        plt.suptitle(filename)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename[:-4]}_profile.png"))
        plt.close()

print("Профили для всех символов сохранены в папке 'letter_profiles'.")
