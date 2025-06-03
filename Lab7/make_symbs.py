import os
import numpy as np
from PIL import Image

# Параметры
image_path = "cropped_mono.bmp"
output_dir = "segmented_letters"
profile_threshold = 3  # минимальное количество чёрных пикселей, ниже которого считаем границей
min_width = 1          # минимальная ширина символа, чтобы избежать шумов

# Создание папки для сохранения сегментов
os.makedirs(output_dir, exist_ok=True)

# Загрузка изображения
img = Image.open(image_path).convert("1")
img_np = np.array(img)  # булев массив: True — белый, False — чёрный
binary = (~img_np).astype(int)  # чёрное = 1, белое = 0

# Вертикальный профиль
vertical_profile = np.sum(binary, axis=0)

# Отладочный вывод
print(vertical_profile)
print(f"Минимальное значение: {vertical_profile.min()}, максимальное: {vertical_profile.max()}")
print(np.unique(img_np))

# Поиск границ символов
segments = []
inside_char = False
start = 0

for i, val in enumerate(vertical_profile):
    if not inside_char and val > profile_threshold:
        start = i
        inside_char = True
    elif inside_char and val <= profile_threshold:
        end = i
        if end - start >= min_width:
            segments.append((start, end))
        inside_char = False

# Вырезаем и сохраняем сегменты
for idx, (start_x, end_x) in enumerate(segments):
    char_img = img.crop((start_x, 0, end_x, img.height))
    char_img.save(os.path.join(output_dir, f"char_{idx+1}.bmp"))

print(f"Сегментировано и сохранено символов: {len(segments)}")
