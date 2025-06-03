import os
import cv2
import numpy as np

# Пути к папкам
input_dir = "segmented_letters"
output_dir = "cut_seg_mono_let"

# Создаем выходную папку, если не существует
os.makedirs(output_dir, exist_ok=True)

# Получаем список всех изображений
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))]

for idx, filename in enumerate(image_files, start=1):
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Преобразуем в бинарное изображение (черно-белое)
    _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

    # Инвертируем изображение: буквы станут белыми (255), фон черным (0)
    inverted = 255 - binary

    # Находим контуры
    coords = cv2.findNonZero(inverted)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y+h, x:x+w]
    else:
        # если символа нет, оставляем оригинал
        cropped = image

    # Преобразуем обрезанное изображение в строго монохромное (0 или 255)
    _, mono = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY)

    # Сохраняем изображение в BMP формате
    char_label = os.path.splitext(filename)[0].split('_')[0]
    out_filename = f"{char_label}_{idx}.bmp"
    cv2.imwrite(os.path.join(output_dir, out_filename), mono)

print(f"Готово! Изображения сохранены в папке {output_dir}.")
