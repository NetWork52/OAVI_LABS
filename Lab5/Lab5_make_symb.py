import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

# Параметры
alphabet = 'аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя'
font_path = "C:/Windows/Fonts/times.ttf"  # Убедись, что файл существует
font_size = 52
output_folder = "symbols"
os.makedirs(output_folder, exist_ok=True)

# Генерация символов
for letter in alphabet:
    img = Image.new("RGB", (100, 100), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Получаем размеры текста
    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Рисуем символ по центру
    draw.text(((100 - w) / 2, (100 - h) / 2), letter, font=font, fill="black")

    # Обрезаем белые поля
    img_gray = img.convert("L")
    img_np = np.array(img_gray)
    coords = np.argwhere(img_np < 255)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped_img = img.crop((x0, y0, x1, y1))
    else:
        cropped_img = img

    # Сохраняем
    cropped_img.save(os.path.join(output_folder, f"{letter}.png"))
