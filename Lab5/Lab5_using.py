import cv2
import numpy as np
from PIL import Image
import os

input_image_path = "test_img.png"  # ← путь до фотографии с текстом
output_folder = "extracted_chars"
os.makedirs(output_folder, exist_ok=True)

# 1. Загрузка изображения и бинаризация
img = cv2.imread(input_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. Поиск контуров
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Фильтрация и сортировка по X (слева направо)
bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # по координате X

# 4. Вырезание символов и сохранение
char_images = []
for i, (x, y, w, h) in enumerate(bounding_boxes):
    # Игнорировать слишком маленькие элементы (шум)
    if w < 5 or h < 5:
        continue

    char_img = binary[y:y+h, x:x+w]
    # Увеличить контраст и сохранить как PNG
    pil_img = Image.fromarray(255 - char_img)  # инверсия обратно в белый фон
    pil_img.save(os.path.join(output_folder, f"char_{i+1}.png"))
