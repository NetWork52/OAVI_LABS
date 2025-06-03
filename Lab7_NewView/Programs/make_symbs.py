import os
import numpy as np
from PIL import Image
import cv2

# Параметры
image_path = "../Photo/mono&cut/alph1.bmp"
output_dir = "../Photo/alphabet"
os.makedirs(output_dir, exist_ok=True)

# Загрузка и инвертирование
img = Image.open(image_path).convert("L")
img_np = np.array(img)
_, binary = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)

# Контуры и сортировка
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = [cv2.boundingRect(cnt) for cnt in contours]
boxes = sorted(boxes, key=lambda b: b[0])
used = [False] * len(boxes)

symbols = []

i = 0
while i < len(boxes):
    if used[i]:
        i += 1
        continue

    x1, y1, w1, h1 = boxes[i]
    box = [x1, y1, x1 + w1, y1 + h1]
    used[i] = True

    # Проверка на точку сверху (для й, ё)
    for j in range(len(boxes)):
        if used[j] or j == i:
            continue
        x2, y2, w2, h2 = boxes[j]
        if y2 + h2 < y1 and abs(x2 - x1) < w1 and (y1 - (y2 + h2)) < 15:
            box[0] = min(box[0], x2)
            box[1] = min(box[1], y2)
            box[2] = max(box[2], x2 + w2)
            box[3] = max(box[3], y2 + h2)
            used[j] = True

    # Проверка на "ы" — узкий штрих справа на той же высоте
    for j in range(len(boxes)):
        if used[j] or j == i:
            continue
        x2, y2, w2, h2 = boxes[j]
        horizontal_gap = x2 - (x1 + w1)
        height_similar = abs(h1 - h2) <= 5
        same_y = abs(y1 - y2) <= 5
        is_narrow = w2 <= 4  # как у узкого штриха
        is_close = 0 < horizontal_gap <= 10
        if is_narrow and is_close and height_similar and same_y:
            # Считаем "ы"
            box[0] = min(box[0], x2)
            box[1] = min(box[1], y2)
            box[2] = max(box[2], x2 + w2)
            box[3] = max(box[3], y2 + h2)
            used[j] = True
            break  # объединяем только один штрих

    symbols.append(box)
    i += 1

# Сортировка и сохранение
symbols.sort(key=lambda b: b[0])
original_img = Image.open(image_path).convert("1")

for idx, (x1, y1, x2, y2) in enumerate(symbols):
    char_img = original_img.crop((x1, y1, x2, y2))
    char_img.save(os.path.join(output_dir, f"char_{idx+1}.bmp"))

print(f"Сегментировано и сохранено символов: {len(symbols)}")
