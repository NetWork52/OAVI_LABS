import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Загрузка изображения ===
image_color = cv2.imread('Photos/ph5/collage_niblack_k_values.bmp')  # Замени на своё изображение
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)  # Для корректного отображения в matplotlib

# === 2. Перевод в полутоновое ===
image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

# === 3. Структурирующий элемент-диск 5×5 ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# === 4. Морфологическое открытие (полутоновое) ===
opened_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)

# === 5. Разностное изображение ===
diff = cv2.absdiff(image_gray, opened_gray)

# === 6. Морфологическая фильтрация каждого канала цветного изображения ===
channels = cv2.split(image_color)
opened_channels = [cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel) for ch in channels]
image_opened_color = cv2.merge(opened_channels)

# === 7. Коллаж: 5 изображений ===
titles = [
    'Исходное (цветное)',
    'Полутоновое изображение',
    'После открытия (полутон)',
    'Разностное изображение',
    'После открытия (цветное)'
]
images = [image_color, image_gray, opened_gray, diff, image_opened_color]

plt.figure(figsize=(20, 8))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
# Сохранение результатов

cv2.imwrite('Photos/ph5/Results/collage_niblack_k_values_gray.bmp', image_gray)
cv2.imwrite('Photos/ph5/Results/collage_niblack_k_values_opened_gray.bmp', opened_gray)
cv2.imwrite('Photos/ph5/Results/collage_niblack_k_values_diff.bmp', diff)
cv2.imwrite('{Photos/ph5/Results/collage_niblack_k_values_opened_color.bmp', cv2.cvtColor(image_opened_color, cv2.COLOR_RGB2BGR))

# Сохранение коллажа
plt.figure(figsize=(20, 8))
titles = [
    'Исходное (цветное)',
    'Полутоновое',
    'Открытие (полутон)',
    'Разностное',
    'Открытие (цветное)'
]
images = [image_color, image_gray, opened_gray, diff, image_opened_color]
for i in range(5):
    plt.subplot(1, 5, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('00_collage.png', bbox_inches='tight')
plt.close()

